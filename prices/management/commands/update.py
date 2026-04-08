import xgboost as xg
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from django.core.cache import cache  # Or store in the database if needed

import numpy as np
import os
import logging

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData, OfficialAgileData, CommodityPriceHistory

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28
MAX_TEST_X = 20000

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "update.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def lighten_cmap(cmap_name="viridis", amount=0.5):
    base = cm.get_cmap(cmap_name)
    cdict = base._segmentdata if hasattr(base, "_segmentdata") else None
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_light",
        [
            (mcolors.to_rgba(c, alpha=1)[:3] + np.array([amount] * 3)) / (1 + amount)
            for c in base(np.linspace(0, 1, 256))
        ],
    )


def kde_quantiles(kde, dt, pred, quantiles={"low": 0.1, "mid": 0.5, "high": 0.9}, lim=(0, 150)):
    if not isinstance(dt, list):
        dt = [dt]
    if not isinstance(pred, list):
        pred = [pred]

    results = {q: [] for q in quantiles}
    for dt1, pred1 in zip(dt, pred):
        x = np.array([[dt1, pred1, p] for p in range(int(lim[0]), int(lim[1]))])
        c = pd.Series(index=x[:, 2], data=np.exp(kde.score_samples(x)).cumsum())
        c /= c.iloc[-1]

        for q in quantiles:
            if len(c[c < quantiles[q]]) > 0:
                idx = c[c < quantiles[q]].index[-1]
                results[q] += [(quantiles[q] - c[idx]) / (c[idx + 1] - c[idx]) + idx]
            else:
                results[q] += [np.nan]
    return results


class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        # parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--debug",
            action="store_true",
        )

        # --min_fd and --min_ad removed: new smart cleanup doesn't need them

        parser.add_argument(
            "--max_days",
        )

        parser.add_argument(
            "--no_day_of_week",
            action="store_true",
        )

        parser.add_argument(
            "--train_frac",
        )

        parser.add_argument(
            "--drop_last",
        )

        parser.add_argument(
            "--ignore_forecast",
            action="append",
        )

        parser.add_argument(
            "--no_ranges",
            action="store_true",
        )

        parser.add_argument(
            "--vol_threshold",
            type=float,
            default=1.5,
            help="Volatility ratio threshold for adaptive window (default: 1.5)",
        )

        parser.add_argument(
            "--short_window",
            type=int,
            default=60,
            help="Short training window in days when high volatility detected (default: 60)",
        )

        parser.add_argument(
            "--spike_threshold",
            type=float,
            default=2000,
            help="Derated margin threshold MW below which spike risk is high (default: 2000)",
        )

        parser.add_argument(
            "--comfortable_margin",
            type=float,
            default=4000,
            help="Derated margin MW above which spike weight is 0 (default: 4000)",
        )

        # --bootstrap removed: new smart cleanup strategy is inherently safe for cold start

    def handle(self, *args, **options):
        # Setup logging

        debug = options.get("debug", False)

        max_days = int(options.get("max_days", 3650) or 3650)

        no_ranges = options.get("no_ranges", False)

        drop_cols = ["emb_wind"]
        if options.get("no_day_of_week", False):
            drop_cols += ["day_of_week"]

        drop_last = int(options.get("drop_last", 0) or 0)

        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        # Cleanup runs AFTER prediction+save (see end of handle) to ensure
        # the current run's output exists before any deletion.

        # Fetch commodity prices (gas + carbon) once at the start
        commodity_prices = fetch_commodity_prices()
        logger.info(
            "Commodity prices: gas=%.2f p/therm, carbon=%.2f EUR/tonne",
            commodity_prices["gas_price_ptherm"],
            commodity_prices["carbon_price_eur"],
        )

        # Store today's commodity prices in DB for historical accumulation
        import datetime as _dt
        today = _dt.date.today()
        CommodityPriceHistory.objects.update_or_create(
            date=today,
            defaults={
                "gas_price_ptherm": commodity_prices["gas_price_ptherm"],
                "carbon_price_eur": commodity_prices["carbon_price_eur"],
            },
        )

        prices, start = model_to_df(PriceHistory)

        if debug:
            logger.info("Getting Historic Prices")
            logger.info(f"Prices\n{prices}")

        # Pages arrive newest-first (descending); filter by original start so we
        # don't re-write records already in the DB, but don't gate on prices.index[-1]
        # (that would discard older pages as soon as one page is written).
        all_agile_pages = []
        try:
            for page_agile in get_agile_pages(start=start, region="F"):
                all_agile_pages.append(page_agile)
                page_day_ahead = day_ahead_to_agile(page_agile, reverse=True, region="F")
                page_new = pd.concat([page_day_ahead, page_agile], axis=1)
                page_new = page_new[page_new.index >= start]
                if len(page_new) > 0:
                    if debug:
                        logger.info(f"New Prices (page)\n{page_new}")
                    logger.info(
                        "Writing %d new price records to DB (up to %s)",
                        len(page_new),
                        page_new.index[-1],
                    )
                    df_to_Model(page_new, PriceHistory, update=True)
                    prices = pd.concat([prices, page_new]).sort_index()
        except Exception as e:
            logger.warning("Octopus Agile price fetch failed: %s — continuing with existing PriceHistory", e)

        agile = (
            pd.concat(all_agile_pages).sort_index()
            if all_agile_pages
            else pd.Series(name="agile", dtype=float)
        )
        agile = agile[~agile.index.duplicated(keep="last")]
        day_ahead = day_ahead_to_agile(agile, reverse=True, region="F")
        prices = prices[~prices.index.duplicated(keep="last")]

        agile_end = prices.index[-1]
        gb60 = get_gb60()

        if debug:
            logger.info(f"GB60:\n{gb60}")

        if gb60 is None:
            logger.warning("GB60: Nord Pool API unavailable, continuing without day-ahead wholesale prices")
            gb60 = pd.Series(dtype=float)

        gb60 = gb60.resample("30min").ffill().loc[agile_end + pd.Timedelta("30min") :]

        if len(gb60) > 0:
            gb60 = gb60.reindex(
                pd.date_range(gb60.index[0], gb60.index[-1] + pd.Timedelta("30min"), freq="30min")
            ).ffill()
            gb60 = pd.concat([gb60, day_ahead_to_agile(gb60)], axis=1).set_axis(
                ["day_ahead", "agile"], axis=1
            )
            prices = pd.concat([prices, gb60]).sort_index()

        # --- Fetch day-ahead auction results from Elexon MID (Req 9) ---
        # Fetch for next 2 days — auction results arrive after ~12:00 UK time
        mid_from = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
        mid_to = (pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        mid_auction = fetch_day_ahead_auction(mid_from, mid_to)

        # Only use auction results for slots not already covered by actual Agile prices
        if len(mid_auction) > 0:
            mid_auction.index = mid_auction.index.tz_convert("GB")
            mid_auction = mid_auction[mid_auction.index > agile_end]

        if len(mid_auction) > 0:
            # Build a DataFrame matching the prices structure (day_ahead + agile)
            mid_prices = pd.DataFrame({"day_ahead": mid_auction})
            mid_prices["agile"] = day_ahead_to_agile(mid_auction, region="F")
            # Only add slots not already in prices (don't overwrite Agile or GB60)
            new_mid_slots = mid_prices.index.difference(prices.index)
            if len(new_mid_slots) > 0:
                mid_prices = mid_prices.loc[new_mid_slots]
                prices = pd.concat([prices, mid_prices]).sort_index()
                logger.info(
                    "MID auction: added %d slots to prices (%s to %s)",
                    len(mid_prices), mid_prices.index[0], mid_prices.index[-1],
                )
            else:
                logger.info("MID auction: %d records fetched but all slots already covered by Agile/GB60", len(mid_auction))
                mid_auction = pd.Series(dtype=float, name="day_ahead_auction")  # reset to empty
        else:
            logger.info("MID auction: no data available (auction may not have run yet)")

        if debug:
            logger.info(f"Merged prices:\n{prices}")

        if drop_last > 0:
            logger.info(f"drop_last: {drop_last}")
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")
            prices = prices.iloc[:-drop_last]
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")

        # --- Adaptive training window based on price volatility (Req 6) ---
        # Compute volatility from daily mean day-ahead prices
        daily_mean_prices = prices["day_ahead"].dropna().resample("D").mean().dropna()
        vol_30d = daily_mean_prices.tail(30).std() if len(daily_mean_prices) >= 2 else 0.0
        vol_180d = daily_mean_prices.tail(180).std() if len(daily_mean_prices) >= 2 else 0.0
        volatility_ratio = vol_30d / vol_180d if vol_180d > 0 else 1.0

        logger.info(
            "Volatility metrics: vol_30d=%.4f, vol_180d=%.4f, ratio=%.4f",
            vol_30d, vol_180d, volatility_ratio,
        )

        VOLATILITY_THRESHOLD = float(options.get("vol_threshold", 1.5) or 1.5)
        SHORT_WINDOW_DAYS = int(options.get("short_window", 60) or 60)

        if volatility_ratio > VOLATILITY_THRESHOLD:
            logger.info(
                "High volatility detected (ratio=%.2f > %.2f), using %d-day window",
                volatility_ratio, VOLATILITY_THRESHOLD, SHORT_WINDOW_DAYS,
            )
            max_days = SHORT_WINDOW_DAYS
        else:
            logger.info(
                "Normal volatility (ratio=%.2f <= %.2f), using configured window (%d days)",
                volatility_ratio, VOLATILITY_THRESHOLD, max_days,
            )

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.exclude(id__in=ignore_forecast).order_by(
                "-created_at"
            )
            last_forecasts = {
                forecast.created_at.date(): forecast.id
                for forecast in base_forecasts.order_by("created_at")
            }

            base_forecasts = base_forecasts.filter(
                id__in=[last_forecasts[k] for k in last_forecasts]
            )

            if debug:
                logger.info("Getting latest Forecast")

            fc, missing_fc = get_latest_forecast()

            if len(missing_fc) > 0:
                logger.error(
                    f">>> ERROR: Unable to run forecast due to missing columns: {', '.join(missing_fc)}"
                )
            else:
                if debug:
                    logger.info(fc)

                if len(fc) > 0:
                    fd = pd.DataFrame(
                        list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values())
                    )
                    ff = pd.DataFrame(
                        list(Forecasts.objects.exclude(id__in=ignore_forecast).values())
                    )
                    scores = []  # initialise so line 707 ref is safe if training block is skipped (e.g. first run)
                    elexon_features_available = False  # track whether Elexon API data was fetched successfully
                    interconnector_available = False  # track whether interconnector API data was fetched successfully
                    margin_available = False  # track whether derated margin API data was fetched successfully
                    french_nuclear_available = False  # track whether French nuclear data was fetched successfully
                    conformal_offsets = {}  # conformal prediction offsets per horizon (Req 10)
                    cal_coverage_rate = None  # conformal calibration coverage rate (Req 10)

                    if len(ff) > 0:
                        logger.info(ff)
                        ff = ff.set_index("id").sort_index()
                        ff["created_at"] = pd.to_datetime(ff["name"]).dt.tz_localize("GB")
                        ff["date"] = ff["created_at"].dt.tz_convert("GB").dt.normalize()
                        ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
                        ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)

                        # Only train on the forecasts closest to 16:15
                        ff["dt1600"] = (
                            (
                                ff["date"]
                                + pd.Timedelta(hours=16, minutes=15)
                                - ff["created_at"].dt.tz_convert("GB")
                            )
                            .dt.total_seconds()
                            .abs()
                        )
                        ff_train = (
                            ff.sort_values("dt1600")
                            .drop_duplicates("date")
                            .sort_index()
                            .drop(["date", "dt1600"], axis=1)
                        )

                        if debug:
                            logger.info(f"Forecasts Database:\n{ff.to_string()}")

                        # df is the full dataset — keep date_time as a column (RangeIndex)
                        # to avoid duplicate-index bugs when multiple forecasts
                        # predict the same settlement period.
                        df = (
                            (fd.merge(ff, right_index=True, left_on="forecast_id"))
                            .drop("day_ahead", axis=1)
                        )
                        df["date_time"] = pd.to_datetime(df["date_time"], utc=True)

                        df["dow"] = df["date_time"].dt.day_of_week
                        df["weekend"] = (df["date_time"].dt.day_of_week >= 5).astype(int)
                        df["time"] = df["date_time"].dt.tz_convert("GB").dt.hour + df["date_time"].dt.tz_convert("GB").dt.minute / 60
                        df["days_ago"] = (
                            (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds()
                            / 3600
                            / 24
                        )
                        df["dt"] = (df["date_time"] - df["created_at"]).dt.total_seconds() / 3600 / 24
                        df["peak"] = ((df["time"] >= 16) & (df["time"] < 19)).astype(float)

                        # Cyclical time encoding for seasonal/diurnal patterns
                        time_gb = df["date_time"].dt.tz_convert("GB")
                        hour = time_gb.dt.hour + time_gb.dt.minute / 60
                        month = time_gb.dt.month
                        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                        df["month_sin"] = np.sin(2 * np.pi * month / 12)
                        df["month_cos"] = np.cos(2 * np.pi * month / 12)

                        # --- Lag price features (from PriceHistory) ---
                        price_series = prices["day_ahead"].sort_index()
                        price_mean = price_series.mean() if len(price_series) > 0 else 0.0

                        lag_features = pd.DataFrame(index=price_series.index)
                        lag_features["price_lag_1"] = price_series.shift(1)
                        lag_features["price_lag_2"] = price_series.shift(2)
                        lag_features["price_lag_3"] = price_series.shift(3)
                        lag_features["price_lag_4"] = price_series.shift(4)
                        lag_features["price_lag_6"] = price_series.shift(6)
                        lag_features["price_lag_12"] = price_series.shift(12)
                        lag_features["price_lag_24"] = price_series.shift(24)
                        lag_features["price_lag_48"] = price_series.shift(48)
                        lag_features["price_lag_336"] = price_series.shift(336)
                        lag_features["price_rolling_mean_48"] = price_series.rolling(48).mean()
                        lag_features = lag_features.fillna(price_mean)

                        for col in lag_features.columns:
                            df[col] = df["date_time"].map(lag_features[col]).fillna(price_mean)

                        # --- Price volatility features ---
                        vol_7d = price_series.rolling(7 * 48, min_periods=48).std()
                        vol_30d = price_series.rolling(30 * 48, min_periods=48).std()

                        if len(price_series) < 48:
                            logger.warning("Fewer than 48 price history records — volatility features set to 0.0")
                            vol_7d = vol_7d.fillna(0.0)
                            vol_30d = vol_30d.fillna(0.0)

                        df["price_volatility_7d"] = df["date_time"].map(vol_7d).ffill().fillna(0.0)
                        df["price_volatility_30d"] = df["date_time"].map(vol_30d).ffill().fillna(0.0)

                        # --- Residual demand feature ---
                        df["residual_demand"] = df["demand"] - df["bm_wind"] - df["solar"]

                        # --- Renewable penetration feature (Req 3 - Round 2) ---
                        df["renewable_penetration"] = np.where(
                            df["demand"] > 0,
                            (df["bm_wind"] + df["solar"]) / df["demand"] * 100,
                            0.0,
                        )

                        # --- Commodity price features (Req 2) ---
                        # Use historical daily prices from DB when available, fall back to today's price
                        commodity_qs = CommodityPriceHistory.objects.all()
                        if commodity_qs.exists():
                            commodity_hist = pd.DataFrame(list(commodity_qs.values("date", "gas_price_ptherm", "carbon_price_eur")))
                            commodity_hist.index = pd.to_datetime(commodity_hist["date"])
                            commodity_hist = commodity_hist.drop("date", axis=1).sort_index()
                            # Merge by date: each training sample gets the commodity price from its target date
                            df_dates = df["date_time"].dt.normalize()
                            df["gas_price_ptherm"] = df_dates.map(commodity_hist["gas_price_ptherm"]).ffill().bfill()
                            df["carbon_price_eur"] = df_dates.map(commodity_hist["carbon_price_eur"]).ffill().bfill()
                            # Fill any remaining NaN (dates outside DB range) with today's price
                            df["gas_price_ptherm"] = df["gas_price_ptherm"].fillna(commodity_prices["gas_price_ptherm"])
                            df["carbon_price_eur"] = df["carbon_price_eur"].fillna(commodity_prices["carbon_price_eur"])
                            logger.info("Commodity features: merged %d days of historical prices from DB", len(commodity_hist))
                        else:
                            # No history yet — use today's price as constant (will accumulate over time)
                            df["gas_price_ptherm"] = commodity_prices["gas_price_ptherm"]
                            df["carbon_price_eur"] = commodity_prices["carbon_price_eur"]
                            logger.info("Commodity features: using today's price as constant (no history in DB yet)")

                        # --- Elexon system prices and CCGT generation (Req 2) ---
                        # Fetch last 30 days only (not full training range — too many API calls)
                        elexon_from = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                        elexon_to = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
                        elexon_features_available = False

                        sys_prices_df = fetch_system_prices(elexon_from, elexon_to)
                        ccgt_gen_df = fetch_ccgt_generation(elexon_from, elexon_to)

                        # Also fetch interconnector flows (Req 3)
                        interconnector_df = fetch_interconnector_flows(elexon_from, elexon_to)
                        interconnector_available = False

                        if len(sys_prices_df) > 0 and len(ccgt_gen_df) > 0:
                            # Merge system prices into training data
                            df["system_buy_price"] = df["date_time"].map(sys_prices_df["system_buy_price"]).ffill().bfill()
                            # Merge CCGT generation into training data
                            df["ccgt_generation_mw"] = df["date_time"].map(ccgt_gen_df["ccgt_generation_mw"]).ffill().bfill()
                            # Compute derived ccgt_share feature
                            df["ccgt_share"] = df["ccgt_generation_mw"] / df["demand"].replace(0, float("nan")) * 100
                            df["ccgt_share"] = df["ccgt_share"].fillna(0.0)

                            # Check if we have enough non-NaN values to be useful
                            if df["system_buy_price"].notna().sum() > 0 and df["ccgt_generation_mw"].notna().sum() > 0:
                                df["system_buy_price"] = df["system_buy_price"].fillna(0.0)
                                df["ccgt_generation_mw"] = df["ccgt_generation_mw"].fillna(0.0)
                                elexon_features_available = True
                                logger.info("Elexon features: system_buy_price, ccgt_generation_mw, ccgt_share added to training data")
                            else:
                                df.drop(columns=["system_buy_price", "ccgt_generation_mw", "ccgt_share"], inplace=True, errors="ignore")
                                logger.warning("Elexon features: insufficient data after merge, omitting from this run")
                        else:
                            logger.warning("Elexon features: API returned empty data, omitting from this run")

                        # --- Interconnector flows (Req 3) ---
                        if len(interconnector_df) > 0:
                            df["net_interconnector_mw"] = df["date_time"].map(interconnector_df["net_interconnector_mw"]).ffill().bfill()
                            if df["net_interconnector_mw"].notna().sum() > 0:
                                df["net_interconnector_mw"] = df["net_interconnector_mw"].fillna(0.0)
                                interconnector_available = True
                                logger.info("Interconnector feature: net_interconnector_mw added to training data")
                            else:
                                df.drop(columns=["net_interconnector_mw"], inplace=True, errors="ignore")
                                logger.warning("Interconnector feature: insufficient data after merge, omitting from this run")
                        else:
                            logger.warning("Interconnector feature: API returned empty data, omitting from this run")

                        # --- Derated margin (Req 1 - Round 2) ---
                        margin_df = fetch_derated_margin(elexon_from, elexon_to)
                        margin_available = False

                        if len(margin_df) > 0:
                            df["derated_margin_mw"] = df["date_time"].map(margin_df["derated_margin_mw"]).ffill().bfill()
                            df["margin_nearest_mw"] = df["date_time"].map(margin_df["margin_nearest_mw"]).ffill().bfill()
                            if df["derated_margin_mw"].notna().sum() > 0:
                                df["derated_margin_mw"] = df["derated_margin_mw"].fillna(0.0)
                                df["margin_nearest_mw"] = df["margin_nearest_mw"].fillna(0.0)
                                margin_available = True
                                logger.info("Derated margin features: derated_margin_mw, margin_nearest_mw added to training data")
                            else:
                                df.drop(columns=["derated_margin_mw", "margin_nearest_mw"], inplace=True, errors="ignore")
                                logger.warning("Derated margin features: insufficient data after merge, omitting from this run")
                        else:
                            logger.warning("Derated margin features: API returned empty data, omitting from this run")

                        # --- French nuclear availability (Req 4 - Round 2) ---
                        french_nuclear_gw = fetch_french_nuclear()
                        french_nuclear_available = french_nuclear_gw is not None

                        if french_nuclear_available:
                            df["french_nuclear_gw"] = french_nuclear_gw  # scalar, same value for all rows
                            logger.info("French nuclear feature: %.2f GW", french_nuclear_gw)

                        features = [
                            "bm_wind",
                            "solar",
                            "demand",
                            "peak",
                            "days_ago",
                            "wind_10m",
                            "weekend",
                            "hour_sin",
                            "hour_cos",
                            "month_sin",
                            "month_cos",
                            # Lag price features (Req 1 + Req 2 Round 2)
                            "price_lag_1",
                            "price_lag_2",
                            "price_lag_3",
                            "price_lag_4",
                            "price_lag_6",
                            "price_lag_12",
                            "price_lag_24",
                            "price_lag_48",
                            "price_lag_336",
                            "price_rolling_mean_48",
                            # Price volatility features (Req 7)
                            "price_volatility_7d",
                            "price_volatility_30d",
                            # Residual demand (Req 8)
                            "residual_demand",
                            # Renewable penetration (Req 3 Round 2)
                            "renewable_penetration",
                            # Commodity prices (Req 2)
                            "gas_price_ptherm",
                            "carbon_price_eur",
                        ]

                        # Conditionally add Elexon features if API data was available
                        if elexon_features_available:
                            features.extend([
                                "system_buy_price",
                                "ccgt_generation_mw",
                                "ccgt_share",
                            ])

                        # Conditionally add interconnector feature (Req 3)
                        if interconnector_available:
                            features.append("net_interconnector_mw")

                        # Conditionally add derated margin features (Req 1 - Round 2)
                        if margin_available:
                            features.extend(["derated_margin_mw", "margin_nearest_mw"])

                        # Conditionally add French nuclear feature (Req 4 - Round 2)
                        if french_nuclear_available:
                            features.append("french_nuclear_gw")

                        # Only use the forecasts closest to 16:15 for training
                        train_X = df[df["forecast_id"].isin(ff_train.index)]
                        train_X = train_X[train_X["days_ago"] < max_days]

                        # Only train on the next agile prices that are set from the pm auction
                        train_X = train_X[
                            (train_X["date_time"] >= train_X["ag_start"])
                            & (train_X["date_time"] < train_X["ag_end"])
                        ]

                        # --- Carry dt through the merge so it stays aligned (Req 5) ---
                        # dt is in df but not in features list; we need it for horizon splitting
                        train_X = train_X[features + ["date_time", "dt"]]

                        # Get the prices to match the forecast — column-based merge on date_time
                        train_X = train_X.merge(
                            prices["day_ahead"].rename_axis("date_time").reset_index(),
                            on="date_time",
                            how="inner",
                        )
                        # Drop date_time now that merge is done
                        train_X = train_X.drop(columns=["date_time"])
                        # Extract dt after merge so it stays aligned with train_X
                        train_dt = train_X.pop("dt")

                        if debug:
                            logger.info(f"train_X:\n{train_X}")

                        train_y = train_X.pop("day_ahead")
                        recency_weight = np.exp(
                            -train_X["days_ago"].values * np.log(2) / 180
                        )
                        sample_weights = (
                            ((np.log10((train_y - train_y.mean()).abs() + 10) * 5) - 4)
                            * recency_weight
                        )

                        xg_model = xg.XGBRegressor(
                            objective="reg:squarederror",
                            booster="gbtree",
                            learning_rate=0.0135,
                            max_depth=8,
                            subsample=0.775,
                            colsample_bytree=0.604,
                            n_estimators=150,
                            gamma=0.093,
                            min_child_weight=4,
                            reg_alpha=0.003,
                            reg_lambda=0.0095,
                        )

                        # --- LightGBM model (Req 4) ---
                        lgb_model = lgb.LGBMRegressor(
                            objective="regression",
                            learning_rate=0.015,
                            max_depth=8,
                            subsample=0.8,
                            colsample_bytree=0.6,
                            n_estimators=150,
                            min_child_weight=4,
                            reg_alpha=0.003,
                            reg_lambda=0.01,
                            verbose=-1,
                        )

                        MAX_CV_SAMPLES = 10_000
                        n_cv = min(5, len(train_X) // 2)
                        if n_cv >= 2:
                            if len(train_X) > MAX_CV_SAMPLES:
                                cv_idx = np.random.default_rng(42).choice(
                                    len(train_X), MAX_CV_SAMPLES, replace=False
                                )
                                cv_X = train_X.iloc[cv_idx]
                                cv_y = train_y.iloc[cv_idx]
                            else:
                                cv_X, cv_y = train_X, train_y

                            # XGBoost cross-validation
                            xg_scores = cross_val_score(
                                xg_model,
                                cv_X,
                                cv_y,
                                cv=n_cv,
                                scoring="neg_root_mean_squared_error",
                            )
                            logger.info(f"XGBoost cross-val score: {xg_scores}")

                            # LightGBM cross-validation
                            lgb_scores = cross_val_score(
                                lgb_model,
                                cv_X,
                                cv_y,
                                cv=n_cv,
                                scoring="neg_root_mean_squared_error",
                            )
                            logger.info(f"LightGBM cross-val score: {lgb_scores}")

                            # Inverse-RMSE ensemble weights (Req 4)
                            xg_weight = 1.0 / abs(xg_scores.mean())
                            lgb_weight = 1.0 / abs(lgb_scores.mean())
                            total_weight = xg_weight + lgb_weight
                            xg_weight /= total_weight
                            lgb_weight /= total_weight
                            logger.info(
                                "Ensemble weights: XGBoost=%.3f, LightGBM=%.3f",
                                xg_weight, lgb_weight,
                            )

                            # Ensemble scores for the Forecasts model mean/stdev
                            scores = xg_weight * xg_scores + lgb_weight * lgb_scores
                            logger.info(f"Ensemble cross-val score: {scores}")

                            # --- Stacking Meta-Model (Req 5 - Round 2) ---
                            # Task 7.1: Ridge base model + out-of-fold predictions
                            ridge_model = Ridge(alpha=1.0)
                            meta_learner_available = False

                            try:
                                oof_xg = cross_val_predict(xg_model, cv_X, cv_y, cv=n_cv)
                                oof_lgb = cross_val_predict(lgb_model, cv_X, cv_y, cv=n_cv)
                                oof_ridge = cross_val_predict(ridge_model, cv_X, cv_y, cv=n_cv)
                                logger.info(
                                    "Out-of-fold predictions: XGBoost=%d, LightGBM=%d, Ridge=%d samples",
                                    len(oof_xg), len(oof_lgb), len(oof_ridge),
                                )

                                # Task 7.2: Train Ridge meta-learner on stacked OOF predictions
                                meta_X = np.column_stack([oof_xg, oof_lgb, oof_ridge])
                                meta_learner = Ridge(alpha=0.1, fit_intercept=True)
                                meta_learner.fit(meta_X, cv_y)
                                logger.info(
                                    "Meta-learner coefficients: XGBoost=%.3f, LightGBM=%.3f, Ridge=%.3f, intercept=%.3f",
                                    meta_learner.coef_[0], meta_learner.coef_[1], meta_learner.coef_[2],
                                    meta_learner.intercept_,
                                )

                                # Update ensemble scores to reflect meta-learner performance
                                meta_oof_pred = meta_learner.predict(meta_X)
                                meta_rmse = np.sqrt(MSE(cv_y, meta_oof_pred))
                                logger.info("Meta-learner OOF RMSE: %.3f", meta_rmse)
                                # Preserve original ensemble scores for stdev, but scale mean to meta-learner RMSE
                                original_mean = abs(scores.mean())
                                if original_mean > 0:
                                    scores = scores * (meta_rmse / original_mean)

                                meta_learner_available = True
                                logger.info("Stacking meta-learner: ACTIVE (replacing inverse-RMSE for combined model)")

                            except Exception as e:
                                # Task 7.4: Fallback to inverse-RMSE ensemble
                                logger.warning(
                                    "Stacking meta-learner: cross_val_predict failed (%s), "
                                    "falling back to inverse-RMSE ensemble",
                                    e,
                                )
                                meta_learner_available = False
                        else:
                            xg_scores = np.array([0.0])
                            lgb_scores = np.array([0.0])
                            xg_weight = 0.5
                            lgb_weight = 0.5
                            scores = np.array([0.0])
                            ridge_model = Ridge(alpha=1.0)
                            meta_learner_available = False
                            logger.info(
                                "Too few training samples for cross-validation (n=%d) — skipping cv and meta-learner",
                                len(train_X),
                            )

                        # --- Conformal Prediction Calibration (Req 10) ---
                        # Task 6.1: Chronological calibration split
                        # Train on 80% oldest data, calibrate on 20% most recent
                        # Then retrain on full data for final predictions
                        cal_split = int(len(train_X) * 0.8)
                        conformal_offsets = {}

                        if cal_split >= 50 and (len(train_X) - cal_split) >= 20:
                            fit_X = train_X.iloc[:cal_split]
                            fit_y = train_y.iloc[:cal_split]
                            fit_weights = sample_weights[:cal_split]
                            cal_X = train_X.iloc[cal_split:]
                            cal_y = train_y.iloc[cal_split:]
                            cal_dt = train_dt.iloc[cal_split:]

                            # Log calibration set info
                            logger.info(
                                "Conformal calibration: fit=%d samples, cal=%d samples (%.0f%%/%.0f%%)",
                                len(fit_X), len(cal_X),
                                100 * len(fit_X) / len(train_X),
                                100 * len(cal_X) / len(train_X),
                            )
                            logger.info(
                                "Conformal calibration set: split at row %d of %d",
                                cal_split, len(train_X),
                            )

                            # Train calibration models on the 80% fit set
                            cal_xg = xg.XGBRegressor(
                                objective="reg:squarederror",
                                booster="gbtree",
                                learning_rate=0.0135,
                                max_depth=8,
                                subsample=0.775,
                                colsample_bytree=0.604,
                                n_estimators=150,
                                gamma=0.093,
                                min_child_weight=4,
                                reg_alpha=0.003,
                                reg_lambda=0.0095,
                            )
                            cal_lgb = lgb.LGBMRegressor(
                                objective="regression",
                                learning_rate=0.015,
                                max_depth=8,
                                subsample=0.8,
                                colsample_bytree=0.6,
                                n_estimators=150,
                                min_child_weight=4,
                                reg_alpha=0.003,
                                reg_lambda=0.01,
                                verbose=-1,
                            )
                            cal_xg.fit(fit_X, fit_y, sample_weight=fit_weights, verbose=False)
                            cal_lgb.fit(fit_X, fit_y, sample_weight=fit_weights)

                            # Compute ensemble predictions on calibration set
                            cal_xg_pred = cal_xg.predict(cal_X)
                            cal_lgb_pred = cal_lgb.predict(cal_X)
                            cal_pred = xg_weight * cal_xg_pred + lgb_weight * cal_lgb_pred

                            # Compute signed residuals: actual - predicted
                            cal_residuals = cal_y.values - cal_pred

                            # Task 6.2: Per-horizon conformal offsets
                            # Partition calibration residuals by horizon bucket
                            CONFORMAL_HORIZON_CONFIGS = [
                                {"name": "near",   "min_hours": -24, "max_hours": 24},
                                {"name": "medium", "min_hours": 24,  "max_hours": 48},
                                {"name": "far",    "min_hours": 48,  "max_hours": 999},
                            ]
                            MIN_CONFORMAL_SAMPLES = 50

                            # Convert cal_dt from days to hours for horizon bucketing
                            cal_dt_hours = cal_dt.values * 24

                            for hcfg in CONFORMAL_HORIZON_CONFIGS:
                                h_mask = (cal_dt_hours >= hcfg["min_hours"]) & (cal_dt_hours < hcfg["max_hours"])
                                h_residuals = cal_residuals[h_mask]

                                if len(h_residuals) >= MIN_CONFORMAL_SAMPLES:
                                    q_low = np.percentile(h_residuals, 10)
                                    q_high = np.percentile(h_residuals, 90)
                                    logger.info(
                                        "Conformal offsets '%s': q10=%.3f, q90=%.3f (%d samples)",
                                        hcfg["name"], q_low, q_high, len(h_residuals),
                                    )
                                else:
                                    # Fall back to overall (all-horizon) percentiles
                                    q_low = np.percentile(cal_residuals, 10)
                                    q_high = np.percentile(cal_residuals, 90)
                                    logger.warning(
                                        "Conformal offsets '%s': only %d samples (< %d), "
                                        "falling back to overall offsets q10=%.3f, q90=%.3f",
                                        hcfg["name"], len(h_residuals), MIN_CONFORMAL_SAMPLES,
                                        q_low, q_high,
                                    )
                                conformal_offsets[hcfg["name"]] = (q_low, q_high)

                            # Task 6.4: Coverage monitoring on calibration set
                            # Compute coverage using point prediction + conformal offsets
                            cal_covered = 0
                            for idx in range(len(cal_y)):
                                dt_h = cal_dt_hours[idx]
                                if dt_h < 24:
                                    h_name = "near"
                                elif dt_h < 48:
                                    h_name = "medium"
                                else:
                                    h_name = "far"
                                q_lo, q_hi = conformal_offsets[h_name]
                                band_low = cal_pred[idx] + q_lo
                                band_high = cal_pred[idx] + q_hi
                                if band_low <= cal_y.values[idx] <= band_high:
                                    cal_covered += 1

                            cal_coverage_rate = 100.0 * cal_covered / len(cal_y) if len(cal_y) > 0 else 0.0
                            logger.info(
                                "Conformal calibration coverage: %.1f%% (%d/%d samples)",
                                cal_coverage_rate, cal_covered, len(cal_y),
                            )
                            if cal_coverage_rate < 78 or cal_coverage_rate > 82:
                                logger.warning(
                                    "Conformal coverage %.1f%% is outside target range 78-82%%",
                                    cal_coverage_rate,
                                )
                        else:
                            logger.warning(
                                "Conformal calibration: insufficient data (n=%d, need fit>=50 and cal>=20), skipping",
                                len(train_X),
                            )
                            cal_coverage_rate = None

                        # Train final models on FULL training data for actual predictions
                        xg_model.fit(train_X, train_y, sample_weight=sample_weights, verbose=True)
                        lgb_model.fit(train_X, train_y, sample_weight=sample_weights)
                        # Task 7.3: Retrain Ridge on full data (XGBoost/LightGBM already retrained above)
                        ridge_model.fit(train_X, train_y)

                        # --- Horizon-specific models (Req 5) ---
                        HORIZON_CONFIGS = [
                            {"name": "near",   "min_days": -1, "max_days": 1,  "min_samples": 100},
                            {"name": "medium", "min_days": 1,  "max_days": 2,  "min_samples": 100},
                            {"name": "far",    "min_days": 2,  "max_days": 15, "min_samples": 100},
                        ]

                        # Task 4.1: Split training data by horizon and log sample counts
                        horizon_models = {}
                        for hcfg in HORIZON_CONFIGS:
                            h_mask = (train_dt >= hcfg["min_days"]) & (train_dt < hcfg["max_days"])
                            h_count = h_mask.sum()
                            logger.info(
                                "Horizon '%s' (%.0f-%.0f days): %d samples",
                                hcfg["name"], hcfg["min_days"], hcfg["max_days"], h_count,
                            )

                            if h_count >= hcfg["min_samples"]:
                                # Task 4.2: Train per-horizon XGBoost + LightGBM ensemble
                                h_train_X = train_X[h_mask]
                                h_train_y = train_y[h_mask]
                                h_weights = sample_weights[h_mask]

                                h_xg = xg.XGBRegressor(
                                    objective="reg:squarederror",
                                    booster="gbtree",
                                    learning_rate=0.0135,
                                    max_depth=8,
                                    subsample=0.775,
                                    colsample_bytree=0.604,
                                    n_estimators=150,
                                    gamma=0.093,
                                    min_child_weight=4,
                                    reg_alpha=0.003,
                                    reg_lambda=0.0095,
                                )
                                h_lgb = lgb.LGBMRegressor(
                                    objective="regression",
                                    learning_rate=0.015,
                                    max_depth=8,
                                    subsample=0.8,
                                    colsample_bytree=0.6,
                                    n_estimators=150,
                                    min_child_weight=4,
                                    reg_alpha=0.003,
                                    reg_lambda=0.01,
                                    verbose=-1,
                                )

                                # Per-horizon cross-validation
                                h_n_cv = min(5, len(h_train_X) // 2)
                                if h_n_cv >= 2:
                                    h_xg_scores = cross_val_score(
                                        h_xg, h_train_X, h_train_y,
                                        cv=h_n_cv, scoring="neg_root_mean_squared_error",
                                    )
                                    h_lgb_scores = cross_val_score(
                                        h_lgb, h_train_X, h_train_y,
                                        cv=h_n_cv, scoring="neg_root_mean_squared_error",
                                    )
                                    h_xg_w = 1.0 / abs(h_xg_scores.mean())
                                    h_lgb_w = 1.0 / abs(h_lgb_scores.mean())
                                    h_total_w = h_xg_w + h_lgb_w
                                    h_xg_w /= h_total_w
                                    h_lgb_w /= h_total_w
                                    h_ensemble_scores = h_xg_w * h_xg_scores + h_lgb_w * h_lgb_scores
                                    logger.info(
                                        "Horizon '%s' CV: XGBoost=%.3f, LightGBM=%.3f, Ensemble=%.3f (weights: XG=%.3f, LGB=%.3f)",
                                        hcfg["name"],
                                        -h_xg_scores.mean(), -h_lgb_scores.mean(),
                                        -h_ensemble_scores.mean(),
                                        h_xg_w, h_lgb_w,
                                    )
                                else:
                                    h_xg_w = 0.5
                                    h_lgb_w = 0.5
                                    logger.info(
                                        "Horizon '%s': too few samples for CV (%d), using equal weights",
                                        hcfg["name"], len(h_train_X),
                                    )

                                h_xg.fit(h_train_X, h_train_y, sample_weight=h_weights, verbose=False)
                                h_lgb.fit(h_train_X, h_train_y, sample_weight=h_weights)

                                horizon_models[hcfg["name"]] = (h_xg, h_lgb, h_xg_w, h_lgb_w)
                            else:
                                # Fall back to combined model
                                logger.info(
                                    "Horizon '%s': %d samples < %d minimum, falling back to combined model",
                                    hcfg["name"], h_count, hcfg["min_samples"],
                                )
                                horizon_models[hcfg["name"]] = (xg_model, lgb_model, xg_weight, lgb_weight)

                        # --- Regime-aware spike model (Req 7 - Round 2) ---
                        spike_model_available = False
                        if margin_available:
                            spike_threshold = float(options.get("spike_threshold", 2000) or 2000)
                            comfortable_margin = float(options.get("comfortable_margin", 4000) or 4000)

                            spike_mask = train_X["derated_margin_mw"] < spike_threshold
                            spike_count = spike_mask.sum()

                            if spike_count >= 50:
                                spike_train_X = train_X[spike_mask]
                                spike_train_y = train_y[spike_mask]
                                spike_weights = sample_weights[spike_mask]

                                # Train separate XGBoost + LightGBM on low-margin data
                                spike_xg = xg.XGBRegressor(
                                    objective="reg:squarederror",
                                    booster="gbtree",
                                    learning_rate=0.0135,
                                    max_depth=8,
                                    subsample=0.775,
                                    colsample_bytree=0.604,
                                    n_estimators=150,
                                    gamma=0.093,
                                    min_child_weight=4,
                                    reg_alpha=0.003,
                                    reg_lambda=0.0095,
                                )
                                spike_lgb = lgb.LGBMRegressor(
                                    objective="regression",
                                    learning_rate=0.015,
                                    max_depth=8,
                                    subsample=0.8,
                                    colsample_bytree=0.6,
                                    n_estimators=150,
                                    min_child_weight=4,
                                    reg_alpha=0.003,
                                    reg_lambda=0.01,
                                    verbose=-1,
                                )

                                # CV for spike ensemble weights
                                spike_n_cv = min(5, len(spike_train_X) // 2)
                                if spike_n_cv >= 2:
                                    spike_xg_scores = cross_val_score(
                                        spike_xg, spike_train_X, spike_train_y,
                                        cv=spike_n_cv, scoring="neg_root_mean_squared_error",
                                    )
                                    spike_lgb_scores = cross_val_score(
                                        spike_lgb, spike_train_X, spike_train_y,
                                        cv=spike_n_cv, scoring="neg_root_mean_squared_error",
                                    )
                                    spike_xg_w = 1.0 / abs(spike_xg_scores.mean())
                                    spike_lgb_w = 1.0 / abs(spike_lgb_scores.mean())
                                    spike_total_w = spike_xg_w + spike_lgb_w
                                    spike_xg_w /= spike_total_w
                                    spike_lgb_w /= spike_total_w
                                else:
                                    spike_xg_w = 0.5
                                    spike_lgb_w = 0.5

                                spike_xg.fit(spike_train_X, spike_train_y, sample_weight=spike_weights, verbose=False)
                                spike_lgb.fit(spike_train_X, spike_train_y, sample_weight=spike_weights)

                                spike_model_available = True
                                logger.info(
                                    "Spike model: trained on %d low-margin samples (threshold=%d MW, weights: XG=%.3f, LGB=%.3f)",
                                    spike_count, spike_threshold, spike_xg_w, spike_lgb_w,
                                )
                            else:
                                logger.warning(
                                    "Spike model: only %d samples below threshold %d MW, skipping",
                                    spike_count, spike_threshold,
                                )
                        else:
                            logger.info("Spike model: skipped (derated margin data unavailable)")

                        # Drop the training data set
                        test_X = df[~df["forecast_id"].isin(ff_train.index)]

                        # Drop any data which is actual ir dt < 0
                        test_X = test_X[test_X["date_time"] > test_X["ag_start"]]

                        # Drop the old data
                        test_X = test_X[test_X["days_ago"] < max_days]

                        test_X = test_X.merge(
                            prices["day_ahead"].rename_axis("date_time").reset_index(),
                            on="date_time",
                            how="inner",
                        )
                        test_y = test_X["day_ahead"]

                        if len(test_X) > MAX_TEST_X:
                            _, test_X, _, _ = train_test_split(test_X, test_y, test_size=MAX_TEST_X)

                        if debug:
                            logger.info(f"len(ff)      : {len(ff)}")
                            logger.info(f"len(ff_train): {len(ff_train)}")
                            logger.info(f"len(train_X) : {len(train_X)}")
                            logger.info(f"len(test_X)  : {len(test_X)}")

                            logger.info(f"Earliest ff   : {ff.index.min()}")
                            logger.info(f"Latest ff     : {ff.index.max()}")
                            logger.info(f"Earliest ff_t : {ff_train.index.min()}")
                            logger.info(f"Latest ff_t   : {ff_train.index.max()}")

                            logger.info("train_cols:")
                            for col in train_X.columns:
                                logger.info(
                                    f"  {col:16s}:  {train_X[col].min():10.2f} {train_X[col].mean():10.2f} {train_X[col].max():10.2f}"
                                )

                            logger.info(f"test_X:\n{test_X}")

                        factor = GLOBAL_SETTINGS["REGIONS"]["X"]["factors"][0]

                        results = test_X[["dt", "day_ahead"]].copy() if len(test_X) > 0 else pd.DataFrame(columns=["dt", "day_ahead"])
                        if len(test_X) > 0:
                            xg_test_pred = xg_model.predict(test_X[features])
                            lgb_test_pred = lgb_model.predict(test_X[features])
                        else:
                            xg_test_pred = np.array([])
                            lgb_test_pred = np.array([])
                            logger.warning(
                                "No test data (all %d forecasts in training set). "
                                "Skipping test RMSE. Resolves as more forecasts accumulate.",
                                len(ff_train),
                            )

                        # Task 7.3: Use meta-learner for test set evaluation when available
                        if len(test_X) > 0 and meta_learner_available:
                            ridge_test_pred = ridge_model.predict(test_X[features].fillna(0.0))
                            test_meta_X = np.column_stack([xg_test_pred, lgb_test_pred, ridge_test_pred])
                            results["pred"] = meta_learner.predict(test_meta_X)
                        elif len(test_X) > 0:
                            results["pred"] = xg_weight * xg_test_pred + lgb_weight * lgb_test_pred
                        else:
                            results["pred"] = []

                        # Log individual and ensemble RMSE on test set (Req 4)
                        if len(test_X) > 0:
                            xg_test_rmse = np.sqrt(MSE(test_X["day_ahead"], xg_test_pred))
                            lgb_test_rmse = np.sqrt(MSE(test_X["day_ahead"], lgb_test_pred))
                            ensemble_test_rmse = np.sqrt(MSE(test_X["day_ahead"], results["pred"]))
                            logger.info(
                                "Test RMSE: XGBoost=%.3f, LightGBM=%.3f, Ensemble=%.3f",
                                xg_test_rmse, lgb_test_rmse, ensemble_test_rmse,
                            )

                        # Add required columns before plotting
                        if len(test_X) > 0:
                            results["forecast_created"] = test_X["created_at"].values
                            results["target_time"] = test_X["date_time"].values
                            results["next_agile"] = (test_X["date_time"].values >= test_X["ag_start"].values) & (
                                test_X["date_time"].values < test_X["ag_end"].values
                            )
                            results["error"] = (results["day_ahead"] - results["pred"]) * factor

                        def save_plot(fig, name):
                            plot_path = os.path.join(PLOT_DIR, f"{name}.png")
                            fig.savefig(plot_path, bbox_inches="tight")
                            plt.close(fig)

                        PLOT_DIR = Path(os.path.join("plots", "trends"))
                        PLOT_DIR.mkdir(parents=True, exist_ok=True)
                        for f in PLOT_DIR.glob("*.png"):
                            f.unlink()

                        fig, ax = plt.subplots(figsize=(16, 6))
                        ff = pd.concat(
                            [
                                ff,
                                pd.DataFrame(
                                    index=[ff.index[-1] + 1],
                                    data={
                                        "created_at": [pd.Timestamp(new_name, tz="GB")],
                                        "mean": [-np.mean(scores)],
                                        "stdev": [np.std(scores)],
                                    },
                                ),
                            ]
                        )

                        ax.plot(
                            ff["created_at"], ff["mean"] * factor, lw=2, color="black", marker="o"
                        )
                        ax.fill_between(
                            ff["created_at"],
                            (ff["mean"] - ff["stdev"]) * factor,
                            (ff["mean"] + ff["stdev"]) * factor,
                            color="yellow",
                            alpha=0.3,
                            label="±1 Stdev",
                        )

                        ax.set_ylabel("Predicted Agile Price RMSE [p/kWh]")
                        ax.set_xlabel("Forecast Date/Time")
                        ax.set_ylim(0)
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
                        fig.autofmt_xdate()  # rotates and aligns labels
                        save_plot(fig, "trend")

                        # Directory to save plots
                        PLOT_DIR = Path(os.path.join("plots", "stats_plots"))
                        PLOT_DIR.mkdir(parents=True, exist_ok=True)

                        # Clean old files (optional)
                        for f in PLOT_DIR.glob("*.png"):
                            f.unlink()

                        if len(results) > 0:
                            # 1. Prediction vs Actual over Time
                            fig, ax = plt.subplots(figsize=(16, 6))

                            subset = results[results["next_agile"]].sort_values("target_time")
                            ax.plot(
                                subset["target_time"],
                                subset["day_ahead"],
                                label="Actual",
                                color="black",
                            )
                            ax.plot(
                                subset["target_time"],
                                subset["pred"],
                                label="Trained Model Prediction",
                                alpha=0.4,
                                markersize=2.5,
                                color="red",
                                lw=0,
                                marker="o",
                            )

                            subset = results[~results["next_agile"]].sort_values("target_time")
                            sc = ax.scatter(
                                x=subset["target_time"],
                                y=subset["pred"],
                                label="Predicted",
                                alpha=0.4,
                                c=subset["dt"],
                                lw=0,
                                marker="o",
                                cmap="viridis",
                            )
                            cbar = fig.colorbar(sc, ax=ax)
                            cbar.set_label("Days Ahead (dt)")

                            # Format datetime axis
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
                            fig.autofmt_xdate()  # rotates and aligns labels

                            ax.set_title("Training Dataset - Actual vs Predicted")
                            ax.set_ylabel("£/MWh")
                            ax.legend()
                            save_plot(fig, "1_actual_vs_predicted_over_time")

                            # 2. Prediction vs Actual Scatter
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sc = ax.scatter(
                                results["day_ahead"],
                                results["pred"],
                                alpha=0.2,
                                c=results["dt"],
                                cmap="plasma",
                            )
                            cbar = fig.colorbar(sc, ax=ax)
                            cbar.set_label("Days Ahead (dt)")
                            ax.plot(
                                [results["day_ahead"].min(), results["day_ahead"].max()],
                                [results["day_ahead"].min(), results["day_ahead"].max()],
                                "--",
                                color="gray",
                            )
                            ax.set_xlabel("Actual Day-Ahead Price [£/MWh]")
                            ax.set_ylabel("Predicted Price [£/MWh]")
                            ax.set_title("Prediction vs Actual")
                            save_plot(fig, "2_scatters")

                            # 3. Residuals
                            fig, ax = plt.subplots(figsize=(8, 6))
                            residuals = (results["day_ahead"] - results["pred"]) * factor
                            sns.histplot(residuals, bins=50, kde=True, ax=ax)
                            ax.set_title("Residuals Distribution")
                            ax.set_xlabel("Error (Actual - Predicted) [p/kWh]")
                            save_plot(fig, "3_residuals")

                            # 4. Forecast Error by Horizon
                            fig, ax = plt.subplots(figsize=(8, 6))
                            kde = sns.kdeplot(
                                data=results,
                                x="dt",
                                y="error",
                                fill=True,
                                cmap="Oranges",
                                levels=10,
                                ax=ax,
                            )

                            # Add a colorbar
                            # cbar = plt.colorbar(kde.collections[0], ax=ax)
                            # cbar.set_label("Density")
                            # sns.scatterplot(
                            #     data=results,
                            #     x="dt",
                            #     y=residuals,
                            #     alpha=0.3,
                            #     ax=ax,
                            #     color="grey",
                            #     linewidth=0,
                            # )
                            ax.set_title("2D KDE: Forecast Error by Horizon")
                            ax.set_xlabel("Days Ahead (dt)")
                            ax.set_ylabel("Error (Actual - Predicted) [p/kWh]")
                            save_plot(fig, "4_kde_error_by_horizon")

                            # 5. Feature Importance (XGBoost built-in)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            xg.plot_importance(
                                xg_model, ax=ax, importance_type="gain", show_values=False
                            )
                            ax.set_title("XGBoost Feature Importance (Gain)")
                            save_plot(fig, "5_feature_importance")

                            # 5b. LightGBM Feature Importance (Req 4)
                            lgb_importance = lgb_model.booster_.feature_importance(importance_type="gain")
                            lgb_feat_names = lgb_model.booster_.feature_name()
                            lgb_imp_series = pd.Series(lgb_importance, index=lgb_feat_names).sort_values()

                            fig, ax = plt.subplots(figsize=(8, 6))
                            lgb_imp_series.plot.barh(ax=ax)
                            ax.set_title("LightGBM Feature Importance (Gain)")
                            ax.set_xlabel("Gain")
                            save_plot(fig, "5b_lgb_feature_importance")

                            # Log feature importance from both models
                            xg_importance = xg_model.get_booster().get_score(importance_type="gain")
                            logger.info("XGBoost feature importance (gain): %s", xg_importance)
                            logger.info(
                                "LightGBM feature importance (gain): %s",
                                dict(zip(lgb_feat_names, lgb_importance.tolist())),
                            )

                            # 5c. Combined Feature Importance (Req 4)
                            xg_imp_series = pd.Series(xg_importance)
                            # Normalise both to sum=1 for fair comparison
                            xg_norm = xg_imp_series / xg_imp_series.sum() if xg_imp_series.sum() > 0 else xg_imp_series
                            lgb_norm = lgb_imp_series / lgb_imp_series.sum() if lgb_imp_series.sum() > 0 else lgb_imp_series
                            combined = pd.DataFrame({
                                "XGBoost": xg_norm,
                                "LightGBM": lgb_norm,
                            }).fillna(0).sort_values("XGBoost", ascending=True)

                            fig, ax = plt.subplots(figsize=(10, 8))
                            combined.plot.barh(ax=ax, width=0.8)
                            ax.set_title("Combined Feature Importance (Normalised Gain)")
                            ax.set_xlabel("Normalised Gain")
                            ax.legend(loc="lower right")
                            fig.tight_layout()
                            save_plot(fig, "5c_combined_feature_importance")

                            # fig, ax = plt.subplots(figsize=(8, 6))
                            # bins = [0, 1, 2, 3, 5, 10, 15]
                            # labels = [f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])]
                            # results["horizon_bucket"] = pd.cut(results["dt"], bins=bins, labels=labels, right=True)
                            # ax = sns.violinplot(data=results, x="horizon_bucket", y="error")
                            # ax.set_xlabel("Days Ahead (dt)")
                            # ax.set_ylabel("Error (Actual - Predicted) [£/MWh]")
                            # ax.set_title("Error Distribution by Time Horion Bin")
                            # ax.legend()
                            # save_plot(fig, "6_binned_error_v_time")

                    fc["weekend"] = (fc.index.day_of_week >= 5).astype(int)
                    fc["days_ago"] = 0
                    fc["time"] = fc.index.tz_convert("GB").hour + fc.index.minute / 60
                    fc["dt"] = (fc.index - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400

                    # Cyclical time encoding for prediction data
                    fc_time_gb = fc.index.tz_convert("GB")
                    fc_hour = fc_time_gb.hour + fc_time_gb.minute / 60
                    fc_month = fc_time_gb.month
                    fc["hour_sin"] = np.sin(2 * np.pi * fc_hour / 24)
                    fc["hour_cos"] = np.cos(2 * np.pi * fc_hour / 24)
                    fc["month_sin"] = np.sin(2 * np.pi * fc_month / 12)
                    fc["month_cos"] = np.cos(2 * np.pi * fc_month / 12)

                    # --- Lag price features for prediction data ---
                    # Use latest known values from the lag_features computed on PriceHistory
                    fc_price_series = prices["day_ahead"].sort_index()
                    fc_price_mean = fc_price_series.mean() if len(fc_price_series) > 0 else 0.0

                    if len(ff) > 0:
                        # lag_features was computed in the training block above
                        for col in ["price_lag_1", "price_lag_2", "price_lag_3", "price_lag_4", "price_lag_6", "price_lag_12", "price_lag_24", "price_lag_48", "price_lag_336", "price_rolling_mean_48"]:
                            latest_val = lag_features[col].dropna().iloc[-1] if len(lag_features[col].dropna()) > 0 else fc_price_mean
                            fc[col] = latest_val
                    else:
                        for col in ["price_lag_1", "price_lag_2", "price_lag_3", "price_lag_4", "price_lag_6", "price_lag_12", "price_lag_24", "price_lag_48", "price_lag_336", "price_rolling_mean_48"]:
                            fc[col] = fc_price_mean

                    # --- Price volatility features for prediction data ---
                    fc_vol_7d = fc_price_series.rolling(7 * 48, min_periods=48).std()
                    fc_vol_30d = fc_price_series.rolling(30 * 48, min_periods=48).std()
                    fc["price_volatility_7d"] = fc_vol_7d.iloc[-1] if len(fc_vol_7d.dropna()) > 0 else 0.0
                    fc["price_volatility_30d"] = fc_vol_30d.iloc[-1] if len(fc_vol_30d.dropna()) > 0 else 0.0

                    # --- Residual demand for prediction data ---
                    fc["residual_demand"] = fc["demand"] - fc["bm_wind"] - fc["solar"]

                    # --- Renewable penetration for prediction data (Req 3 - Round 2) ---
                    fc["renewable_penetration"] = np.where(
                        fc["demand"] > 0,
                        (fc["bm_wind"] + fc["solar"]) / fc["demand"] * 100,
                        0.0,
                    )

                    # --- Commodity price features for prediction data (Req 2) ---
                    fc["gas_price_ptherm"] = commodity_prices["gas_price_ptherm"]
                    fc["carbon_price_eur"] = commodity_prices["carbon_price_eur"]

                    # --- Elexon features for prediction data (Req 2) ---
                    # Forward-fill from the latest known values
                    if len(ff) > 0 and elexon_features_available:
                        # Use the latest known values from the training data
                        latest_sbp = df["system_buy_price"].dropna().iloc[-1] if len(df["system_buy_price"].dropna()) > 0 else 0.0
                        latest_ccgt = df["ccgt_generation_mw"].dropna().iloc[-1] if len(df["ccgt_generation_mw"].dropna()) > 0 else 0.0
                        fc["system_buy_price"] = latest_sbp
                        fc["ccgt_generation_mw"] = latest_ccgt
                        # Compute ccgt_share for prediction data
                        fc["ccgt_share"] = fc["ccgt_generation_mw"] / fc["demand"].replace(0, float("nan")) * 100
                        fc["ccgt_share"] = fc["ccgt_share"].fillna(0.0)
                        logger.info(
                            "Elexon prediction features: system_buy_price=%.2f, ccgt_generation_mw=%.0f (forward-filled)",
                            latest_sbp, latest_ccgt,
                        )

                    # --- Interconnector feature for prediction data (Req 3) ---
                    # Forward-fill from the latest known interconnector flow
                    if len(ff) > 0 and interconnector_available:
                        latest_interconnector = df["net_interconnector_mw"].dropna().iloc[-1] if len(df["net_interconnector_mw"].dropna()) > 0 else 0.0
                        fc["net_interconnector_mw"] = latest_interconnector
                        logger.info(
                            "Interconnector prediction feature: net_interconnector_mw=%.0f (forward-filled)",
                            latest_interconnector,
                        )

                    # --- Derated margin features for prediction data (Req 1 - Round 2) ---
                    # Forward-fill from the latest known margin values
                    if len(ff) > 0 and margin_available:
                        latest_margin = df["derated_margin_mw"].dropna().iloc[-1] if len(df["derated_margin_mw"].dropna()) > 0 else 0.0
                        latest_nearest = df["margin_nearest_mw"].dropna().iloc[-1] if len(df["margin_nearest_mw"].dropna()) > 0 else 0.0
                        fc["derated_margin_mw"] = latest_margin
                        fc["margin_nearest_mw"] = latest_nearest
                        logger.info(
                            "Derated margin prediction features: derated_margin_mw=%.0f, margin_nearest_mw=%.0f (forward-filled)",
                            latest_margin, latest_nearest,
                        )

                    # --- French nuclear feature for prediction data (Req 4 - Round 2) ---
                    if len(ff) > 0 and french_nuclear_available:
                        fc["french_nuclear_gw"] = french_nuclear_gw
                        logger.info(
                            "French nuclear prediction feature: french_nuclear_gw=%.2f GW",
                            french_nuclear_gw,
                        )

                    if len(ff) > 0:
                        fc_pred_input = fc.drop("emb_wind", axis=1).reindex(train_X.columns, axis=1)

                        # --- Task 4.3: Horizon-based prediction selection (Req 5) ---
                        fc_dt = fc["dt"].values
                        fc_day_ahead = np.zeros(len(fc))
                        horizon_slot_counts = {"near": 0, "medium": 0, "far": 0}

                        for i in range(len(fc)):
                            dt_val = fc_dt[i]
                            if dt_val < 1:
                                h_name = "near"
                            elif dt_val < 2:
                                h_name = "medium"
                            else:
                                h_name = "far"

                            h_xg, h_lgb, h_xg_w, h_lgb_w = horizon_models[h_name]
                            row_input = fc_pred_input.iloc[[i]]

                            # Task 7.3: Use meta-learner for combined (fallback) model slots
                            if meta_learner_available and h_xg is xg_model:
                                # This horizon fell back to the combined model — use meta-learner
                                xg_p = h_xg.predict(row_input)[0]
                                lgb_p = h_lgb.predict(row_input)[0]
                                # Ridge requires NaN-free input
                                ridge_input = row_input.fillna(0.0)
                                ridge_p = ridge_model.predict(ridge_input)[0]
                                pred_stack = np.array([[xg_p, lgb_p, ridge_p]])
                                fc_day_ahead[i] = meta_learner.predict(pred_stack)[0]
                            else:
                                # Per-horizon model with its own inverse-RMSE weights
                                fc_day_ahead[i] = h_xg_w * h_xg.predict(row_input)[0] + h_lgb_w * h_lgb.predict(row_input)[0]
                            horizon_slot_counts[h_name] += 1

                        fc["day_ahead"] = fc_day_ahead
                        logger.info(
                            "Horizon prediction slots: near=%d, medium=%d, far=%d",
                            horizon_slot_counts["near"],
                            horizon_slot_counts["medium"],
                            horizon_slot_counts["far"],
                        )

                        # --- Spike weight blending (Req 7 - Round 2) ---
                        if spike_model_available:
                            spike_weight = np.clip(
                                1.0 - fc["derated_margin_mw"].values / comfortable_margin, 0.0, 1.0
                            )

                            spike_xg_pred = spike_xg.predict(fc_pred_input)
                            spike_lgb_pred = spike_lgb.predict(fc_pred_input)
                            spike_pred = spike_xg_w * spike_xg_pred + spike_lgb_w * spike_lgb_pred

                            normal_pred = fc["day_ahead"].values
                            fc["day_ahead"] = (1 - spike_weight) * normal_pred + spike_weight * spike_pred

                            n_blended = (spike_weight > 0).sum()
                            mean_spike_w = spike_weight[spike_weight > 0].mean() if n_blended > 0 else 0.0
                            logger.info(
                                "Spike blending: %d/%d slots blended, mean spike_weight=%.3f",
                                n_blended, len(fc), mean_spike_w,
                            )

                        if (len(test_X) > 10) and (not no_ranges):
                            # Graduated quantile regression: tighter bands near-term,
                            # wider far-term for consistent ~5% exceedance per horizon
                            qr_schedule = [
                                (-2, 2, 0.12, 0.88),  # days 0-2: tight
                                (2, 4, 0.08, 0.92),  # days 2-4: medium
                                (4, 7, 0.05, 0.95),  # days 4-7: wider
                                (7, 15, 0.03, 0.97),  # days 7-14: widest
                            ]

                            # Train quantile models — per-horizon where possible (Req 5)
                            # For each quantile pair, train on the appropriate horizon subset
                            # if it has enough samples, otherwise fall back to combined training data
                            def _train_qr_pair(q_lo, q_hi, tr_X, tr_y, sw, xg_w, lgb_w, pred_input):
                                """Train XGBoost+LightGBM quantile pair and return blended predictions."""
                                qr_params_lo = dict(
                                    objective="reg:quantileerror",
                                    quantile_alpha=q_lo,
                                    booster="gbtree",
                                    learning_rate=0.0135,
                                    max_depth=8,
                                    subsample=0.775,
                                    colsample_bytree=0.604,
                                    n_estimators=150,
                                    gamma=0.093,
                                    min_child_weight=4,
                                    reg_alpha=0.003,
                                    reg_lambda=0.0095,
                                )
                                qr_params_hi = dict(qr_params_lo)
                                qr_params_hi["quantile_alpha"] = q_hi

                                xg_m_lo = xg.XGBRegressor(**qr_params_lo)
                                xg_m_lo.fit(tr_X, tr_y, sample_weight=sw, verbose=False)
                                xg_m_hi = xg.XGBRegressor(**qr_params_hi)
                                xg_m_hi.fit(tr_X, tr_y, sample_weight=sw, verbose=False)

                                lgb_m_lo = lgb.LGBMRegressor(
                                    objective="quantile", alpha=q_lo,
                                    learning_rate=0.015, max_depth=8, subsample=0.8,
                                    colsample_bytree=0.6, n_estimators=150,
                                    min_child_weight=4, reg_alpha=0.003, reg_lambda=0.01, verbose=-1,
                                )
                                lgb_m_lo.fit(tr_X, tr_y, sample_weight=sw)
                                lgb_m_hi = lgb.LGBMRegressor(
                                    objective="quantile", alpha=q_hi,
                                    learning_rate=0.015, max_depth=8, subsample=0.8,
                                    colsample_bytree=0.6, n_estimators=150,
                                    min_child_weight=4, reg_alpha=0.003, reg_lambda=0.01, verbose=-1,
                                )
                                lgb_m_hi.fit(tr_X, tr_y, sample_weight=sw)

                                blended_lo = xg_w * xg_m_lo.predict(pred_input) + lgb_w * lgb_m_lo.predict(pred_input)
                                blended_hi = xg_w * xg_m_hi.predict(pred_input) + lgb_w * lgb_m_hi.predict(pred_input)
                                return blended_lo, blended_hi

                            # Train combined (fallback) quantile models
                            qr_models_combined = {}
                            for _, _, q_lo, q_hi in qr_schedule:
                                if (q_lo, q_hi) not in qr_models_combined:
                                    qr_models_combined[(q_lo, q_hi)] = _train_qr_pair(
                                        q_lo, q_hi, train_X, train_y, sample_weights,
                                        xg_weight, lgb_weight, fc_pred_input,
                                    )

                            # Train per-horizon quantile models where horizon has dedicated model
                            qr_models_horizon = {}  # key: (horizon_name, q_lo, q_hi)
                            for hcfg in HORIZON_CONFIGS:
                                h_name = hcfg["name"]
                                h_xg, h_lgb, h_xg_w, h_lgb_w = horizon_models[h_name]
                                # Only train horizon-specific quantile models if the horizon
                                # has its own dedicated model (not the combined fallback)
                                if h_xg is not xg_model:
                                    h_mask = (train_dt >= hcfg["min_days"]) & (train_dt < hcfg["max_days"])
                                    h_tr_X = train_X[h_mask]
                                    h_tr_y = train_y[h_mask]
                                    h_sw = sample_weights[h_mask]
                                    for _, _, q_lo, q_hi in qr_schedule:
                                        if (h_name, q_lo, q_hi) not in qr_models_horizon:
                                            qr_models_horizon[(h_name, q_lo, q_hi)] = _train_qr_pair(
                                                q_lo, q_hi, h_tr_X, h_tr_y, h_sw,
                                                h_xg_w, h_lgb_w, fc_pred_input,
                                            )

                            # Assign bands per slot using horizon-specific quantile models
                            horizon_days = fc["dt"].values
                            low_pred = np.full(len(fc), np.nan)
                            high_pred = np.full(len(fc), np.nan)

                            for h_lo, h_hi, q_lo, q_hi in qr_schedule:
                                mask = (horizon_days >= h_lo) & (horizon_days < h_hi)
                                if not mask.any():
                                    continue

                                # Determine which horizon bucket each slot belongs to
                                for i in np.where(mask)[0]:
                                    dt_val = horizon_days[i]
                                    if dt_val < 1:
                                        h_name = "near"
                                    elif dt_val < 2:
                                        h_name = "medium"
                                    else:
                                        h_name = "far"

                                    # Use horizon-specific quantile model if available
                                    if (h_name, q_lo, q_hi) in qr_models_horizon:
                                        bl, bh = qr_models_horizon[(h_name, q_lo, q_hi)]
                                    else:
                                        bl, bh = qr_models_combined[(q_lo, q_hi)]
                                    low_pred[i] = bl[i]
                                    high_pred[i] = bh[i]

                            # Fallback for any unassigned slots (use widest quantile)
                            remaining = np.isnan(low_pred)
                            if remaining.any():
                                widest_q = (qr_schedule[-1][2], qr_schedule[-1][3])
                                ml_pred, mh_pred = qr_models_combined[widest_q]
                                low_pred[remaining] = ml_pred[remaining]
                                high_pred[remaining] = mh_pred[remaining]

                            # Ensure low <= point <= high
                            fc["day_ahead_low"] = np.minimum(low_pred, fc["day_ahead"].values)
                            fc["day_ahead_high"] = np.maximum(high_pred, fc["day_ahead"].values)

                            # --- Task 6.3: Apply conformal widening to bands (Req 10) ---
                            if conformal_offsets:
                                fc_dt_hours = fc["dt"].values * 24
                                # Identify auction-filled slots to skip conformal widening
                                auction_slots = set()
                                if len(mid_auction) > 0:
                                    auction_slots = set(mid_auction.index.intersection(fc.index))

                                n_widened = 0
                                for i in range(len(fc)):
                                    # Skip auction-filled slots (Req 9) — keep their ±1 p/kWh bands
                                    if fc.index[i] in auction_slots:
                                        continue

                                    dt_h = fc_dt_hours[i]
                                    if dt_h < 24:
                                        h_name = "near"
                                    elif dt_h < 48:
                                        h_name = "medium"
                                    else:
                                        h_name = "far"

                                    q_lo, q_hi = conformal_offsets[h_name]
                                    point_pred = fc["day_ahead"].values[i]
                                    current_low = fc["day_ahead_low"].values[i]
                                    current_high = fc["day_ahead_high"].values[i]

                                    # Widen only (never narrow): use min for low, max for high
                                    new_low = min(current_low, point_pred + q_lo)
                                    new_high = max(current_high, point_pred + q_hi)

                                    # Ensure low <= point <= high
                                    new_low = min(new_low, point_pred)
                                    new_high = max(new_high, point_pred)

                                    if new_low != current_low or new_high != current_high:
                                        n_widened += 1

                                    fc.iloc[i, fc.columns.get_loc("day_ahead_low")] = new_low
                                    fc.iloc[i, fc.columns.get_loc("day_ahead_high")] = new_high

                                logger.info(
                                    "Conformal widening: %d/%d slots widened (%d auction slots skipped)",
                                    n_widened, len(fc) - len(auction_slots), len(auction_slots),
                                )

                        else:
                            fc["day_ahead_low"] = fc["day_ahead"] * 0.9
                            fc["day_ahead_high"] = fc["day_ahead"] * 1.1

                    else:
                        fc["day_ahead"] = None
                        fc["day_ahead_low"] = None
                        fc["day_ahead_high"] = None

                    if debug:
                        logger.info(f"Forecast from {fc.index[0]} tp {fc.index[-1]}")
                        logger.info(f"Agile to      {agile_end}")
                        if len(gb60) > 0:
                            logger.info(f"GB60 to       {prices.index[-1]}")

                        logger.info(f"Forecast\n{fc}")

                    sfs = [
                        pd.DataFrame(
                            index=pd.date_range(fc.index[0], agile_end, freq="30min"),
                            data={"mult": 0, "shift": 1},
                        )
                    ]

                    # Track which slots are covered by actual/known prices
                    covered_idx = sfs[0].index

                    if len(gb60) > 0:
                        sfs.append(
                            pd.DataFrame(
                                index=pd.date_range(gb60.index[0], prices.index[-1], freq="30min"),
                                data={"mult": 0, "shift": 5},
                            )
                        )
                        covered_idx = covered_idx.union(sfs[-1].index)

                    # --- MID auction scale factors (Req 9) ---
                    # Slots with auction results that aren't already covered by Agile or GB60
                    # get mult=0 (use actual auction price) and shift=1 (±1 p/kWh bands)
                    n_auction_slots = 0
                    if len(mid_auction) > 0:
                        mid_auction_idx = mid_auction.index.intersection(fc.index).difference(covered_idx)
                        if len(mid_auction_idx) > 0:
                            sfs.append(
                                pd.DataFrame(
                                    index=mid_auction_idx,
                                    data={"mult": 0, "shift": 1},
                                )
                            )
                            covered_idx = covered_idx.union(mid_auction_idx)
                            n_auction_slots = len(mid_auction_idx)

                    # Remaining slots use model predictions
                    model_pred_idx = fc.index.difference(covered_idx)
                    sfs.append(
                        pd.DataFrame(
                            index=model_pred_idx,
                            data={"mult": 1, "shift": 0},
                        )
                    )

                    # Log slot counts (Req 9 acceptance criteria 5)
                    logger.info(
                        "Blending: %d actual Agile, %d GB60, %d MID auction, %d model prediction slots",
                        len(sfs[0]),
                        len(sfs[1]) if len(gb60) > 0 else 0,
                        n_auction_slots,
                        len(model_pred_idx),
                    )

                    fc = fc.astype(float)
                    scale_factors = pd.concat(sfs)

                    if debug:
                        for i, sf in enumerate(sfs):
                            if len(sf.index) > 0:
                                logger.info(f"idx{i}: {sf.index[0]}:{sf.index[-1]}\n{sf}")
                        logger.info(f"Scale factors\n{scale_factors}")

                    scale_factors = pd.concat(
                        [scale_factors, prices.reindex(scale_factors.index).fillna(0)], axis=1
                    )

                    if debug:
                        logger.info(f"Scale Factors:\n{scale_factors}")

                    fc["day_ahead"] = fc["day_ahead"] * scale_factors["mult"] + scale_factors[
                        "day_ahead"
                    ] * (1 - scale_factors["mult"])
                    fc["day_ahead_low"] = (
                        fc["day_ahead_low"] * scale_factors["mult"]
                        + scale_factors["day_ahead"] * (1 - scale_factors["mult"])
                        - scale_factors["shift"]
                    )
                    fc["day_ahead_high"] = (
                        fc["day_ahead_high"] * scale_factors["mult"]
                        + scale_factors["day_ahead"] * (1 - scale_factors["mult"])
                        + scale_factors["shift"]
                    )

                    if debug:
                        logger.info(
                            pd.concat(
                                [
                                    scale_factors,
                                    fc[["day_ahead", "day_ahead_low", "day_ahead_high"]],
                                ],
                                axis=1,
                            )
                        )

                    ag = pd.concat(
                        [
                            pd.DataFrame(
                                index=fc.index,
                                data={
                                    "region": region,
                                    "agile_pred": day_ahead_to_agile(fc["day_ahead"], region=region)
                                    .astype(float)
                                    .round(2),
                                    "agile_low": day_ahead_to_agile(
                                        fc["day_ahead_low"], region=region
                                    )
                                    .astype(float)
                                    .round(2),
                                    "agile_high": day_ahead_to_agile(
                                        fc["day_ahead_high"], region=region
                                    )
                                    .astype(float)
                                    .round(2),
                                },
                            )
                            for region in regions
                        ]
                    )

                    # fc = fc[list(fd.columns)[3:]]
                    fc = fc[
                        [
                            "bm_wind",
                            "solar",
                            "emb_wind",
                            "temp_2m",
                            "wind_10m",
                            "rad",
                            "demand",
                            "day_ahead",
                        ]
                    ]

                    if debug:
                        logger.info(f"Final forecast from {fc.index[0]} to {fc.index[-1]}")
                        logger.info(f"Forecast\n{fc}")

                    mean_score = -np.mean(scores) if len(scores) > 0 else 0.0
                    stdev_score = np.std(scores) if len(scores) > 0 else 0.0

                    # Log conformal coverage rate alongside forecast metadata (Req 10)
                    if cal_coverage_rate is not None:
                        logger.info(
                            "Forecast metadata: RMSE=%.3f, stdev=%.3f, conformal_coverage=%.1f%%",
                            mean_score, stdev_score, cal_coverage_rate,
                        )

                    # Drop rows with NaN predictions (slots beyond forecast data range
                    # where scale factor alignment produces NaN)
                    nan_count = (
                        ag[["agile_pred", "agile_low", "agile_high"]].isna().any(axis=1).sum()
                    )
                    if nan_count > 0:
                        logger.warning(
                            "Dropping %d/%d AgileData rows with NaN predictions", nan_count, len(ag)
                        )
                    ag = ag.dropna(subset=["agile_pred", "agile_low", "agile_high"])

                    this_forecast = Forecasts(name=new_name, mean=mean_score, stdev=stdev_score)
                    this_forecast.save()
                    fc["forecast"] = this_forecast
                    ag["forecast"] = this_forecast
                    df_to_Model(fc, ForecastData)
                    df_to_Model(ag, AgileData)

        # --- Data source health summary ---
        logger.info(
            "Data sources: Agile=%d pages, GB60=%s, MID=%d, Elexon=%s, Interconnector=%s, Margin=%s, Nuclear=%s, Commodity=ok",
            len(all_agile_pages),
            "ok" if isinstance(gb60, pd.DataFrame) and len(gb60) > 0 else "unavailable",
            len(mid_auction) if hasattr(mid_auction, '__len__') else 0,
            "ok" if elexon_features_available else "unavailable",
            "ok" if interconnector_available else "unavailable",
            "ok" if margin_available else "unavailable",
            "ok" if french_nuclear_available else "unavailable",
        )

        # --- Fetch official agilepredict.com predictions for comparison ---
        try:
            _this_fc = this_forecast  # noqa: F841 — will NameError if prediction was skipped
        except NameError:
            _this_fc = None

        if _this_fc is not None:
            try:
                import requests as _requests

                OFFICIAL_API = "https://agilepredict.com/api/F/?high_low=true"
                OFFICIAL_TIMEOUT = 30

                resp = _requests.get(OFFICIAL_API, timeout=OFFICIAL_TIMEOUT)
                resp.raise_for_status()
                official_data = resp.json()

                official_rows = []
                for forecast_block in official_data:
                    for price in forecast_block.get("prices", []):
                        official_rows.append({
                            "forecast": _this_fc,
                            "date_time": pd.Timestamp(price["date_time"]),
                            "agile_pred": price["agile_pred"],
                            "agile_low": price["agile_low"],
                            "agile_high": price["agile_high"],
                        })

                created = 0
                for row in official_rows:
                    _, was_created = OfficialAgileData.objects.get_or_create(
                        forecast=row["forecast"],
                        date_time=row["date_time"],
                        defaults=row,
                    )
                    if was_created:
                        created += 1
                logger.info("Official predictions: stored %d/%d rows", created, len(official_rows))

            except Exception as e:
                logger.warning("Failed to fetch official agilepredict.com predictions: %s", e)
        else:
            logger.info("Official predictions: skipped (no forecast this run)")

        if debug:
            for f in Forecasts.objects.all().order_by("-created_at"):
                logger.info(f"{f.id:4d}: {f.name}")
        else:
            try:
                logger.info(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                logger.info("No forecast added")

        # --- Smart cleanup: runs AFTER prediction+save ---
        # Strategy: preserve all training data, minimise storage growth
        #
        # Phase 1 — Dedup: keep up to 3 forecasts per calendar day:
        #           - closest to 08:00 (pre-auction baseline)
        #           - closest to 16:15 (post-auction, best quality)
        #           - closest to 00:00 (overnight)
        #           + always the newest.  Delete duplicates via CASCADE
        #           (removes their ForecastData + AgileData).
        #           OfficialAgileData is NOT deleted — kept permanently
        #           for head-to-head comparison analysis.
        #
        # Phase 2 — AgileData trim: for kept forecasts older than
        #           AGILE_DATA_RETENTION_DAYS, delete AgileData for all
        #           regions EXCEPT our local region (config: region).
        #           Local region AgileData is kept forever to enable
        #           ongoing prediction accuracy analysis (prediction vs
        #           actual from PriceHistory).  ~18 MB/year, ~175 MB/10yr.
        #           StatsView error heatmap also benefits from this.
        #
        # OfficialAgileData is NEVER deleted (except via CASCADE when
        # duplicate forecasts are removed). Kept permanently for
        # head-to-head comparison. Growth: ~1.8 KB/day, ~0.6 MB/year.
        #
        # Never deleted: PriceHistory, ForecastData, AgileData(local region),
        #                OfficialAgileData.
        # Growth: ~28 MB/year (FD ~15 MB + PH ~0.5 MB + AD(F) ~12 MB + OAD ~0.6 MB).
        AGILE_DATA_RETENTION_DAYS = 14
        # Region whose AgileData is kept permanently for accuracy analysis.
        # Matches the DNO region code used by run_agile_predict_update.py.
        # Default "F" (East Midlands); override via AGILE_LOCAL_REGION env var.
        import os as _os
        local_region = _os.environ.get("AGILE_LOCAL_REGION", "F")

        all_forecasts = Forecasts.objects.all().order_by("-created_at")
        if all_forecasts.exists():
            keep_ids = set()

            # Always keep the forecast just created by this run
            try:
                keep_ids.add(this_forecast.id)
            except NameError:
                # this_forecast may not exist if prediction phase was skipped
                keep_ids.add(all_forecasts.first().id)

            # Pre-compute which forecasts have AgileData (real predictions)
            ids_with_agile = set(AgileData.objects.values_list("forecast_id", flat=True).distinct())

            # Keep best forecasts per calendar day (up to 3: closest to 00:00, 08:00, 16:15)
            # Prefer real forecasts (with AgileData) over backfilled ones
            # This enables intra-day convergence analysis and time-of-day accuracy comparison
            KEEP_TARGETS = [
                ("night", pd.Timedelta(hours=0, minutes=0)),
                ("morning", pd.Timedelta(hours=8, minutes=0)),
                ("afternoon", pd.Timedelta(hours=16, minutes=15)),
            ]
            daily_best = {}  # key: (date_key, target_name) → (f.id, has_agile, distance)
            for f in all_forecasts:
                try:
                    dt = pd.to_datetime(f.name).tz_localize("GB")
                except (ValueError, TypeError):
                    continue
                date_key = dt.normalize()
                has_agile = f.id in ids_with_agile
                for target_name, target_offset in KEEP_TARGETS:
                    target_time = date_key + target_offset
                    distance = abs((dt - target_time).total_seconds())
                    slot_key = (date_key, target_name)
                    if slot_key not in daily_best:
                        daily_best[slot_key] = (f.id, has_agile, distance)
                    else:
                        _, prev_has_agile, prev_distance = daily_best[slot_key]
                        if has_agile and not prev_has_agile:
                            daily_best[slot_key] = (f.id, has_agile, distance)
                        elif has_agile == prev_has_agile and distance < prev_distance:
                            daily_best[slot_key] = (f.id, has_agile, distance)

            keep_ids.update(fid for fid, _, _ in daily_best.values())

            # Phase 1: delete duplicate forecasts (CASCADE removes FD + AD + OAD)
            to_delete = Forecasts.objects.exclude(id__in=keep_ids)
            n_delete = to_delete.count()
            if n_delete > 0:
                if debug:
                    for f in to_delete:
                        fd_n = ForecastData.objects.filter(forecast=f).count()
                        ad_n = AgileData.objects.filter(forecast=f).count()
                        logger.info(f"  Deleting duplicate {f.id}: {f.name} (FD={fd_n}, AD={ad_n})")
                to_delete.delete()
                logger.info(
                    "Cleanup phase 1: deleted %d duplicate forecasts, kept %d",
                    n_delete,
                    len(keep_ids),
                )
            elif debug:
                logger.info("Cleanup phase 1: no duplicates (%d forecasts)", all_forecasts.count())

            # Phase 2: trim AgileData from old kept forecasts
            cutoff = pd.Timestamp.now(tz="GB") - pd.Timedelta(days=AGILE_DATA_RETENTION_DAYS)
            old_forecast_ids = []
            for f in Forecasts.objects.filter(id__in=keep_ids).exclude(id=all_forecasts.first().id):
                try:
                    dt = pd.to_datetime(f.name).tz_localize("GB")
                except (ValueError, TypeError):
                    continue
                if dt < cutoff:
                    old_forecast_ids.append(f.id)

            if old_forecast_ids:
                # Delete non-local regions only; keep local region forever for accuracy analysis
                ad_to_delete = AgileData.objects.filter(
                    forecast_id__in=old_forecast_ids
                ).exclude(region=local_region)
                ad_count = ad_to_delete.count()
                if ad_count > 0:
                    ad_to_delete.delete()
                    logger.info(
                        "Cleanup phase 2: deleted %d AgileData rows (non-%s regions) from %d"
                        " forecasts older than %d days (region %s kept permanently)",
                        ad_count,
                        local_region,
                        len(old_forecast_ids),
                        AGILE_DATA_RETENTION_DAYS,
                        local_region,
                    )
                elif debug:
                    logger.info("Cleanup phase 2: no old AgileData to trim")
            elif debug:
                logger.info(
                    "Cleanup phase 2: no forecasts older than %d days", AGILE_DATA_RETENTION_DAYS
                )
