import xgboost as xg
from pathlib import Path
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

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
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

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

        parser.add_argument(
            "--min_fd",
        )

        parser.add_argument(
            "--min_ad",
        )

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
            "--bootstrap",
            action="store_true",
            help="Skip forecast deletion so training data can accumulate across runs (use for initial setup only)",
        )

    def handle(self, *args, **options):
        # Setup logging

        debug = options.get("debug", False)

        min_fd = int(options.get("min_fd", 600) or 600)
        min_ad = int(options.get("min_ad", 1500) or 1500)
        max_days = int(options.get("max_days", 60) or 60)

        no_ranges = options.get("no_ranges", False)

        drop_cols = ["emb_wind"]
        if options.get("no_day_of_week", False):
            drop_cols += ["day_of_week"]

        drop_last = int(options.get("drop_last", 0) or 0)

        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        # Clean any invalid forecasts
        if debug:
            logger.info(f"Max days: {max_days}")

            logger.info(f"  ID  |       Name       |  #FD  |   #AD   | Days |")
            logger.info(f"------+------------------+-------+---------+------+")
        keep = []
        for f in Forecasts.objects.all().order_by("-created_at"):
            fd = ForecastData.objects.filter(forecast=f)
            ad = AgileData.objects.filter(forecast=f)
            dt = pd.to_datetime(f.name).tz_localize("GB")
            days = (pd.Timestamp.now(tz="GB") - dt).days
            if fd.count() < min_fd or ad.count() < min_ad:
                fail = " <- Fail"
            else:
                fail = " <- Manual"
                if days < max_days * 2:
                    for hour in [6, 10, 11, 16, 22]:
                        if f"{hour:02d}:15" in f.name:
                            keep.append(f.id)
                            fail = ""
                else:
                    fail = "<- Old"
            if debug:
                logger.info(f"{f.id:5d} | {f.name} | {fd.count():5d} | {ad.count():7d} | {days:4d} | {fail}")

        bootstrap = options.get("bootstrap", False)
        if bootstrap:
            logger.info("Bootstrap mode: skipping forecast deletion so training data can accumulate")
        else:
            forecasts_to_delete = Forecasts.objects.exclude(id__in=keep)
            if debug:
                logger.info(f"\nDeleting ({forecasts_to_delete})\n")
            forecasts_to_delete.delete()

        prices, start = model_to_df(PriceHistory)

        if debug:
            logger.info("Getting Historic Prices")
            logger.info(f"Prices\n{prices}")

        # Pages arrive newest-first (descending); filter by original start so we
        # don't re-write records already in the DB, but don't gate on prices.index[-1]
        # (that would discard older pages as soon as one page is written).
        all_agile_pages = []
        for page_agile in get_agile_pages(start=start, region="F"):
            all_agile_pages.append(page_agile)
            page_day_ahead = day_ahead_to_agile(page_agile, reverse=True, region="F")
            page_new = pd.concat([page_day_ahead, page_agile], axis=1)
            page_new = page_new[page_new.index >= start]
            if len(page_new) > 0:
                if debug:
                    logger.info(f"New Prices (page)\n{page_new}")
                logger.info("Writing %d new price records to DB (up to %s)", len(page_new), page_new.index[-1])
                df_to_Model(page_new, PriceHistory, update=True)
                prices = pd.concat([prices, page_new]).sort_index()

        agile = pd.concat(all_agile_pages).sort_index() if all_agile_pages else pd.Series(name="agile", dtype=float)
        agile = agile[~agile.index.duplicated(keep="last")]
        day_ahead = day_ahead_to_agile(agile, reverse=True, region="F")
        prices = prices[~prices.index.duplicated(keep="last")]

        agile_end = prices.index[-1]
        gb60 = get_gb60()

        if debug:
            logger.info(f"GB60:\n{gb60}")

        gb60 = gb60.resample("30min").ffill().loc[agile_end + pd.Timedelta("30min") :]

        if len(gb60) > 0:
            gb60 = gb60.reindex(
                pd.date_range(gb60.index[0], gb60.index[-1] + pd.Timedelta("30min"), freq="30min")
            ).ffill()
            gb60 = pd.concat([gb60, day_ahead_to_agile(gb60)], axis=1).set_axis(["day_ahead", "agile"], axis=1)
            prices = pd.concat([prices, gb60]).sort_index()

        if debug:
            logger.info(f"Merged prices:\n{prices}")

        if drop_last > 0:
            logger.info(f"drop_last: {drop_last}")
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")
            prices = prices.iloc[:-drop_last]
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.exclude(id__in=ignore_forecast).order_by("-created_at")
            last_forecasts = {
                forecast.created_at.date(): forecast.id for forecast in base_forecasts.order_by("created_at")
            }

            base_forecasts = base_forecasts.filter(id__in=[last_forecasts[k] for k in last_forecasts])

            if debug:
                logger.info("Getting latest Forecast")

            fc, missing_fc = get_latest_forecast()

            if len(missing_fc) > 0:
                logger.error(f">>> ERROR: Unable to run forecast due to missing columns: {', '.join(missing_fc)}")
            else:
                if debug:
                    logger.info(fc)

                if len(fc) > 0:
                    fd = pd.DataFrame(list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values()))
                    ff = pd.DataFrame(list(Forecasts.objects.exclude(id__in=ignore_forecast).values()))
                    scores = []  # initialise so line 707 ref is safe if training block is skipped (e.g. first run)

                    if len(ff) > 0:
                        logger.info(ff)
                        ff = ff.set_index("id").sort_index()
                        ff["created_at"] = pd.to_datetime(ff["name"]).dt.tz_localize("GB")
                        ff["date"] = ff["created_at"].dt.tz_convert("GB").dt.normalize()
                        ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
                        ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)

                        # Only train on the forecasts closest to 16:15
                        ff["dt1600"] = (
                            (ff["date"] + pd.Timedelta(hours=16, minutes=15) - ff["created_at"].dt.tz_convert("GB"))
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

                        # df is the full dataset
                        df = (
                            (fd.merge(ff, right_index=True, left_on="forecast_id"))
                            .set_index("date_time")
                            .drop("day_ahead", axis=1)
                        )

                        df["dow"] = df.index.day_of_week
                        df["weekend"] = (df.index.day_of_week >= 5).astype(int)
                        df["time"] = df.index.tz_convert("GB").hour + df.index.minute / 60
                        df["days_ago"] = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds() / 3600 / 24
                        df["dt"] = (df.index - df["created_at"]).dt.total_seconds() / 3600 / 24
                        df["peak"] = ((df["time"] >= 16) & (df["time"] < 19)).astype(float)

                        # Cyclical time encoding for seasonal/diurnal patterns
                        time_gb = df.index.tz_convert("GB")
                        hour = time_gb.hour + time_gb.minute / 60
                        month = time_gb.month
                        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
                        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
                        df["month_sin"] = np.sin(2 * np.pi * month / 12)
                        df["month_cos"] = np.cos(2 * np.pi * month / 12)

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
                        ]

                        # Only use the forecasts closest to 16:15 for training
                        train_X = df[df["forecast_id"].isin(ff_train.index)]
                        train_X = train_X[train_X["days_ago"] < max_days]

                        # Only train on the next agile prices that are set from the pm auction
                        train_X = train_X[
                            (train_X.index >= train_X["ag_start"]) & (train_X.index < train_X["ag_end"])
                        ][features]

                        # Get the prices to match the forecast
                        train_X = train_X.merge(prices["day_ahead"], left_index=True, right_index=True)

                        if debug:
                            logger.info(f"train_X:\n{train_X}")

                        train_y = train_X.pop("day_ahead")
                        sample_weights = ((np.log10((train_y - train_y.mean()).abs() + 10) * 5) - 4).round(0)

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

                        n_cv = min(5, len(train_X) // 2)
                        if n_cv >= 2:
                            scores = cross_val_score(
                                xg_model, train_X, train_y, cv=n_cv, scoring="neg_root_mean_squared_error"
                            )
                            logger.info(f"Cross-val scrore: {scores}")
                        else:
                            scores = np.array([0.0])
                            logger.info("Too few training samples for cross-validation (n=%d) — skipping cv", len(train_X))

                        xg_model.fit(train_X, train_y, sample_weight=sample_weights, verbose=True)

                        # Drop the training data set
                        test_X = df[~df["forecast_id"].isin(ff_train.index)]

                        # Drop any data which is actual ir dt < 0
                        test_X = test_X[test_X.index > test_X["ag_start"]]

                        # Drop the old data
                        test_X = test_X[test_X["days_ago"] < max_days]

                        test_X = test_X.merge(prices["day_ahead"], left_index=True, right_index=True)
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

                        results = test_X[["dt", "day_ahead"]].copy()
                        results["pred"] = xg_model.predict(test_X[features])

                        # Add required columns before plotting
                        results["forecast_created"] = test_X["created_at"]
                        results["target_time"] = test_X.index
                        results["next_agile"] = (test_X.index >= test_X["ag_start"]) & (
                            test_X.index < test_X["ag_end"]
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

                        ax.plot(ff["created_at"], ff["mean"] * factor, lw=2, color="black", marker="o")
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

                        # 1. Prediction vs Actual over Time
                        fig, ax = plt.subplots(figsize=(16, 6))

                        subset = results[results["next_agile"]].sort_values("target_time")
                        ax.plot(subset["target_time"], subset["day_ahead"], label="Actual", color="black")
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
                            results["day_ahead"], results["pred"], alpha=0.2, c=results["dt"], cmap="plasma"
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
                        xg.plot_importance(xg_model, ax=ax, importance_type="gain", show_values=False)
                        ax.set_title("XGBoost Feature Importance (Gain)")
                        save_plot(fig, "5_feature_importance")

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

                    if len(ff) > 0:
                        fc_pred_input = fc.drop("emb_wind", axis=1).reindex(train_X.columns, axis=1)
                        fc["day_ahead"] = xg_model.predict(fc_pred_input)

                        if (len(test_X) > 10) and (not no_ranges):
                            # Graduated quantile regression: tighter bands near-term,
                            # wider far-term for consistent ~5% exceedance per horizon
                            qr_schedule = [
                                (-2, 2, 0.12, 0.88),   # days 0-2: tight
                                (2, 4, 0.08, 0.92),    # days 2-4: medium
                                (4, 7, 0.05, 0.95),    # days 4-7: wider
                                (7, 15, 0.03, 0.97),   # days 7-14: widest
                            ]

                            # Train quantile models (reuse same training data)
                            qr_models = {}
                            for _, _, q_lo, q_hi in qr_schedule:
                                if (q_lo, q_hi) not in qr_models:
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

                                    m_lo = xg.XGBRegressor(**qr_params_lo)
                                    m_lo.fit(train_X, train_y, sample_weight=sample_weights, verbose=False)
                                    m_hi = xg.XGBRegressor(**qr_params_hi)
                                    m_hi.fit(train_X, train_y, sample_weight=sample_weights, verbose=False)
                                    qr_models[(q_lo, q_hi)] = (
                                        m_lo.predict(fc_pred_input),
                                        m_hi.predict(fc_pred_input),
                                    )

                            # Assign bands per horizon bucket
                            horizon_days = fc["dt"].values
                            low_pred = np.full(len(fc), np.nan)
                            high_pred = np.full(len(fc), np.nan)

                            for h_lo, h_hi, q_lo, q_hi in qr_schedule:
                                mask = (horizon_days >= h_lo) & (horizon_days < h_hi)
                                ml_pred, mh_pred = qr_models[(q_lo, q_hi)]
                                low_pred[mask] = ml_pred[mask]
                                high_pred[mask] = mh_pred[mask]

                            # Fallback for any unassigned slots (use widest quantile)
                            remaining = np.isnan(low_pred)
                            if remaining.any():
                                widest_q = (qr_schedule[-1][2], qr_schedule[-1][3])
                                ml_pred, mh_pred = qr_models[widest_q]
                                low_pred[remaining] = ml_pred[remaining]
                                high_pred[remaining] = mh_pred[remaining]

                            # Ensure low <= point <= high
                            fc["day_ahead_low"] = np.minimum(low_pred, fc["day_ahead"].values)
                            fc["day_ahead_high"] = np.maximum(high_pred, fc["day_ahead"].values)

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
                            index=pd.date_range(fc.index[0], agile_end, freq="30min"), data={"mult": 0, "shift": 1}
                        )
                    ]

                    if len(gb60) > 0:
                        sfs.append(
                            pd.DataFrame(
                                index=pd.date_range(gb60.index[0], prices.index[-1], freq="30min"),
                                data={"mult": 0, "shift": 5},
                            )
                        )
                        sfs.append(
                            pd.DataFrame(
                                index=fc.index.difference(sfs[0].index.union(sfs[1].index)),
                                data={"mult": 1, "shift": 0},
                            )
                        )
                    else:
                        sfs.append(pd.DataFrame(index=fc.index.difference(sfs[0].index), data={"mult": 1, "shift": 0}))

                    fc = fc.astype(float)
                    scale_factors = pd.concat(sfs)

                    if debug:
                        for i, sf in enumerate(sfs):
                            if len(sf.index) > 0:
                                logger.info(f"idx{i}: {sf.index[0]}:{sf.index[-1]}\n{sf}")
                        logger.info(f"Scale factors\n{scale_factors}")

                    scale_factors = pd.concat([scale_factors, prices.reindex(scale_factors.index).fillna(0)], axis=1)

                    if debug:
                        logger.info(f"Scale Factors:\n{scale_factors}")

                    fc["day_ahead"] = fc["day_ahead"] * scale_factors["mult"] + scale_factors["day_ahead"] * (
                        1 - scale_factors["mult"]
                    )
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
                            pd.concat([scale_factors, fc[["day_ahead", "day_ahead_low", "day_ahead_high"]]], axis=1)
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
                                    "agile_low": day_ahead_to_agile(fc["day_ahead_low"], region=region)
                                    .astype(float)
                                    .round(2),
                                    "agile_high": day_ahead_to_agile(fc["day_ahead_high"], region=region)
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
                    this_forecast = Forecasts(name=new_name, mean=mean_score, stdev=stdev_score)
                    this_forecast.save()
                    fc["forecast"] = this_forecast
                    ag["forecast"] = this_forecast
                    df_to_Model(fc, ForecastData)
                    df_to_Model(ag, AgileData)

        if debug:
            for f in Forecasts.objects.all().order_by("-created_at"):
                logger.info(f"{f.id:4d}: {f.name}")
        else:
            try:
                logger.info(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                logger.info("No forecast added")
