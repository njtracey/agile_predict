import json
import os

import pandas as pd
import requests
import time
import logging

from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib import parse
from datetime import datetime
from config.settings import GLOBAL_SETTINGS
from django.core.management import call_command

from prices.models import History, PriceHistory, Forecasts, ForecastData, AgileData

OCTOPUS_PRODUCT_URL = r"https://api.octopus.energy/v1/products/"


logger = logging.getLogger(__name__)

TIME_FORMAT = "%d/%m %H:%M %Z"
MAX_ITERS = 3
RETRIES = 3
RETRY_CODES = [
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
]

regions = GLOBAL_SETTINGS["REGIONS"]

# OilPriceAPI key for gas and carbon price fetching (repo is not public)
OILPRICE_API_KEY = "09287899c129971df72aa7166ab8243649c5dbbb04eadd292fa5c3a1d5c29b5c"

# Default commodity prices used when API and cache are both unavailable
DEFAULT_GAS_PRICE_PTHERM = 80.0
DEFAULT_CARBON_PRICE_EUR = 70.0


def fetch_commodity_prices(cache_path="cache/commodity_prices.json"):
    """Fetch daily gas and carbon prices from OilPriceAPI, with caching.

    Returns dict with 'gas_price_ptherm' and 'carbon_price_eur'.
    Uses cache if less than 24 hours old to conserve API quota (50 req/month).
    Fetches both in a single request: by_code=NATURAL_GAS_GBP,EU_CARBON_EUR
    """
    defaults = {
        "gas_price_ptherm": DEFAULT_GAS_PRICE_PTHERM,
        "carbon_price_eur": DEFAULT_CARBON_PRICE_EUR,
    }

    # Ensure cache directory exists
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Check cache freshness
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            last_updated = pd.Timestamp(cached["last_updated"])
            age_hours = (pd.Timestamp.now(tz="UTC") - last_updated).total_seconds() / 3600
            if age_hours < 24:
                logger.info("Commodity prices: using cache (%.1f hours old)", age_hours)
                return {
                    "gas_price_ptherm": cached["gas_price_ptherm"],
                    "carbon_price_eur": cached["carbon_price_eur"],
                }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Commodity prices: cache read error: %s", e)

    # Fetch from API
    url = "https://api.oilpriceapi.com/v1/prices/latest"
    params = {"by_code": "NATURAL_GAS_GBP,EU_CARBON_EUR"}
    headers = {"Authorization": f"Token {OILPRICE_API_KEY}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse response — API returns a dict with 'data' containing 'prices' list
        prices_data = data.get("data", {}).get("prices", [])
        result = {}
        for item in prices_data:
            code = item.get("code", "")
            price = item.get("price")
            if code == "NATURAL_GAS_GBP" and price is not None:
                result["gas_price_ptherm"] = float(price)
            elif code == "EU_CARBON_EUR" and price is not None:
                result["carbon_price_eur"] = float(price)

        if "gas_price_ptherm" not in result or "carbon_price_eur" not in result:
            logger.warning("Commodity prices: API response missing expected price codes, data: %s", data)
            # Fill in any missing values from cache or defaults
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    result.setdefault("gas_price_ptherm", cached.get("gas_price_ptherm", defaults["gas_price_ptherm"]))
                    result.setdefault("carbon_price_eur", cached.get("carbon_price_eur", defaults["carbon_price_eur"]))
                except (json.JSONDecodeError, KeyError):
                    result.setdefault("gas_price_ptherm", defaults["gas_price_ptherm"])
                    result.setdefault("carbon_price_eur", defaults["carbon_price_eur"])
            else:
                result.setdefault("gas_price_ptherm", defaults["gas_price_ptherm"])
                result.setdefault("carbon_price_eur", defaults["carbon_price_eur"])

        # Write to cache
        cache_data = {
            "last_updated": pd.Timestamp.now(tz="UTC").isoformat(),
            "gas_price_ptherm": result["gas_price_ptherm"],
            "carbon_price_eur": result["carbon_price_eur"],
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info(
            "Commodity prices: fetched gas=%.2f p/therm, carbon=%.2f EUR/tonne",
            result["gas_price_ptherm"],
            result["carbon_price_eur"],
        )
        return result

    except Exception as e:
        logger.warning("Commodity prices: API fetch failed: %s", e)

        # Fall back to cache (even if stale)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                logger.warning("Commodity prices: using stale cache as fallback")
                return {
                    "gas_price_ptherm": cached["gas_price_ptherm"],
                    "carbon_price_eur": cached["carbon_price_eur"],
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # No cache available — use defaults
        logger.warning(
            "Commodity prices: no cache available, using defaults gas=%.0f, carbon=%.0f",
            defaults["gas_price_ptherm"],
            defaults["carbon_price_eur"],
        )
        return defaults


def fetch_system_prices(date_from, date_to):
    """Fetch half-hourly system buy prices from Elexon BMRS.

    Iterates over each date in the range and fetches settlement system prices.
    Returns DataFrame with 'system_buy_price' column (GBP/MWh) indexed by UTC datetime.
    Returns empty DataFrame on failure.
    """
    all_rows = []
    try:
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
        logger.info("Elexon system prices: fetching %d days (%s to %s)", len(dates), date_from, date_to)

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            url = f"https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date_str}"
            params = {"format": "json"}
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                for record in data:
                    settlement_date = record.get("settlementDate")
                    settlement_period = record.get("settlementPeriod")
                    system_buy_price = record.get("systemBuyPrice")
                    if settlement_date and settlement_period is not None and system_buy_price is not None:
                        # Settlement period 1 = 00:00-00:30 UTC
                        dt = pd.Timestamp(settlement_date) + (int(settlement_period) - 1) * pd.Timedelta("30min")
                        dt = dt.tz_localize("UTC")
                        all_rows.append({"datetime": dt, "system_buy_price": float(system_buy_price)})
            except Exception as e:
                logger.warning("Elexon system prices: failed for date %s: %s", date_str, e)
                continue

    except Exception as e:
        logger.warning("Elexon system prices: fetch failed: %s", e)
        return pd.DataFrame()

    if not all_rows:
        logger.warning("Elexon system prices: no data returned for %s to %s", date_from, date_to)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("Elexon system prices: fetched %d records (%s to %s)", len(df), df.index[0], df.index[-1])
    return df


def fetch_ccgt_generation(date_from, date_to):
    """Fetch half-hourly CCGT generation from Elexon FUELHH dataset.

    Iterates over the date range in 7-day chunks (API has date range limits).
    Filters for fuelType=CCGT and sums generation per settlement period.
    Returns DataFrame with 'ccgt_generation_mw' column indexed by UTC datetime.
    Returns empty DataFrame on failure.
    """
    ccgt_rows = {}
    try:
        start = pd.Timestamp(date_from).normalize()
        end = pd.Timestamp(date_to).normalize()
        logger.info("Elexon CCGT generation: fetching %s to %s", date_from, date_to)
        # Iterate in 7-day chunks to stay within API limits
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + pd.Timedelta(days=6), end)
            chunk_from_str = chunk_start.strftime("%Y-%m-%d")
            chunk_to_str = chunk_end.strftime("%Y-%m-%d")
            url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"
            params = {
                "settlementDateFrom": chunk_from_str,
                "settlementDateTo": chunk_to_str,
                "format": "json",
            }
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("data", [])

                for record in data:
                    if record.get("fuelType") != "CCGT":
                        continue
                    settlement_date = record.get("settlementDate")
                    settlement_period = record.get("settlementPeriod")
                    generation = record.get("generation")
                    if settlement_date and settlement_period is not None and generation is not None:
                        dt = pd.Timestamp(settlement_date) + (int(settlement_period) - 1) * pd.Timedelta("30min")
                        dt = dt.tz_localize("UTC")
                        ccgt_rows[dt] = ccgt_rows.get(dt, 0.0) + float(generation)
            except Exception as e:
                logger.warning("Elexon FUELHH: failed for chunk %s to %s: %s", chunk_from_str, chunk_to_str, e)

            chunk_start = chunk_end + pd.Timedelta(days=1)

    except Exception as e:
        logger.warning("Elexon CCGT generation: fetch failed: %s", e)
        return pd.DataFrame()

    if not ccgt_rows:
        logger.warning("Elexon FUELHH: no CCGT data found for %s to %s", date_from, date_to)
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"datetime": dt, "ccgt_generation_mw": gen} for dt, gen in ccgt_rows.items()]
    ).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info("Elexon CCGT generation: fetched %d records (%s to %s)", len(df), df.index[0], df.index[-1])
    return df


def fetch_interconnector_flows(date_from, date_to):
    """Fetch half-hourly net interconnector flows from Elexon BMRS.

    Sums all individual interconnector generation values per settlement period.
    Returns DataFrame with 'net_interconnector_mw' column (positive = import to GB).
    Iterates in 7-day chunks to stay within API limits.
    Returns empty DataFrame on failure.
    """
    flow_rows = {}
    try:
        start = pd.Timestamp(date_from).normalize()
        end = pd.Timestamp(date_to).normalize() + pd.Timedelta(days=1)
        logger.info("Elexon interconnectors: fetching %s to %s", date_from, date_to)
        # Iterate in 2-day chunks (7-day returns very large responses)
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=2), end)
            from_str = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            to_str = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
            url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/interconnectors"
            params = {
                "from": from_str,
                "to": to_str,
                "format": "json",
            }
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                logger.info("Elexon interconnectors: chunk %s→%s: %d records", from_str[:10], to_str[:10], len(data))

                for record in data:
                    start_time = record.get("startTime")
                    generation = record.get("generation")
                    if start_time and generation is not None:
                        dt = pd.Timestamp(start_time)
                        if dt.tzinfo is None:
                            dt = dt.tz_localize("UTC")
                        else:
                            dt = dt.tz_convert("UTC")
                        flow_rows[dt] = flow_rows.get(dt, 0.0) + float(generation)
            except Exception as e:
                logger.warning(
                    "Elexon interconnectors: failed for chunk %s to %s: %s",
                    from_str, to_str, e,
                )

            chunk_start = chunk_end

    except Exception as e:
        logger.warning("Elexon interconnector flows: fetch failed: %s", e)
        return pd.DataFrame()

    if not flow_rows:
        logger.warning("Elexon interconnectors: no data found for %s to %s", date_from, date_to)
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"datetime": dt, "net_interconnector_mw": flow} for dt, flow in flow_rows.items()]
    ).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info(
        "Elexon interconnectors: fetched %d records (%s to %s)",
        len(df), df.index[0], df.index[-1],
    )
    return df


def fetch_derated_margin(date_from, date_to):
    """Fetch half-hourly derated margin from Elexon BMRS MELNGC dataset.

    Extracts national boundary (N) and nearest boundary (B1) margin values.
    Iterates in 7-day chunks to stay within API limits.

    Returns DataFrame with columns:
        - derated_margin_mw: National boundary margin (MW)
        - margin_nearest_mw: Nearest boundary (B1) margin (MW)
    Indexed by UTC datetime. Returns empty DataFrame on failure.
    """
    n_rows = {}
    b1_rows = {}
    try:
        start = pd.Timestamp(date_from).normalize()
        end = pd.Timestamp(date_to).normalize()
        logger.info("Elexon MELNGC derated margin: fetching %s to %s", date_from, date_to)

        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + pd.Timedelta(days=6), end)
            from_str = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            to_str = (chunk_end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MELNGC"
            params = {
                "from": from_str,
                "to": to_str,
                "format": "json",
            }
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                logger.info(
                    "Elexon MELNGC: chunk %s→%s: %d records",
                    from_str[:10], to_str[:10], len(data),
                )

                for record in data:
                    boundary = record.get("boundary")
                    margin = record.get("margin")
                    settlement_date = record.get("settlementDate")
                    settlement_period = record.get("settlementPeriod")
                    if settlement_date and settlement_period is not None and margin is not None:
                        dt = pd.Timestamp(settlement_date) + (int(settlement_period) - 1) * pd.Timedelta("30min")
                        dt = dt.tz_localize("UTC")
                        if boundary == "N":
                            n_rows[dt] = float(margin)
                        elif boundary == "B1":
                            b1_rows[dt] = float(margin)
            except Exception as e:
                logger.warning(
                    "Elexon MELNGC: failed for chunk %s to %s: %s",
                    from_str, to_str, e,
                )

            chunk_start = chunk_end + pd.Timedelta(days=1)

    except Exception as e:
        logger.warning("Elexon MELNGC derated margin: fetch failed: %s", e)
        return pd.DataFrame()

    if not n_rows and not b1_rows:
        logger.warning("Elexon MELNGC: no margin data found for %s to %s", date_from, date_to)
        return pd.DataFrame()

    # Build DataFrame with both columns
    all_dts = sorted(set(list(n_rows.keys()) + list(b1_rows.keys())))
    rows = []
    for dt in all_dts:
        rows.append({
            "datetime": dt,
            "derated_margin_mw": n_rows.get(dt, float("nan")),
            "margin_nearest_mw": b1_rows.get(dt, float("nan")),
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info(
        "Elexon MELNGC derated margin: fetched %d records (%s to %s)",
        len(df), df.index[0], df.index[-1],
    )
    return df


def fetch_day_ahead_auction(date_from, date_to):
    """Fetch EPEX day-ahead auction prices from Elexon MID dataset.

    Returns Series of day-ahead prices (GBP/MWh) indexed by UTC datetime.
    Only includes APXMIDP (EPEX) records.
    Returns empty Series on failure or when no data is available
    (e.g. before ~12:00 UK time, or on bank holidays).
    """
    auction_rows = {}
    try:
        from_str = pd.Timestamp(date_from).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = pd.Timestamp(date_to).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("Elexon MID auction: fetching %s to %s", date_from, date_to)

        url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
        params = {
            "from": from_str,
            "to": to_str,
            "format": "json",
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        for record in data:
            if record.get("dataProvider") != "APXMIDP":
                continue
            start_time = record.get("startTime")
            price = record.get("price")
            if start_time and price is not None:
                dt = pd.Timestamp(start_time)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC")
                else:
                    dt = dt.tz_convert("UTC")
                auction_rows[dt] = float(price)

    except Exception as e:
        logger.warning("Elexon MID auction: fetch failed: %s", e)
        return pd.Series(dtype=float, name="day_ahead_auction")

    if not auction_rows:
        logger.info("Elexon MID auction: no APXMIDP data for %s to %s (auction may not have run yet)", date_from, date_to)
        return pd.Series(dtype=float, name="day_ahead_auction")

    series = pd.Series(auction_rows, name="day_ahead_auction").sort_index()
    series = series[~series.index.duplicated(keep="last")]
    logger.info("Elexon MID auction: fetched %d APXMIDP records (%s to %s)", len(series), series.index[0], series.index[-1])
    return series


def get_gb60():
    # url = "https://www.nordpoolgroup.com/api/marketdata/page/325?currency=GBP"
    url = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"

    params = {
        "date": (pd.Timestamp.now() + pd.Timedelta("13h")).strftime("%Y-%m-%d"),
        "market": "N2EX_DayAhead",
        "deliveryArea": "UK",
        "currency": "GBP",
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

    except requests.exceptions.RequestException as e:
        return

    price = pd.Series(
        {
            pd.Timestamp(row["deliveryStart"]).tz_convert("GB"): float(row["entryPerArea"]["UK"])
            for row in r.json()["multiAreaEntries"]
        }
    )
    return price


def _oct_time(d):
    # print(d)
    return datetime(
        year=pd.Timestamp(d).year,
        month=pd.Timestamp(d).month,
        day=pd.Timestamp(d).day,
    )


def queryset_to_df(queryset):
    df = pd.DataFrame(list(queryset.values()))
    df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
    df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
    # df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)
    df.index = pd.to_datetime(df["date_time"])
    df.index = df.index.tz_convert("GB")
    df.drop(["id", "date_time"], axis=1, inplace=True)

    return df


def get_history_from_model():
    if History.objects.count() == 0:
        df = pd.DataFrame()
    else:
        queryset = History.objects.all()
        df = queryset_to_df(queryset=queryset)

    return df.sort_index()


def get_forecast_from_model(forecast):
    if Forecasts.objects.count() == 0:
        df = pd.DataFrame()
    else:
        queryset = ForecastData.objects.filter(forecast=forecast)
        if queryset.count() > 0:
            df = queryset_to_df(queryset=queryset)
        else:
            df = pd.DataFrame()

    return df.sort_index()


def get_latest_history(start):
    delta = int((pd.Timestamp(start) - pd.Timestamp("2023-07-01", tz="GB")).total_seconds() / 1800)
    history_data = [
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "bf5ab335-9b40-4ea4-b93a-ab4af7bce003" WHERE "SETTLEMENT_DATE" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "SETTLEMENT_DATE",
            "period_col": "SETTLEMENT_PERIOD",
            "cols": "ND",
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "f6d02c0f-957b-48cb-82ee-09003f2ba759" WHERE "SETTLEMENT_DATE" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "SETTLEMENT_DATE",
            "period_col": "SETTLEMENT_PERIOD",
            "cols": "ND",
        },
        {
            "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/INDO?format=json",
            "params": {
                "publishDateTimeFrom": (pd.Timestamp.now() - pd.Timedelta("27D")).strftime("%Y-%m-%d"),
                "publishDateTimeTo": (pd.Timestamp.now() + pd.Timedelta("1D")).strftime("%Y-%m-%d"),
            },
            "record_path": ["data"],
            "date_col": "startTime",
            "cols": ["demand"],
            "rename": ["ND"],
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "7524ec65-f782-4258-aaf8-5b926c17b966" WHERE "Datetime_GMT" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 40000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "Datetime_GMT",
            "tz": "UTC",
            "cols": ["Incentive_forecast"],
            "rename": ["bm_wind"],
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "f93d1835-75bc-43e5-84ad-12472b180a98" WHERE "DATETIME" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "DATETIME",
            "cols": ["SOLAR", "WIND"],
            "rename": ["solar", "total_wind"],
        },
        {
            "url": "https://archive-api.open-meteo.com/v1/archive",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp.now().normalize().strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            },
            "record_path": ["hourly"],
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m", "wind_10m", "rad"],
        },
        {
            "url": "https://api.open-meteo.com/v1/forecast",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "start_date": (pd.Timestamp.now().normalize() - pd.Timedelta("5D")).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp.now().normalize().strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            },
            "record_path": ["hourly"],
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m_f", "wind_10m_f", "rad_f"],
        },
    ]

    downloaded_data = []
    download_errors = []

    for x in history_data:
        data, e = DataSet(**x).download()
        if len(data) > 0:
            downloaded_data += [data]
        else:
            download_errors += [e]

    hist = pd.concat(downloaded_data, axis=1).loc[: pd.Timestamp.now(tz="GB")]
    # print(hist.iloc[-48:].to_string())

    if isinstance(hist["ND"], pd.DataFrame):
        hist["demand"] = hist["ND"].mean(axis=1)
    else:
        hist["demand"] = hist["ND"]
    hist.index = pd.to_datetime(hist.index)
    hist = hist.drop("ND", axis=1).sort_index()

    meteo_cols = ["temp_2m", "wind_10m", "rad"]

    for c in [m for m in meteo_cols if m in hist.columns]:
        hist.loc[hist[c].isnull(), c] = hist.loc[hist[c].isnull(), f"{c}_f"]

    hist = hist.drop([f"{c}_f" for c in meteo_cols if c in hist.columns], axis=1)

    all_cols = ["total_wind", "bm_wind", "solar", "demand"] + meteo_cols
    missing_cols = [c for c in all_cols if c not in hist.columns]
    if len(missing_cols) > 0:
        logger.error(f">>> ERROR: No historic data for {missing_cols} ")
        return pd.DataFrame(), missing_cols
    else:
        return hist.astype(float).dropna(), missing_cols


def get_latest_forecast():
    ndf_from = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    ndf_to = (pd.Timestamp.now().normalize() + pd.Timedelta("24h")).strftime("%Y-%m-%d")

    forecast_data = [
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=93c3048e-1dab-4057-a2a9-417540583929&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "date_col": "Datetime",
            "cols": ["Wind_Forecast"],
            "rename": ["bm_wind"],
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=b2f03146-f05d-4824-a663-3a4f36090c71&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "date_col": "Datetime_GMT",
            "cols": ["Incentive_forecast"],
            "rename": ["da_wind"],
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=db6c038f-98af-4570-ab60-24d71ebd0ae5&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "cols": ["EMBEDDED_SOLAR_FORECAST", "EMBEDDED_WIND_FORECAST"],
            "rename": ["solar", "emb_wind"],
            "date_col": "DATE_GMT",
            "time_col": "TIME_GMT",
        },
        {
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=7c0411cd-2714-4bb5-a408-adb065edf34d&limit=5000",
            "record_path": ["result", "records"],
            "date_col": "GDATETIME",
            "tz": "UTC",
            "cols": ["NATIONALDEMAND"],
        },
        {
            "url": "https://api.open-meteo.com/v1/forecast",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "current": "temperature_2m",
                "minutely_15": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
                "forecast_days": 14,
            },
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "record_path": ["minutely_15"],
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m", "wind_10m", "rad"],
        },
        {
            # "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF?publishDateTimeFrom={ndf_from}&publishDateTimeTo={ndf_to}",
            "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF",
            "params": {"publishDateTimeFrom": ndf_from, "publishDateTimeTo": ndf_to},
            "record_path": ["data"],
            "date_col": "startTime",
            "cols": "demand",
            "sort_col": "publishTime",
        },
    ]

    downloaded_data = []
    download_errors = []

    for x in forecast_data:
        data, e = DataSet(**x).download()
        if len(data) > 0:
            downloaded_data += [data]
            # print(f"{x}:\n{data}\n\n")
        else:
            download_errors += [e]

    df = pd.concat(downloaded_data, axis=1)

    demand_cols = ["demand", "NATIONALDEMAND"]
    if all([c in df.columns for c in demand_cols]):
        df["demand"] = df[demand_cols].mean(axis=1)
        df.drop(["NATIONALDEMAND"], axis=1, inplace=True)
        missing_cols = []
    elif "NATIONALDEMAND" not in df.columns:
        missing_cols = ["NATIONALDEMAND"]
    else:
        missing_cols = []

    df.loc[df["da_wind"] > 0, "bm_wind"] = df["da_wind"]
    df.drop("da_wind", axis=1, inplace=True)

    all_cols = ["emb_wind", "bm_wind", "solar", "demand", "temp_2m", "wind_10m", "rad"]
    missing_cols += [c for c in all_cols if c not in df.columns]
    if len(missing_cols) > 0:
        print(f">>> ERROR: No forecast data for {missing_cols} ")
        return pd.DataFrame(), missing_cols
    else:
        df["date_time"] = pd.to_datetime(df.index)
        df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
        df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
        # df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)

        df.index = pd.to_datetime(df.index).tz_convert("GB")
        df.drop(["date_time"], axis=1, inplace=True)

        return df.sort_index().dropna(), missing_cols


class DataSet:
    def __init__(self, *args, **kwargs) -> None:
        self.params = kwargs.pop("params", {})
        self.tz = kwargs.pop("tz", "UTC")
        self.__dict__ = self.__dict__ | kwargs

        # self.__dict__ = self.__dict__ | kwargs

    def update(self, download_all=False, hdf=None):
        pass

    def download(self, tz="GB", params={}):
        logger.info(f"    {self.url}")
        for n in range(RETRIES):
            try:
                response = requests.get(url=self.url, params=self.params)
                response.raise_for_status()
                code = None
                break

            except HTTPError as exc:
                code = exc.response.status_code

                if code in RETRY_CODES:
                    # retry after n seconds
                    time.sleep(n)
                    continue

        try:
            df = pd.json_normalize(response.json(), self.record_path)
        except:
            try:
                df = pd.DataFrame(response.json()[self.record_path[0]])
            except Exception as e:
                print(f">>> ERROR {e} for URL {self.url}\n>>> with params {self.params}")
                return pd.DataFrame(), code

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i = 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df.index = pd.to_datetime(df[self.date_col])
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize(self.tz, ambiguous="infer")
        except Exception as e:
            print(f">>> Error: {e}")
            print(df.index)

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df.index = pd.to_datetime(df["Date"]) + (df["Settlement_period"] - 1) * pd.Timedelta("30min")
            df.index = df.index.tz_localize("UTC")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df.index += pd.to_datetime(df[self.time_col].str[:5], format="%H:%M") - pd.Timestamp("1900-01-01")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df.index += (df[self.period_col] - 1) * pd.Timedelta("30min")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df.index = df.index.tz_convert(tz)
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df = df[self.cols]
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            if "func" in self.__dict__:
                df = df.resample(self.resample).aggregate(self.func)
            elif "resample" in self.__dict__:
                df = df.resample(self.resample).mean()
        except Exception as e:

            print(e)

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df = df.interpolate()
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        try:
            df = df.sort_values(self.sort_col)
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        if isinstance(df, pd.DataFrame):
            try:
                df = df.set_axis(self.rename, axis=1)
            except:
                pass
        elif isinstance(df, pd.Series):
            try:
                df = df.rename(self.rename)
            except:
                pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.info(f"{i}:\n{df.iloc[:30]}")

        df = df.sort_index()
        df = df[~df.index.duplicated()]
        return df, None


def get_agile_pages(start=pd.Timestamp("2023-07-01"), tz="GB", region="G"):
    """Yield one page of Agile prices at a time (for incremental DB writes).

    Each yielded value is a pandas Series named "agile" covering up to 1500
    half-hour slots. Retries on HTTP 429 with exponential backoff; logs
    progress per page so callers can monitor long backfills.
    """
    start = pd.Timestamp(start).tz_convert("UTC")
    product = "AGILE-24-10-01"
    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta("48h")
    code = f"E-1R-{product}-{region}"
    url = OCTOPUS_PRODUCT_URL + f"{product}/electricity-tariffs/{code}/standard-unit-rates/"

    _PAGE_DELAY = 2      # seconds between successful page fetches
    _MAX_RETRIES = 5     # max retries per page on 429
    _RETRY_DELAY = 60    # base seconds to wait on 429 (multiplied by attempt)
    page_num = 0

    while end > start:
        page_num += 1
        params = {
            "page_size": 1500,
            "order_by": "period",
            "period_from": _oct_time(start),
            "period_to": _oct_time(end),
        }
        logger.info("Page %d: fetching %s → %s", page_num, _oct_time(start), _oct_time(end))

        for attempt in range(_MAX_RETRIES):
            r = requests.get(url, params=params, headers={"User-Agent": "curl/7.64.1"})
            if r.status_code == 429:
                wait = _RETRY_DELAY * (attempt + 1)
                logger.warning("Octopus API 429 on attempt %d/%d — waiting %ds", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                break
        else:
            raise RuntimeError(f"Octopus API returned 429 after {_MAX_RETRIES} retries — aborting")

        results = r.json().get("results", [])
        if not results:
            logger.info("Page %d: no results returned — backfill complete", page_num)
            break

        logger.info("Page %d: got %d records up to %s", page_num, len(results), results[-1]["valid_from"])

        page_df = pd.DataFrame(results).set_index("valid_from")[["value_inc_vat"]]
        page_df.index = pd.to_datetime(page_df.index).tz_convert(tz)
        page_df = page_df.sort_index()["value_inc_vat"]
        page_df = page_df[~page_df.index.duplicated()]
        yield page_df.rename("agile")

        end = pd.Timestamp(results[-1]["valid_from"]).ceil("24h")
        time.sleep(_PAGE_DELAY)


def get_agile(start=pd.Timestamp("2023-07-01"), tz="GB", region="G"):
    """Fetch all Agile prices from start to now as a single Series."""
    pages = list(get_agile_pages(start=start, tz=tz, region=region))
    if not pages:
        return pd.Series(name="agile", dtype=float)
    df = pd.concat(pages).sort_index()
    return df[~df.index.duplicated()]


def day_ahead_to_agile(df, reverse=False, region="G"):
    df.index = df.index.tz_convert("GB")
    x = pd.DataFrame(df).set_axis(["In"], axis=1)
    x["Out"] = x["In"]
    x["Peak"] = (x.index.hour >= 16) & (x.index.hour < 19)
    if reverse:
        x.loc[x["Peak"], "Out"] -= regions[region]["factors"][1]
        x["Out"] /= regions[region]["factors"][0]
    else:
        # print(region)
        x["Out"] *= regions[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += regions[region]["factors"][1]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


def df_to_Model(df, myModel, update=False):
    # df = df.dropna()
    for index, row in df.iterrows():
        if update:
            try:
                obj = myModel.objects.get(date_time=index)
                for key, value in row.items():
                    setattr(obj, key, value)
                obj.save()
            except myModel.DoesNotExist:
                new_values = {"date_time": index}
                new_values.update(row)
                obj = myModel(**new_values)
                obj.save()
        else:
            try:
                new_values = {"date_time": index}
                new_values.update(row)
                obj = myModel(**new_values)
                obj.save()
            except Exception as e:
                print(f"Failed to update {myModel} with data for datetime {index}: {e}")


def model_to_df(myModel):
    df = pd.DataFrame(list(myModel.objects.all().values()))
    start = pd.Timestamp("2023-07-01", tz="GB")
    if len(df) > 0:
        df.index = pd.to_datetime(df["date_time"])
        df = df.sort_index()
        df.index = df.index.tz_convert("GB")
        df.drop(["id", "date_time"], axis=1, inplace=True)
        start = df.index[-1] + pd.Timedelta("30min")
    return df, start
