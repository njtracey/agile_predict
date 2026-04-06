#!/usr/bin/env python3
"""Comprehensive model performance analysis for LLM interpretation.

Connects directly to db.sqlite3 using pure sqlite3 + pandas (no Django dependency).
Outputs structured JSON (or formatted text) to stdout.

Usage:
    cd agile_predict
    venv/bin/python bin/analyze_model_performance.py [--days 30] [--format json|text]
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

# Region F Agile pricing factors
REGION_F_MULT = 0.21
REGION_F_ADDER = 12  # peak adder (16:00-19:00)

# Horizon bucket definitions (hours)
HORIZON_BUCKETS = [
    ("0-12h", 0, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
    ("2-4d", 48, 96),
    ("4-7d", 96, 168),
    ("7-14d", 168, 336),
]

MIN_PAIRS_FOR_METRIC = 5


def get_db_path():
    """Return path to db.sqlite3 relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "db.sqlite3")
    return os.path.abspath(db_path)


def table_exists(conn, table_name):
    """Check if a table exists in the database."""
    cur = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone()[0] > 0


def safe_float(val):
    """Convert to float, returning None for NaN/Inf."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return round(float(val), 4)


def compute_accuracy_metrics(predicted, actual, low=None, high=None):
    """Compute MAE, RMSE, bias, coverage from paired arrays."""
    if len(predicted) < MIN_PAIRS_FOR_METRIC:
        return {"status": "insufficient_data", "n": len(predicted)}

    errors = predicted - actual
    mae = safe_float(np.mean(np.abs(errors)))
    rmse = safe_float(np.sqrt(np.mean(errors**2)))
    bias = safe_float(np.mean(errors))
    n = int(len(predicted))

    result = {"mae": mae, "rmse": rmse, "bias": bias, "n": n}

    if low is not None and high is not None:
        covered = ((actual >= low) & (actual <= high)).sum()
        result["coverage_pct"] = safe_float(100.0 * covered / n)

    return result


def build_metadata(db_path, days):
    """Build metadata section."""
    return {
        "script_version": SCRIPT_VERSION,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "database_path": db_path,
        "analysis_window_days": days,
    }


def build_database_summary(conn):
    """Build database summary with record counts and date ranges."""
    summary = {}

    tables = {
        "price_history": "prices_pricehistory",
        "forecasts": "prices_forecasts",
        "agile_data": "prices_agiledata",
        "official_agile_data": "prices_officialagiledata",
        "forecast_data": "prices_forecastdata",
        "history": "prices_history",
    }

    for key, table in tables.items():
        if not table_exists(conn, table):
            summary[key] = {"count": 0, "note": f"Table {table} does not exist"}
            continue

        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        entry = {"count": count}

        if count > 0:
            date_col = "created_at" if table == "prices_forecasts" else "date_time"
            row = conn.execute(
                f"SELECT MIN({date_col}), MAX({date_col}) FROM {table}"
            ).fetchone()
            entry["min_date"] = row[0]
            entry["max_date"] = row[1]

        summary[key] = entry

    # Extra: count forecasts with training scores
    scored = conn.execute(
        "SELECT COUNT(*) FROM prices_forecasts WHERE mean IS NOT NULL AND mean != 0"
    ).fetchone()[0]
    summary["forecasts"]["with_training_scores"] = scored

    # Extra: AgileData region F count
    if table_exists(conn, "prices_agiledata"):
        region_f = conn.execute(
            "SELECT COUNT(*) FROM prices_agiledata WHERE region='F'"
        ).fetchone()[0]
        summary["agile_data"]["region_f_count"] = region_f

    return summary


def compute_model_accuracy(conn, days, source="ours"):
    """Compute accuracy metrics for our model or official model.

    Joins predictions with PriceHistory actuals on date_time.
    """
    pred_table = "prices_agiledata" if source == "ours" else "prices_officialagiledata"

    if not table_exists(conn, pred_table):
        return {"status": "no_data", "note": f"{pred_table} table does not exist"}

    # Check if table has data
    count = conn.execute(f"SELECT COUNT(*) FROM {pred_table}").fetchone()[0]
    if count == 0:
        return {
            "status": "no_data",
            "note": f"{'OfficialAgileData' if source == 'official' else pred_table} not yet populated",
        }

    # Build the query — join predictions with actuals
    cutoff = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")

    region_filter = "AND p.region = 'F'" if source == "ours" else ""

    query = f"""
        SELECT
            p.agile_pred,
            p.agile_low,
            p.agile_high,
            p.date_time AS pred_dt,
            ph.agile AS actual,
            f.name AS forecast_name,
            f.created_at AS forecast_created
        FROM {pred_table} p
        JOIN prices_pricehistory ph ON p.date_time = ph.date_time
        JOIN prices_forecasts f ON p.forecast_id = f.id
        WHERE ph.date_time >= ?
        {region_filter}
    """

    df = pd.read_sql_query(query, conn, params=[cutoff])

    if len(df) < MIN_PAIRS_FOR_METRIC:
        return {"status": "insufficient_data", "n": len(df)}

    # Parse timestamps for horizon computation
    df["pred_dt"] = pd.to_datetime(df["pred_dt"])
    df["forecast_created"] = pd.to_datetime(df["forecast_created"])
    df["horizon_hours"] = (
        df["pred_dt"] - df["forecast_created"]
    ).dt.total_seconds() / 3600

    # Overall metrics
    overall = compute_accuracy_metrics(
        df["agile_pred"].values,
        df["actual"].values,
        df["agile_low"].values,
        df["agile_high"].values,
    )

    # Per-horizon metrics
    by_horizon = {}
    for label, h_min, h_max in HORIZON_BUCKETS:
        mask = (df["horizon_hours"] >= h_min) & (df["horizon_hours"] < h_max)
        subset = df[mask]
        if len(subset) == 0:
            by_horizon[label] = {"status": "no_data", "n": 0}
        else:
            by_horizon[label] = compute_accuracy_metrics(
                subset["agile_pred"].values,
                subset["actual"].values,
                subset["agile_low"].values,
                subset["agile_high"].values,
            )

    return {"overall": overall, "by_horizon": by_horizon}


def compute_head_to_head(conn, days):
    """Compare our model vs official per horizon bucket."""
    if not table_exists(conn, "prices_officialagiledata"):
        return {"status": "no_data", "note": "OfficialAgileData not yet populated"}

    official_count = conn.execute(
        "SELECT COUNT(*) FROM prices_officialagiledata"
    ).fetchone()[0]
    if official_count == 0:
        return {"status": "no_data", "note": "OfficialAgileData not yet populated"}

    cutoff = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Find matched triples: same forecast, same target slot, both have predictions
    query = """
        SELECT
            ours.agile_pred AS our_pred,
            official.agile_pred AS official_pred,
            ph.agile AS actual,
            ours.date_time AS target_dt,
            f.created_at AS forecast_created
        FROM prices_agiledata ours
        JOIN prices_officialagiledata official
            ON ours.forecast_id = official.forecast_id
            AND ours.date_time = official.date_time
        JOIN prices_pricehistory ph ON ours.date_time = ph.date_time
        JOIN prices_forecasts f ON ours.forecast_id = f.id
        WHERE ours.region = 'F'
        AND ph.date_time >= ?
    """

    df = pd.read_sql_query(query, conn, params=[cutoff])

    if len(df) < MIN_PAIRS_FOR_METRIC:
        return {"status": "insufficient_data", "n": len(df)}

    df["target_dt"] = pd.to_datetime(df["target_dt"])
    df["forecast_created"] = pd.to_datetime(df["forecast_created"])
    df["horizon_hours"] = (
        df["target_dt"] - df["forecast_created"]
    ).dt.total_seconds() / 3600

    by_horizon = {}
    for label, h_min, h_max in HORIZON_BUCKETS:
        mask = (df["horizon_hours"] >= h_min) & (df["horizon_hours"] < h_max)
        subset = df[mask]
        if len(subset) < MIN_PAIRS_FOR_METRIC:
            by_horizon[label] = {"status": "insufficient_data", "n": len(subset)}
            continue

        our_mae = safe_float(np.mean(np.abs(subset["our_pred"] - subset["actual"])))
        off_mae = safe_float(np.mean(np.abs(subset["official_pred"] - subset["actual"])))
        delta = safe_float(our_mae - off_mae) if our_mae and off_mae else None
        winner = "ours" if delta and delta < 0 else "official" if delta and delta > 0 else "tie"

        by_horizon[label] = {
            "our_mae": our_mae,
            "official_mae": off_mae,
            "delta_mae": delta,
            "winner": winner,
            "n": int(len(subset)),
        }

    return {"by_horizon": by_horizon, "total_matched_pairs": int(len(df))}


def compute_prediction_convergence(conn, days):
    """Show how accuracy improves as the same target slot is predicted from closer forecasts.

    Groups predictions by target slot, then by how far ahead the forecast was made.
    """
    cutoff = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")

    query = """
        SELECT
            p.agile_pred,
            p.date_time AS target_dt,
            f.created_at AS forecast_created,
            ph.agile AS actual
        FROM prices_agiledata p
        JOIN prices_pricehistory ph ON p.date_time = ph.date_time
        JOIN prices_forecasts f ON p.forecast_id = f.id
        WHERE p.region = 'F'
        AND ph.date_time >= ?
    """

    df = pd.read_sql_query(query, conn, params=[cutoff])

    if len(df) < MIN_PAIRS_FOR_METRIC:
        return {"status": "insufficient_data", "n": len(df)}

    df["target_dt"] = pd.to_datetime(df["target_dt"])
    df["forecast_created"] = pd.to_datetime(df["forecast_created"])
    df["horizon_hours"] = (
        df["target_dt"] - df["forecast_created"]
    ).dt.total_seconds() / 3600

    # Group by horizon bucket and compute MAE
    convergence = {}
    for label, h_min, h_max in HORIZON_BUCKETS:
        mask = (df["horizon_hours"] >= h_min) & (df["horizon_hours"] < h_max)
        subset = df[mask]
        if len(subset) < MIN_PAIRS_FOR_METRIC:
            convergence[label] = {"status": "insufficient_data", "n": len(subset)}
        else:
            errors = np.abs(subset["agile_pred"] - subset["actual"])
            convergence[label] = {
                "mae": safe_float(errors.mean()),
                "median_ae": safe_float(errors.median()),
                "n": int(len(subset)),
                "unique_target_slots": int(subset["target_dt"].nunique()),
            }

    return {"by_horizon": convergence}


def compute_training_state(conn):
    """Extract RMSE trend, training data volume, feature list from Forecasts."""
    query = """
        SELECT name, created_at, mean, stdev
        FROM prices_forecasts
        WHERE mean IS NOT NULL AND mean != 0
        ORDER BY created_at DESC
        LIMIT 20
    """
    df = pd.read_sql_query(query, conn)

    if len(df) == 0:
        return {"status": "no_data", "note": "No forecasts with training scores"}

    rmse_trend = []
    for _, row in df.iterrows():
        rmse_trend.append({
            "forecast_name": row["name"],
            "created_at": row["created_at"],
            "rmse": safe_float(row["mean"]),
            "stdev": safe_float(row["stdev"]),
        })

    # Training data volume
    fd_count = conn.execute("SELECT COUNT(*) FROM prices_forecastdata").fetchone()[0]

    # Feature list from ForecastData columns (proxy for what the model uses)
    cur = conn.execute("PRAGMA table_info(prices_forecastdata)")
    fd_cols = [
        r[1] for r in cur.fetchall()
        if r[1] not in ("id", "forecast_id", "date_time")
    ]

    return {
        "rmse_trend": rmse_trend,
        "training_data_volume": fd_count,
        "forecast_data_features": fd_cols,
    }


def compute_actual_price_statistics(conn, days):
    """Daily summaries of actual Agile prices from PriceHistory."""
    cutoff = (
        pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")

    query = """
        SELECT date_time, agile
        FROM prices_pricehistory
        WHERE date_time >= ?
        ORDER BY date_time
    """
    df = pd.read_sql_query(query, conn, params=[cutoff])

    if len(df) == 0:
        return {"status": "no_data"}

    df["date_time"] = pd.to_datetime(df["date_time"])
    df["date"] = df["date_time"].dt.date.astype(str)

    daily = df.groupby("date")["agile"].agg(["mean", "min", "max", "std", "count"])
    daily = daily.reset_index()

    recent_days = []
    for _, row in daily.tail(14).iterrows():
        recent_days.append({
            "date": row["date"],
            "mean": safe_float(row["mean"]),
            "min": safe_float(row["min"]),
            "max": safe_float(row["max"]),
            "stdev": safe_float(row["std"]),
            "slots": int(row["count"]),
        })

    return {"recent_days": recent_days}


def format_text_output(result):
    """Format the JSON result as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("  AGILE PREDICT MODEL PERFORMANCE ANALYSIS")
    lines.append("=" * 70)

    meta = result["metadata"]
    lines.append(f"\nScript v{meta['script_version']}  |  {meta['run_timestamp']}")
    lines.append(f"Database: {meta['database_path']}")
    lines.append(f"Analysis window: {meta['analysis_window_days']} days")

    # Database summary
    lines.append("\n--- Database Summary ---")
    for key, val in result["database_summary"].items():
        label = key.replace("_", " ").title()
        count = val.get("count", 0)
        extra = ""
        if "min_date" in val:
            extra = f"  ({val['min_date']} to {val['max_date']})"
        if "note" in val:
            extra = f"  [{val['note']}]"
        lines.append(f"  {label:30s}: {count:>8,d}{extra}")

    # Our model accuracy
    lines.append("\n--- Our Model Accuracy ---")
    ours = result["our_model_accuracy"]
    if "status" in ours:
        lines.append(f"  {ours['status']}: {ours.get('note', '')}")
    else:
        ov = ours["overall"]
        lines.append(
            f"  Overall: MAE={ov['mae']}, RMSE={ov['rmse']}, "
            f"Bias={ov['bias']}, Coverage={ov.get('coverage_pct', 'N/A')}%, n={ov['n']}"
        )
        lines.append("  By Horizon:")
        for h, m in ours["by_horizon"].items():
            if "status" in m:
                lines.append(f"    {h:10s}: {m['status']} (n={m.get('n', 0)})")
            else:
                lines.append(
                    f"    {h:10s}: MAE={m['mae']}, RMSE={m['rmse']}, "
                    f"Bias={m['bias']}, Cov={m.get('coverage_pct', 'N/A')}%, n={m['n']}"
                )

    # Official model accuracy
    lines.append("\n--- Official Model Accuracy ---")
    off = result["official_model_accuracy"]
    if "status" in off:
        lines.append(f"  {off['status']}: {off.get('note', '')}")
    else:
        ov = off["overall"]
        lines.append(
            f"  Overall: MAE={ov['mae']}, RMSE={ov['rmse']}, "
            f"Bias={ov['bias']}, Coverage={ov.get('coverage_pct', 'N/A')}%, n={ov['n']}"
        )
        lines.append("  By Horizon:")
        for h, m in off["by_horizon"].items():
            if "status" in m:
                lines.append(f"    {h:10s}: {m['status']} (n={m.get('n', 0)})")
            else:
                lines.append(
                    f"    {h:10s}: MAE={m['mae']}, RMSE={m['rmse']}, "
                    f"Bias={m['bias']}, Cov={m.get('coverage_pct', 'N/A')}%, n={m['n']}"
                )

    # Head to head
    lines.append("\n--- Head-to-Head Comparison ---")
    h2h = result["head_to_head"]
    if "status" in h2h:
        lines.append(f"  {h2h['status']}: {h2h.get('note', '')}")
    else:
        lines.append(f"  Total matched pairs: {h2h['total_matched_pairs']}")
        for h, m in h2h["by_horizon"].items():
            if "status" in m:
                lines.append(f"    {h:10s}: {m['status']} (n={m.get('n', 0)})")
            else:
                lines.append(
                    f"    {h:10s}: Ours={m['our_mae']}, Official={m['official_mae']}, "
                    f"Delta={m['delta_mae']}, Winner={m['winner']} (n={m['n']})"
                )

    # Convergence
    lines.append("\n--- Prediction Convergence ---")
    conv = result["prediction_convergence"]
    if "status" in conv:
        lines.append(f"  {conv['status']}")
    else:
        for h, m in conv["by_horizon"].items():
            if "status" in m:
                lines.append(f"    {h:10s}: {m['status']} (n={m.get('n', 0)})")
            else:
                lines.append(
                    f"    {h:10s}: MAE={m['mae']}, MedianAE={m['median_ae']}, "
                    f"n={m['n']}, slots={m['unique_target_slots']}"
                )

    # Training state
    lines.append("\n--- Training State ---")
    ts = result["training_state"]
    if "status" in ts:
        lines.append(f"  {ts['status']}")
    else:
        lines.append(f"  Training data volume: {ts['training_data_volume']:,d}")
        lines.append(f"  Features: {', '.join(ts['forecast_data_features'])}")
        lines.append("  Recent RMSE trend:")
        for entry in ts["rmse_trend"][:10]:
            lines.append(
                f"    {entry['forecast_name']:20s}  RMSE={entry['rmse']}  stdev={entry['stdev']}"
            )

    # Actual price stats
    lines.append("\n--- Actual Price Statistics (last 14 days) ---")
    aps = result["actual_price_statistics"]
    if "status" in aps:
        lines.append(f"  {aps['status']}")
    else:
        lines.append(f"  {'Date':12s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Stdev':>8s} {'Slots':>6s}")
        for d in aps["recent_days"]:
            lines.append(
                f"  {d['date']:12s} {d['mean'] or 0:8.2f} {d['min'] or 0:8.2f} "
                f"{d['max'] or 0:8.2f} {d['stdev'] or 0:8.2f} {d['slots']:6d}"
            )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse agile_predict model performance"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Analysis window in days (default: 30)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(json.dumps({"error": f"Database not found: {db_path}"}), file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    result = {
        "metadata": build_metadata(db_path, args.days),
        "database_summary": build_database_summary(conn),
        "our_model_accuracy": compute_model_accuracy(conn, args.days, source="ours"),
        "official_model_accuracy": compute_model_accuracy(
            conn, args.days, source="official"
        ),
        "head_to_head": compute_head_to_head(conn, args.days),
        "prediction_convergence": compute_prediction_convergence(conn, args.days),
        "training_state": compute_training_state(conn),
        "actual_price_statistics": compute_actual_price_statistics(conn, args.days),
    }

    conn.close()

    if args.format == "text":
        print(format_text_output(result))
    else:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
