"""Property tests for derated margin (MELNGC) boundary extraction.

Feature: agile-predict-advanced-features, Property 1: MELNGC Boundary Extraction
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def extract_boundaries(records: list[dict]) -> pd.DataFrame:
    """Replicate the boundary extraction logic from fetch_derated_margin."""
    n_rows = {}
    b1_rows = {}
    for record in records:
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

    if not n_rows and not b1_rows:
        return pd.DataFrame()

    all_dts = sorted(set(list(n_rows.keys()) + list(b1_rows.keys())))
    df = pd.DataFrame(index=all_dts)
    df["derated_margin_mw"] = pd.Series(n_rows)
    df["margin_nearest_mw"] = pd.Series(b1_rows)
    return df


def _make_record(boundary, margin, date, period):
    return {
        "boundary": boundary,
        "margin": margin,
        "settlementDate": date,
        "settlementPeriod": period,
    }


@given(
    n_margin=st.floats(min_value=-20000, max_value=50000),
    b1_margin=st.floats(min_value=-20000, max_value=50000),
    b2_margin=st.floats(min_value=-20000, max_value=50000),
    period=st.integers(min_value=1, max_value=48),
)
@settings(max_examples=100)
def test_extracts_only_n_and_b1_boundaries(n_margin, b1_margin, b2_margin, period):
    """Property 1: Extracts exactly N into derated_margin_mw and B1 into margin_nearest_mw.

    Test Case:
    test_derated_margin\\test_extracts_only_n_and_b1_boundaries

    Purpose:
    Validates that the boundary extraction logic correctly separates
    N (national) and B1 (nearest) boundaries, ignoring all others.

    Test Conditions:
    - Records with N, B1, and B2 boundaries for the same settlement period
    - Random margin values in realistic MW range

    Key Properties:
        For any MELNGC response with N/B1/B2+ boundaries,
        extract_boundaries returns exactly N in derated_margin_mw
        and B1 in margin_nearest_mw, ignoring B2+.

    Expected Behaviour:
    - derated_margin_mw contains only the N boundary value
    - margin_nearest_mw contains only the B1 boundary value
    - B2 boundary is not present in either column
    """
    records = [
        _make_record("N", n_margin, "2026-04-06", period),
        _make_record("B1", b1_margin, "2026-04-06", period),
        _make_record("B2", b2_margin, "2026-04-06", period),
    ]
    df = extract_boundaries(records)

    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    assert df["derated_margin_mw"].iloc[0] == pytest.approx(n_margin)
    assert df["margin_nearest_mw"].iloc[0] == pytest.approx(b1_margin)


def test_empty_records_returns_empty_dataframe():
    """Edge case: no records returns empty DataFrame.

    Test Case:
    test_derated_margin\\test_empty_records_returns_empty_dataframe

    Purpose:
    Validates graceful handling of empty API response.

    Test Conditions:
    - Empty record list

    Key Properties:
        Empty input produces empty DataFrame.

    Expected Behaviour:
    - Returns empty DataFrame
    """
    df = extract_boundaries([])
    assert len(df) == 0


def test_ignores_records_with_missing_fields():
    """Edge case: records with missing fields are skipped.

    Test Case:
    test_derated_margin\\test_ignores_records_with_missing_fields

    Purpose:
    Validates that malformed records don't cause errors.

    Test Conditions:
    - Records missing margin, settlementDate, or settlementPeriod

    Key Properties:
        Malformed records are silently skipped.

    Expected Behaviour:
    - Only valid records are extracted
    """
    records = [
        {"boundary": "N", "margin": None, "settlementDate": "2026-04-06", "settlementPeriod": 1},
        {"boundary": "N", "margin": 5000, "settlementDate": None, "settlementPeriod": 1},
        {"boundary": "N", "margin": 5000, "settlementDate": "2026-04-06", "settlementPeriod": None},
        {"boundary": "N", "margin": 3000, "settlementDate": "2026-04-06", "settlementPeriod": 2},
    ]
    df = extract_boundaries(records)
    assert len(df) == 1
    assert df["derated_margin_mw"].iloc[0] == pytest.approx(3000)
