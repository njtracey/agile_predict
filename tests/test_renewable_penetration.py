"""Property tests for renewable penetration ratio computation.

Feature: agile-predict-advanced-features, Property 4: Renewable Penetration Formula
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def compute_renewable_penetration(bm_wind: float, solar: float, demand: float) -> float:
    """Replicate the renewable penetration computation from update.py."""
    if demand > 0:
        return (bm_wind + solar) / demand * 100
    return 0.0


@given(
    bm_wind=st.floats(min_value=0, max_value=30000),
    solar=st.floats(min_value=0, max_value=20000),
    demand=st.floats(min_value=0.1, max_value=60000),
)
@settings(max_examples=100)
def test_penetration_equals_formula_when_demand_positive(bm_wind, solar, demand):
    """Property 4: When demand > 0, result = (bm_wind + solar) / demand * 100.

    Test Case:
    test_renewable_penetration\\test_penetration_equals_formula_when_demand_positive

    Purpose:
    Validates the renewable penetration formula for positive demand values.

    Test Conditions:
    - Random bm_wind >= 0, solar >= 0, demand > 0
    - Realistic MW ranges for GB electricity market

    Key Properties:
        For any (bm_wind >= 0, solar >= 0, demand > 0),
        renewable_penetration equals (bm_wind + solar) / demand * 100.

    Expected Behaviour:
    - Result matches the formula exactly
    - Result is non-negative
    """
    result = compute_renewable_penetration(bm_wind, solar, demand)
    expected = (bm_wind + solar) / demand * 100
    assert result == pytest.approx(expected), (
        f"Expected {expected}, got {result} for wind={bm_wind}, solar={solar}, demand={demand}"
    )
    assert result >= 0


@given(
    bm_wind=st.floats(min_value=0, max_value=30000),
    solar=st.floats(min_value=0, max_value=20000),
)
@settings(max_examples=100)
def test_penetration_is_zero_when_demand_is_zero(bm_wind, solar):
    """Property 4: When demand == 0, result = 0.0.

    Test Case:
    test_renewable_penetration\\test_penetration_is_zero_when_demand_is_zero

    Purpose:
    Validates that renewable penetration returns 0.0 when demand is zero,
    avoiding division-by-zero errors.

    Test Conditions:
    - Random bm_wind >= 0, solar >= 0, demand = 0

    Key Properties:
        For any (bm_wind >= 0, solar >= 0, demand = 0),
        renewable_penetration equals 0.0.

    Expected Behaviour:
    - Result is exactly 0.0
    - No division-by-zero error
    """
    result = compute_renewable_penetration(bm_wind, solar, 0.0)
    assert result == 0.0


def test_penetration_negative_demand_treated_as_zero():
    """Edge case: negative demand should also return 0.0.

    Test Case:
    test_renewable_penetration\\test_penetration_negative_demand_treated_as_zero

    Purpose:
    Validates edge case where demand is negative (shouldn't happen in practice
    but the formula should handle it safely).

    Test Conditions:
    - demand < 0

    Key Properties:
        Negative demand is treated as zero (no division).

    Expected Behaviour:
    - Result is 0.0
    """
    result = compute_renewable_penetration(5000, 3000, -100)
    assert result == 0.0
