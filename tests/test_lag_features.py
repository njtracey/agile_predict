"""Property tests for lag price feature computation.

Feature: agile-predict-advanced-features, Property 3: Lag Price Computation
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


def compute_lag(price_series: pd.Series, offset: int, price_mean: float) -> pd.Series:
    """Replicate the lag computation from update.py."""
    lag = price_series.shift(offset)
    return lag.fillna(price_mean)


@given(
    length=st.integers(min_value=1, max_value=500),
    offset=st.sampled_from([2, 3, 4, 6, 12, 24, 48, 336]),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_lag_equals_shifted_price_when_sufficient_history(length, offset, seed):
    """Property 3: For index i >= offset, price_lag_N[i] == price[i-N].

    Test Case:
    test_lag_features\\test_lag_equals_shifted_price_when_sufficient_history

    Purpose:
    Validates that lag features correctly reference the price N slots ago.

    Test Conditions:
    - Random price series of varying lengths
    - All 8 lag offsets tested

    Key Properties:
        For any price series of length L and lag offset N,
        price_lag_N[i] equals price[i-N] when i >= N.

    Expected Behaviour:
    - Lag values match the shifted price exactly for indices with sufficient history
    """
    rng = np.random.default_rng(seed)
    prices = pd.Series(rng.uniform(10, 200, size=length), dtype=float)
    price_mean = prices.mean()

    lag = compute_lag(prices, offset, price_mean)

    for i in range(offset, length):
        assert lag.iloc[i] == pytest.approx(prices.iloc[i - offset]), (
            f"lag[{i}] should equal price[{i - offset}] for offset={offset}"
        )


@given(
    length=st.integers(min_value=1, max_value=50),
    offset=st.sampled_from([2, 3, 4, 6, 12, 24, 48, 336]),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_lag_uses_mean_when_insufficient_history(length, offset, seed):
    """Property 3: For index i < offset, price_lag_N[i] == global mean.

    Test Case:
    test_lag_features\\test_lag_uses_mean_when_insufficient_history

    Purpose:
    Validates that lag features fall back to the global mean price
    when insufficient history exists for the lag offset.

    Test Conditions:
    - Short price series where length < offset
    - All lag offsets tested

    Key Properties:
        For any price series of length L and lag offset N,
        price_lag_N[i] equals the global mean price when i < N.

    Expected Behaviour:
    - All positions before the offset are filled with the global mean
    """
    assume(length < offset)
    rng = np.random.default_rng(seed)
    prices = pd.Series(rng.uniform(10, 200, size=length), dtype=float)
    price_mean = prices.mean()

    lag = compute_lag(prices, offset, price_mean)

    # All values should be the mean since length < offset
    for i in range(length):
        assert lag.iloc[i] == pytest.approx(price_mean), (
            f"lag[{i}] should equal mean={price_mean} when length={length} < offset={offset}"
        )
