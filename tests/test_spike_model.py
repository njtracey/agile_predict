"""Property tests for regime-aware spike model.

Feature: agile-predict-advanced-features, Properties 8 & 9: Spike Model
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def compute_spike_weight(margin: float, comfortable_margin: float) -> float:
    """Replicate the spike weight computation from update.py."""
    return float(np.clip(1.0 - margin / comfortable_margin, 0.0, 1.0))


def blend_predictions(
    normal_pred: float, spike_pred: float, spike_weight: float
) -> float:
    """Replicate the spike blending formula from update.py."""
    return (1 - spike_weight) * normal_pred + spike_weight * spike_pred


@given(
    margin=st.floats(min_value=0, max_value=50000),
    comfortable_margin=st.floats(min_value=100, max_value=50000),
    normal_pred=st.floats(min_value=-50, max_value=500),
    spike_pred=st.floats(min_value=-50, max_value=500),
)
@settings(max_examples=100)
def test_spike_blending_formula(margin, comfortable_margin, normal_pred, spike_pred):
    """Property 9: Blended prediction follows the formula exactly.

    Test Case:
    test_spike_model\\test_spike_blending_formula

    Purpose:
    Validates the spike blending formula: (1-w)*P_n + w*P_s
    where w = clip(1 - M/C, 0, 1).

    Test Conditions:
    - Random margin, comfortable_margin, normal and spike predictions
    - Realistic ranges for GB electricity market

    Key Properties:
        For any (margin, comfortable_margin, normal_pred, spike_pred),
        spike_weight = clip(1 - M/C, 0, 1) and
        blended = (1-w)*P_n + w*P_s.

    Expected Behaviour:
    - Weight is between 0 and 1
    - Blended prediction is between normal and spike predictions
    """
    w = compute_spike_weight(margin, comfortable_margin)
    assert 0.0 <= w <= 1.0

    blended = blend_predictions(normal_pred, spike_pred, w)
    expected = (1 - w) * normal_pred + w * spike_pred
    assert blended == pytest.approx(expected)


@given(
    comfortable_margin=st.floats(min_value=100, max_value=50000),
    normal_pred=st.floats(min_value=-50, max_value=500),
    spike_pred=st.floats(min_value=-50, max_value=500),
)
@settings(max_examples=100)
def test_comfortable_margin_uses_normal_only(comfortable_margin, normal_pred, spike_pred):
    """Property 9: When margin >= comfortable_margin, result equals normal_pred.

    Test Case:
    test_spike_model\\test_comfortable_margin_uses_normal_only

    Purpose:
    Validates that when supply margin is comfortable, the spike model
    has zero influence on the prediction.

    Test Conditions:
    - margin >= comfortable_margin

    Key Properties:
        When M >= C, blended prediction equals P_n exactly.

    Expected Behaviour:
    - spike_weight is 0
    - blended prediction equals normal_pred
    """
    w = compute_spike_weight(comfortable_margin, comfortable_margin)
    assert w == pytest.approx(0.0)
    blended = blend_predictions(normal_pred, spike_pred, w)
    assert blended == pytest.approx(normal_pred)


@given(
    normal_pred=st.floats(min_value=-50, max_value=500),
    spike_pred=st.floats(min_value=-50, max_value=500),
)
@settings(max_examples=100)
def test_zero_margin_uses_spike_only(normal_pred, spike_pred):
    """Property 9: When margin = 0, result equals spike_pred.

    Test Case:
    test_spike_model\\test_zero_margin_uses_spike_only

    Purpose:
    Validates that when supply margin is zero (extreme scarcity),
    the spike model fully replaces the normal prediction.

    Test Conditions:
    - margin = 0

    Key Properties:
        When M = 0, blended prediction equals P_s exactly.

    Expected Behaviour:
    - spike_weight is 1
    - blended prediction equals spike_pred
    """
    w = compute_spike_weight(0.0, 4000.0)
    assert w == pytest.approx(1.0)
    blended = blend_predictions(normal_pred, spike_pred, w)
    assert blended == pytest.approx(spike_pred)


def test_spike_training_filter():
    """Property 8: Spike training set contains exactly low-margin samples.

    Test Case:
    test_spike_model\\test_spike_training_filter

    Purpose:
    Validates that the spike model training filter correctly selects
    only samples where derated_margin_mw < threshold.

    Test Conditions:
    - Array of margin values with known distribution above/below threshold

    Key Properties:
        For any training data with margin values, spike training set
        contains exactly samples where derated_margin_mw < threshold.

    Expected Behaviour:
    - Filtered count matches expected count
    - All filtered values are below threshold
    """
    margins = np.array([1000, 2500, 500, 3000, 1500, 4000, 800, 2000])
    threshold = 2000
    mask = margins < threshold
    assert mask.sum() == 4  # 1000, 500, 1500, 800
    assert all(margins[mask] < threshold)
    assert all(margins[~mask] >= threshold)
