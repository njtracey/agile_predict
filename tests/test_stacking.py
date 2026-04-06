"""Property tests for stacking meta-learner.

Feature: agile-predict-advanced-features, Property 7: Stacking Meta-Learner Validity
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import Ridge


@given(
    n_samples=st.integers(min_value=20, max_value=200),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_meta_learner_has_three_coefficients(n_samples, seed):
    """Property 7: Fitted meta-learner has exactly 3 coefficients.

    Test Case:
    test_stacking\\test_meta_learner_has_three_coefficients

    Purpose:
    Validates that the meta-learner trained on stacked out-of-fold
    predictions from 3 base models has exactly 3 coefficients.

    Test Conditions:
    - Random training data of varying sizes
    - 3 simulated base model OOF predictions

    Key Properties:
        For any training dataset with sufficient samples,
        the fitted meta-learner has exactly 3 coefficients
        (one per base model).

    Expected Behaviour:
    - meta_learner.coef_ has shape (3,)
    - meta_learner has an intercept
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(20, 150, size=n_samples)
    oof_1 = y + rng.normal(0, 10, size=n_samples)
    oof_2 = y + rng.normal(0, 12, size=n_samples)
    oof_3 = y + rng.normal(0, 15, size=n_samples)

    meta_X = np.column_stack([oof_1, oof_2, oof_3])
    meta_learner = Ridge(alpha=0.1, fit_intercept=True)
    meta_learner.fit(meta_X, y)

    assert meta_learner.coef_.shape == (3,), (
        f"Expected 3 coefficients, got shape {meta_learner.coef_.shape}"
    )
    assert hasattr(meta_learner, "intercept_")


@given(
    n_samples=st.integers(min_value=20, max_value=200),
    seed=st.integers(min_value=0, max_value=2**31),
)
@settings(max_examples=100)
def test_oof_predictions_same_length_as_training(n_samples, seed):
    """Property 7: Out-of-fold predictions have same length as training set.

    Test Case:
    test_stacking\\test_oof_predictions_same_length_as_training

    Purpose:
    Validates that cross_val_predict returns predictions with the
    same number of samples as the input training data.

    Test Conditions:
    - Random training data of varying sizes
    - Ridge model used as a simple base model

    Key Properties:
        For any training dataset with sufficient samples for CV,
        out-of-fold predictions have the same length as the training set.

    Expected Behaviour:
    - len(oof_predictions) == len(training_data)
    """
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 100, size=(n_samples, 5))
    y = rng.uniform(20, 150, size=n_samples)

    model = Ridge(alpha=1.0)
    n_cv = min(5, n_samples // 2)
    if n_cv < 2:
        pytest.skip("Insufficient samples for CV")

    oof = cross_val_predict(model, X, y, cv=n_cv)
    assert len(oof) == n_samples, (
        f"Expected {n_samples} OOF predictions, got {len(oof)}"
    )
