"""
Unit tests for Churn feature engineering and predictive modeling.
"""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, clean_and_preprocess_data
from src.churn_model import engineer_churn_features, train_and_evaluate_models


@pytest.fixture
def sample_clean_data():
    raw = generate_synthetic_data(num_customers=60, seed=101)
    return clean_and_preprocess_data(raw)


def test_engineer_churn_features(sample_clean_data):
    features = engineer_churn_features(sample_clean_data, churn_inactivity_days=100)
    assert not features.empty
    assert "Churned" in features.columns
    assert "AverageOrderValue" in features.columns
    assert "TenureDays" in features.columns
    assert set(features["Churned"].unique()).issubset({0, 1})


def test_train_and_evaluate_models(sample_clean_data):
    features = engineer_churn_features(sample_clean_data, churn_inactivity_days=100)
    results, preds = train_and_evaluate_models(features, random_state=42)

    assert "best_model" in results
    assert "feature_importances" in results
    assert "ChurnProbability" in preds.columns
    assert "RiskTier" in preds.columns
    assert preds["ChurnProbability"].between(0.0, 1.0).all()
    assert set(preds["RiskTier"].unique()).issubset({"Low Risk", "Medium Risk", "High Risk"})
