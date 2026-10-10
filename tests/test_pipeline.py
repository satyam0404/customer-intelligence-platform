
from src.pipeline import (
    generate_synthetic_data, clean_and_preprocess_data,
    compute_rfm_metrics, score_rfm_quintiles, apply_kmeans_clustering, map_rfm_segments,
    generate_segment_summary,
    engineer_churn_features, train_and_evaluate_models, extract_transaction_baskets,
    get_recommendations_for_product,
    generate_association_rules, calculate_customer_lifetime_value
)
"""
Unit tests for data generation and cleaning routines.
"""

from datetime import datetime
import pandas as pd
import pytest



def test_generate_synthetic_data():
    df = generate_synthetic_data(num_customers=50, num_transactions=200, seed=42)
    assert not df.empty
    assert "CustomerID" in df.columns
    assert "InvoiceNo" in df.columns
    assert "Quantity" in df.columns
    assert "UnitPrice" in df.columns
    assert "InvoiceDate" in df.columns
    assert df["CustomerID"].nunique() <= 50


def test_clean_and_preprocess_data():
    raw_df = generate_synthetic_data(num_customers=20, seed=123)
    cleaned = clean_and_preprocess_data(raw_df)

    assert "LineTotal" in cleaned.columns
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()
    assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])
    assert (cleaned["LineTotal"] == cleaned["Quantity"] * cleaned["UnitPrice"]).all()
"""
Unit tests for RFM calculation and segmentation logic.
"""

import pandas as pd
import pytest



@pytest.fixture
def sample_clean_data():
    raw = generate_synthetic_data(num_customers=40, seed=99)
    return clean_and_preprocess_data(raw)


def test_compute_rfm_metrics(sample_clean_data):
    rfm = compute_rfm_metrics(sample_clean_data)
    assert not rfm.empty
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()


def test_score_and_map_segments(sample_clean_data):
    rfm = compute_rfm_metrics(sample_clean_data)
    scored = score_rfm_quintiles(rfm)
    assert "R_Score" in scored.columns
    assert "F_Score" in scored.columns
    assert "M_Score" in scored.columns
    assert scored["R_Score"].between(1, 5).all()

    segmented = map_rfm_segments(scored)
    assert "Segment" in segmented.columns
    assert segmented["Segment"].nunique() > 0

    clustered, kmeans = apply_kmeans_clustering(segmented, n_clusters=3)
    assert "Cluster" in clustered.columns
    assert kmeans is not None

    summary = generate_segment_summary(clustered)
    assert not summary.empty
    assert "CustomerCount" in summary.columns
    assert "TotalRevenue" in summary.columns
"""
Unit tests for Churn feature engineering and predictive modeling.
"""

import pandas as pd
import pytest



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
"""
Unit tests for Market Basket Analysis and Association Rules.
"""

import pandas as pd
import pytest



@pytest.fixture
def sample_clean_data():
    raw = generate_synthetic_data(num_customers=50, seed=77)
    return clean_and_preprocess_data(raw)


def test_basket_analysis_pipeline(sample_clean_data):
    baskets = extract_transaction_baskets(sample_clean_data)
    assert isinstance(baskets, list)

    rules = generate_association_rules(baskets, min_support=0.005, min_confidence=0.1, min_lift=1.0)
    assert isinstance(rules, pd.DataFrame)
    if not rules.empty:
        assert "Antecedent" in rules.columns
        assert "Consequent" in rules.columns
        assert "Lift" in rules.columns
        assert (rules["Lift"] >= 1.0).all()

        prod = rules["Antecedent"].iloc[0]
        recs = get_recommendations_for_product(prod, rules, top_n=2)
        assert len(recs) <= 2
"""
Unit tests for Customer Lifetime Value (CLV) calculation.
"""

import pandas as pd
import pytest



@pytest.fixture
def sample_clean_data():
    raw = generate_synthetic_data(num_customers=30, seed=55)
    return clean_and_preprocess_data(raw)


def test_calculate_clv(sample_clean_data):
    rfm = compute_rfm_metrics(sample_clean_data)
    clv_df = calculate_customer_lifetime_value(rfm, profit_margin=0.25, discount_rate=0.10)

    assert not clv_df.empty
    assert "HistoricalProfit" in clv_df.columns
    assert "PredictiveCLV_2Year" in clv_df.columns
    assert "TotalEstimatedCLV" in clv_df.columns
    assert "CLV_Tier" in clv_df.columns
    assert (clv_df["TotalEstimatedCLV"] >= clv_df["HistoricalProfit"]).all()
    assert set(clv_df["CLV_Tier"].unique()).issubset({"Platinum", "Gold", "Silver", "Bronze"})
