"""
Unit tests for Customer Lifetime Value (CLV) calculation.
"""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, clean_and_preprocess_data
from src.rfm_analysis import compute_rfm_metrics
from src.clv_calculator import calculate_customer_lifetime_value


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
