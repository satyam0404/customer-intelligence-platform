"""
Unit tests for Market Basket Analysis and Association Rules.
"""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, clean_and_preprocess_data
from src.basket_analysis import (
    extract_transaction_baskets,
    generate_association_rules,
    get_recommendations_for_product,
)


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
