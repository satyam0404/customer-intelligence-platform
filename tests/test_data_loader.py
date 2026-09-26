"""
Unit tests for data generation and cleaning routines.
"""

from datetime import datetime
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, clean_and_preprocess_data


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
