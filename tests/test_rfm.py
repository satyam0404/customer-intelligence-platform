"""
Unit tests for RFM calculation and segmentation logic.
"""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data, clean_and_preprocess_data
from src.rfm_analysis import (
    compute_rfm_metrics,
    score_rfm_quintiles,
    map_rfm_segments,
    apply_kmeans_clustering,
    generate_segment_summary,
)


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
