"""
RFM (Recency, Frequency, Monetary) Customer Segmentation Module.
Computes RFM scores, segments customers using rule-based heuristics and K-Means clustering,
and outputs actionable marketing segments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def compute_rfm_metrics(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary metrics for each unique customer.
    Filters out returned transactions for monetary calculations.
    """
    if "LineTotal" not in df.columns:
        df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    valid_orders = df[df["Returned"] == 0] if "Returned" in df.columns else df

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = valid_orders.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("LineTotal", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Country=("Country", "first"),
        Age=("Age", "mean") if "Age" in valid_orders.columns else ("CustomerID", "count"),
        Satisfaction=("Satisfaction", "mean") if "Satisfaction" in valid_orders.columns else ("CustomerID", "count"),
    ).reset_index()

    # Calculate Customer Tenure in days
    rfm["TenureDays"] = (reference_date - rfm["FirstPurchase"]).dt.days
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["Age"] = rfm["Age"].round(1) if "Age" in rfm.columns else 35.0
    rfm["Satisfaction"] = rfm["Satisfaction"].round(2) if "Satisfaction" in rfm.columns else 4.0

    return rfm


def score_rfm_quintiles(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns quintile scores (1 to 5) to Recency, Frequency, and Monetary values.
    For Recency: 5 is best (most recent), 1 is worst.
    For Frequency and Monetary: 5 is highest, 1 is lowest.
    """
    df = rfm_df.copy()

    # Recency quintiles (inverted: lower days = higher score)
    df["R_Score"] = pd.qcut(df["Recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency quintiles
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Monetary quintiles
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Composite"] = (df["R_Score"] + df["F_Score"] + df["M_Score"]) / 3.0

    return df


def map_rfm_segments(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps customer quintiles to standard marketing segments:
    - Champions: Bought recently, buy often, and spend the most.
    - Loyal Customers: Buy on a regular basis, responsive to promotions.
    - Potential Loyalists: Recent customers with average frequency.
    - New Customers: High recency but low frequency.
    - Promising: Recent shoppers, but haven't spent much.
    - Customers Needing Attention: Above average recency, frequency, and monetary.
    - At Risk: Spent big money and purchased often, but long time ago.
    - Cant Lose Them: Made big purchases, and often, but haven't returned for a long time.
    - Hibernating: Low frequency, low spend, last purchase long ago.
    - Lost / Churned: Lowest recency, frequency, and monetary scores.
    """
    df = scored_df.copy()

    def get_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]
        rf = (r + f) / 2.0

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f in [1, 2]:
            return "New Customers" if f == 1 else "Potential Loyalists"
        elif r == 3 and f in [1, 2]:
            return "Promising"
        elif r in [2, 3] and f >= 3:
            return "Customers Needing Attention"
        elif r in [1, 2] and f >= 4 and m >= 4:
            return "Can't Lose Them"
        elif r in [1, 2] and f >= 2:
            return "At Risk"
        elif r == 2 and f == 1:
            return "Hibernating"
        else:
            return "Lost"

    df["Segment"] = df.apply(get_segment, axis=1)
    return df


def apply_kmeans_clustering(rfm_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans]:
    """
    Applies K-Means clustering on log-transformed, normalized RFM metrics.
    Compares unsupervised clusters against rule-based segments.
    """
    df = rfm_df.copy()

    # Log transform to handle right-skewed distributions
    rfm_log = pd.DataFrame()
    rfm_log["Recency"] = np.log1p(df["Recency"])
    rfm_log["Frequency"] = np.log1p(df["Frequency"])
    rfm_log["Monetary"] = np.log1p(df["Monetary"].clip(lower=0))

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10)
    df["Cluster"] = kmeans.fit_predict(rfm_scaled)
    df["Cluster"] = df["Cluster"].astype(str)

    return df, kmeans


def generate_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produces executive summary metrics grouped by customer segment."""
    summary = df.groupby("Segment").agg(
        CustomerCount=("CustomerID", "count"),
        AvgRecencyDays=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetarySpend=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
        AvgSatisfaction=("Satisfaction", "mean"),
    ).reset_index()

    total_customers = len(df)
    total_rev = summary["TotalRevenue"].sum()

    summary["CustomerPct"] = (summary["CustomerCount"] / total_customers * 100).round(1)
    summary["RevenuePct"] = (summary["TotalRevenue"] / total_rev * 100).round(1)
    summary["AvgRecencyDays"] = summary["AvgRecencyDays"].round(1)
    summary["AvgFrequency"] = summary["AvgFrequency"].round(1)
    summary["AvgMonetarySpend"] = summary["AvgMonetarySpend"].round(2)
    summary["TotalRevenue"] = summary["TotalRevenue"].round(2)
    summary["AvgSatisfaction"] = summary["AvgSatisfaction"].round(2)

    return summary.sort_values(by="TotalRevenue", ascending=False)


def run_rfm_pipeline(transactions_path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    """Executes the complete RFM analysis pipeline and saves results."""
    logger.info("Loading transaction data from: %s", transactions_path)
    if str(transactions_path).endswith(".parquet"):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    rfm = compute_rfm_metrics(df)
    scored = score_rfm_quintiles(rfm)
    segmented = map_rfm_segments(scored)
    clustered, _ = apply_kmeans_clustering(segmented)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "customer_rfm_segments.csv"
        clustered.to_csv(out_path, index=False)
        logger.info("Saved RFM segmentation output to: %s", out_path)

    return clustered


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_transactions.parquet"
    if not data_path.exists():
        data_path = project_root / "data" / "raw" / "sample_ecommerce_data.csv"

    out_dir = project_root / "data" / "processed"
    run_rfm_pipeline(data_path, out_dir)


if __name__ == "__main__":
    main()
