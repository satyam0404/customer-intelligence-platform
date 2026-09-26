"""
Customer Lifetime Value (CLV) Calculation Module.
Computes historical revenue contributions and forecasts multi-period predictive CLV
factoring in retention probability, profit margins, and discount rates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_customer_lifetime_value(
    rfm_df: pd.DataFrame,
    churn_df: pd.DataFrame | None = None,
    profit_margin: float = 0.25,
    discount_rate: float = 0.10,
    forecast_years: int = 2,
) -> pd.DataFrame:
    """
    Computes both Historical and Predictive CLV for each customer.
    Formula for period t:
      ExpectedValue_t = (AOV * AnnualFrequency * ProfitMargin) * (RetentionRate^t) / ((1 + DiscountRate)^t)
    """
    df = rfm_df.copy()

    # Calculate Annual Purchase Frequency
    tenure_years = (df["TenureDays"] / 365.0).clip(lower=0.1)
    df["AnnualFrequency"] = (df["Frequency"] / tenure_years).round(2)
    df["AverageOrderValue"] = (df["Monetary"] / df["Frequency"].clip(lower=1)).round(2)

    # Attach churn probability if available
    if churn_df is not None and "ChurnProbability" in churn_df.columns:
        churn_map = churn_df.set_index("CustomerID")["ChurnProbability"].to_dict()
        df["ChurnProbability"] = df["CustomerID"].map(churn_map).fillna(0.30)
    else:
        # Heuristic retention based on Recency
        df["ChurnProbability"] = (df["Recency"] / 365.0).clip(0.05, 0.95)

    df["RetentionRate"] = 1.0 - df["ChurnProbability"]

    # Historical CLV (Historical Cumulative Profit)
    df["HistoricalProfit"] = (df["Monetary"] * profit_margin).round(2)

    # Calculate Multi-year Discounted Future Cash Flows
    predictive_clv = np.zeros(len(df))
    for t in range(1, forecast_years + 1):
        cash_flow = df["AverageOrderValue"] * df["AnnualFrequency"] * profit_margin
        survival_prob = df["RetentionRate"] ** t
        discount_factor = (1.0 + discount_rate) ** t
        predictive_clv += (cash_flow * survival_prob) / discount_factor

    df["PredictiveCLV_2Year"] = predictive_clv.round(2)
    df["TotalEstimatedCLV"] = (df["HistoricalProfit"] + df["PredictiveCLV_2Year"]).round(2)

    # Assign CLV Tier
    quantiles = df["TotalEstimatedCLV"].quantile([0.50, 0.80, 0.95]).to_dict()
    q50, q80, q95 = quantiles[0.50], quantiles[0.80], quantiles[0.95]

    def assign_tier(val):
        if val >= q95:
            return "Platinum"
        elif val >= q80:
            return "Gold"
        elif val >= q50:
            return "Silver"
        else:
            return "Bronze"

    df["CLV_Tier"] = df["TotalEstimatedCLV"].apply(assign_tier)
    return df


def run_clv_pipeline(
    rfm_path: Path,
    churn_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Runs CLV calculations and saves results."""
    logger.info("Calculating CLV from RFM path: %s", rfm_path)
    rfm_df = pd.read_csv(rfm_path)
    churn_df = pd.read_csv(churn_path) if churn_path and Path(churn_path).exists() else None

    clv_df = calculate_customer_lifetime_value(rfm_df, churn_df)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "customer_clv_predictions.csv"
        clv_df.to_csv(out_path, index=False)
        logger.info("Saved CLV analysis to: %s", out_path)

    return clv_df


def main():
    project_root = Path(__file__).resolve().parent.parent
    rfm_path = project_root / "data" / "processed" / "customer_rfm_segments.csv"
    churn_path = project_root / "data" / "processed" / "customer_churn_predictions.csv"
    out_dir = project_root / "data" / "processed"
    run_clv_pipeline(rfm_path, churn_path, out_dir)


if __name__ == "__main__":
    main()
