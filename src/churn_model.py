"""
Customer Churn Prediction and Risk Modeling Module.
Performs behavioral feature engineering, trains classification algorithms (Random Forest,
Gradient Boosting, Logistic Regression), calculates ROC-AUC/Confusion Matrix,
and outputs customer-level churn risk probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def engineer_churn_features(
    transactions_df: pd.DataFrame,
    churn_inactivity_days: int = 120,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Extracts rich customer-level behavioral and demographic signals for churn prediction.
    Target variable: Churned = 1 if Recency > churn_inactivity_days, else 0.
    """
    df = transactions_df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if "LineTotal" not in df.columns:
        df["LineTotal"] = df["Quantity"] * df["UnitPrice"]

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Calculate customer aggregates
    grouped = df.groupby("CustomerID")

    # Purchase dates for calculating inter-purchase times
    def calc_interval_std(dates):
        if len(dates) < 2:
            return 0.0
        sorted_dates = sorted(dates)
        diffs = [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]
        return float(np.std(diffs))

    intervals = grouped["InvoiceDate"].apply(calc_interval_std).rename("PurchaseIntervalStd")

    # Return rate
    if "Returned" in df.columns:
        return_rate = grouped["Returned"].mean().rename("ReturnRate")
    else:
        return_rate = pd.Series(0.0, index=df["CustomerID"].unique(), name="ReturnRate")

    # Primary customer stats
    customer_stats = grouped.agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        TotalSpend=("LineTotal", "sum"),
        TotalQuantity=("Quantity", "sum"),
        TenureDays=("InvoiceDate", lambda x: (reference_date - x.min()).days),
        Age=("Age", "mean") if "Age" in df.columns else ("LineTotal", "count"),
        Satisfaction=("Satisfaction", "mean") if "Satisfaction" in df.columns else ("LineTotal", "count"),
        Device=("Device", lambda x: x.mode()[0]) if "Device" in df.columns else ("LineTotal", "count"),
        Country=("Country", "first"),
    )

    features = customer_stats.join(intervals).join(return_rate).reset_index()

    # Derived KPIs
    features["AverageOrderValue"] = (features["TotalSpend"] / features["Frequency"].clip(lower=1)).round(2)
    features["AverageBasketQuantity"] = (features["TotalQuantity"] / features["Frequency"].clip(lower=1)).round(2)

    # Churn Definition: Inactive beyond threshold
    features["Churned"] = (features["Recency"] > churn_inactivity_days).astype(int)

    return features


def train_and_evaluate_models(
    features_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains multiple ML models and benchmarks their classification performance.
    Returns comparison dictionary and customer predictions dataframe with churn risk tiers.
    """
    feature_cols = [
        "Frequency",
        "TotalSpend",
        "AverageOrderValue",
        "AverageBasketQuantity",
        "TenureDays",
        "PurchaseIntervalStd",
        "ReturnRate",
        "Age",
        "Satisfaction",
    ]

    # Ensure all feature columns exist and are numeric
    available_cols = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available_cols].fillna(0)
    y = features_df["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
    }

    results = {}
    best_model_name = "Gradient Boosting"
    best_model = None
    best_score = -1.0

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
        }

        if auc > best_score:
            best_score = auc
            best_model_name = name
            best_model = model

    # Feature importances for best tree model
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(available_cols, [round(float(v), 4) for v in best_model.feature_importances_]))
        sorted_importance = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    else:
        sorted_importance = {}

    results["best_model"] = best_model_name
    results["feature_importances"] = sorted_importance

    # Generate predictions on full dataset
    pred_df = features_df.copy()
    if best_model_name == "Logistic Regression":
        all_scaled = scaler.transform(X)
        pred_df["ChurnProbability"] = best_model.predict_proba(all_scaled)[:, 1].round(4)
    else:
        pred_df["ChurnProbability"] = best_model.predict_proba(X)[:, 1].round(4)

    # Assign risk tiers
    def assign_risk_tier(prob):
        if prob < 0.35:
            return "Low Risk"
        elif prob < 0.70:
            return "Medium Risk"
        else:
            return "High Risk"

    pred_df["RiskTier"] = pred_df["ChurnProbability"].apply(assign_risk_tier)

    return results, pred_df


def run_churn_pipeline(transactions_path: Path, output_dir: Path | None = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs end-to-end churn engineering, training, and saves predictions."""
    logger.info("Executing churn pipeline on %s...", transactions_path)
    if str(transactions_path).endswith(".parquet"):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)

    features = engineer_churn_features(df)
    results, predictions = train_and_evaluate_models(features)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / "customer_churn_predictions.csv"
        predictions.to_csv(pred_path, index=False)
        logger.info("Saved churn predictions to: %s", pred_path)

    return results, predictions


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_transactions.parquet"
    if not data_path.exists():
        data_path = project_root / "data" / "raw" / "sample_ecommerce_data.csv"

    out_dir = project_root / "data" / "processed"
    results, preds = run_churn_pipeline(data_path, out_dir)
    print("\n--- Churn Model Benchmark Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
