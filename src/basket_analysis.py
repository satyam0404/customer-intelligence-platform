"""
Market Basket Analysis (MBA) and Association Rule Mining Module.
Computes Support, Confidence, Lift, and Conviction across transaction baskets,
and generates cross-selling product bundle recommendations.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def extract_transaction_baskets(df: pd.DataFrame, item_col: str = "Description") -> List[List[str]]:
    """
    Groups line items by InvoiceNo to form distinct basket item lists.
    Filters out single-item orders to focus on co-purchased bundles.
    """
    valid_orders = df[df["Returned"] == 0] if "Returned" in df.columns else df
    baskets = valid_orders.groupby("InvoiceNo")[item_col].apply(lambda s: sorted(list(set(s)))).tolist()
    multi_item_baskets = [b for b in baskets if len(b) > 1]
    return multi_item_baskets


def generate_association_rules(
    baskets: List[List[str]],
    min_support: float = 0.01,
    min_confidence: float = 0.20,
    min_lift: float = 1.05,
) -> pd.DataFrame:
    """
    Extracts 2-item association rules: (Antecedent -> Consequent)
    Calculates:
      - Support: P(A and B) = count(A & B) / total_baskets
      - Confidence: P(B | A) = support(A & B) / support(A)
      - Lift: confidence(A -> B) / support(B)
      - Conviction: (1 - support(B)) / (1 - confidence(A -> B))
    """
    total_baskets = len(baskets)
    if total_baskets == 0:
        return pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Conviction"])

    # Frequency of individual items
    item_counts = Counter()
    pair_counts = Counter()

    for basket in baskets:
        unique_items = set(basket)
        for item in unique_items:
            item_counts[item] += 1
        for pair in combinations(sorted(unique_items), 2):
            pair_counts[pair] += 1

    rules = []

    for (item_a, item_b), pair_freq in pair_counts.items():
        support_ab = pair_freq / total_baskets
        if support_ab < min_support:
            continue

        support_a = item_counts[item_a] / total_baskets
        support_b = item_counts[item_b] / total_baskets

        # Rule 1: A -> B
        conf_a_b = support_ab / support_a if support_a > 0 else 0
        lift_a_b = conf_a_b / support_b if support_b > 0 else 0
        conv_a_b = (1 - support_b) / (1 - conf_a_b) if conf_a_b < 1.0 else float("inf")

        if conf_a_b >= min_confidence and lift_a_b >= min_lift:
            rules.append({
                "Antecedent": item_a,
                "Consequent": item_b,
                "Support": round(support_ab, 4),
                "Confidence": round(conf_a_b, 4),
                "Lift": round(lift_a_b, 3),
                "Conviction": round(conv_a_b, 3) if conv_a_b != float("inf") else 999.0,
                "PairCount": pair_freq,
            })

        # Rule 2: B -> A
        conf_b_a = support_ab / support_b if support_b > 0 else 0
        lift_b_a = conf_b_a / support_a if support_a > 0 else 0
        conv_b_a = (1 - support_a) / (1 - conf_b_a) if conf_b_a < 1.0 else float("inf")

        if conf_b_a >= min_confidence and lift_b_a >= min_lift:
            rules.append({
                "Antecedent": item_b,
                "Consequent": item_a,
                "Support": round(support_ab, 4),
                "Confidence": round(conf_b_a, 4),
                "Lift": round(lift_b_a, 3),
                "Conviction": round(conv_b_a, 3) if conv_b_a != float("inf") else 999.0,
                "PairCount": pair_freq,
            })

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=["Lift", "Confidence"], ascending=[False, False]).reset_index(drop=True)

    return rules_df


def get_recommendations_for_product(
    product_name: str,
    rules_df: pd.DataFrame,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Returns top complementary products to recommend given a primary product."""
    if rules_df.empty:
        return []

    matches = rules_df[rules_df["Antecedent"] == product_name].sort_values(by=["Lift", "Confidence"], ascending=False)
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        recommendations.append({
            "product": row["Consequent"],
            "confidence": f"{row['Confidence'] * 100:.1f}%",
            "lift": row["Lift"],
            "support": f"{row['Support'] * 100:.1f}%",
        })
    return recommendations


def run_basket_analysis_pipeline(transactions_path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    """Runs end-to-end basket analysis and saves association rules."""
    logger.info("Executing basket analysis on %s...", transactions_path)
    if str(transactions_path).endswith(".parquet"):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)

    baskets = extract_transaction_baskets(df, item_col="Description")
    rules_df = generate_association_rules(baskets, min_support=0.01, min_confidence=0.15, min_lift=1.02)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "market_basket_rules.csv"
        rules_df.to_csv(out_path, index=False)
        logger.info("Saved association rules to: %s (total %d rules)", out_path, len(rules_df))

    return rules_df


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_transactions.parquet"
    if not data_path.exists():
        data_path = project_root / "data" / "raw" / "sample_ecommerce_data.csv"

    out_dir = project_root / "data" / "processed"
    rules = run_basket_analysis_pipeline(data_path, out_dir)
    print("\n--- Top 10 Product Co-Purchase Association Rules ---")
    print(rules.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
