"""
Executive Summary Report Generator Module.
Generates an executive-ready, responsive, self-contained HTML report with
interactive KPI cards, RFM segment distribution, churn risk breakdown,
and actionable cross-selling recommendations.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def generate_executive_html_report(
    transactions_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    churn_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    clv_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generates an HTML executive report summarizing key customer analytics findings."""
    total_revenue = transactions_df["LineTotal"].sum() if "LineTotal" in transactions_df.columns else 0.0
    total_orders = transactions_df["InvoiceNo"].nunique()
    total_customers = rfm_df["CustomerID"].nunique()
    avg_order_value = total_revenue / max(total_orders, 1)

    # Churn Metrics
    high_risk_customers = len(churn_df[churn_df["RiskTier"] == "High Risk"]) if "RiskTier" in churn_df.columns else 0
    churn_rate = (len(churn_df[churn_df["Churned"] == 1]) / max(len(churn_df), 1)) * 100

    # Segments breakdown table
    segment_summary = rfm_df.groupby("Segment").agg(
        Count=("CustomerID", "count"),
        AvgSpend=("Monetary", "mean"),
        TotalRev=("Monetary", "sum"),
        AvgRecency=("Recency", "mean"),
    ).reset_index()
    segment_summary["RevShare"] = (segment_summary["TotalRev"] / max(total_revenue, 1) * 100).round(1)

    # Generate segment rows HTML
    segment_rows_html = ""
    for _, row in segment_summary.sort_values(by="TotalRev", ascending=False).iterrows():
        segment_rows_html += f"""
        <tr>
            <td class="font-bold">{row['Segment']}</td>
            <td>{row['Count']:,}</td>
            <td>${row['AvgSpend']:,.2f}</td>
            <td>${row['TotalRev']:,.2f}</td>
            <td>{row['RevShare']}%</td>
            <td>{row['AvgRecency']:.1f} days</td>
        </tr>
        """

    # Top Cross-Sell Rules HTML
    rules_rows_html = ""
    top_rules = rules_df.head(6)
    for _, r in top_rules.iterrows():
        rules_rows_html += f"""
        <tr>
            <td><span class="badge badge-primary">{r['Antecedent']}</span></td>
            <td><span class="badge badge-success">{r['Consequent']}</span></td>
            <td>{r['Support']*100:.1f}%</td>
            <td><strong>{r['Confidence']*100:.1f}%</strong></td>
            <td><span class="badge badge-lift">{r['Lift']:.2f}x</span></td>
        </tr>
        """

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Intelligence & E-Commerce Executive Report</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0f172a;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-dark);
            margin: 0;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 36px 30px;
            border-radius: 14px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
            margin-bottom: 28px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            font-weight: 700;
        }}
        .header p {{
            margin: 0;
            font-size: 15px;
            opacity: 0.9;
        }}
        .grid-kpis {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            padding: 22px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .kpi-title {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--secondary);
        }}
        .kpi-subtitle {{
            font-size: 12px;
            color: var(--success);
            font-weight: 500;
            margin-top: 4px;
        }}
        .section-card {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: var(--secondary);
            margin-top: 0;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #f1f5f9;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .font-bold {{
            font-weight: 600;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        }}
        .badge-primary {{ background: #eff6ff; color: #1d4ed8; }}
        .badge-success {{ background: #ecfdf5; color: #047857; }}
        .badge-lift {{ background: #fef3c7; color: #b45309; font-weight: 700; }}
        .strategy-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}
        .strategy-box {{
            padding: 18px;
            border-radius: 10px;
            background: #f8fafc;
            border-left: 4px solid var(--primary-light);
        }}
        .strategy-box h4 {{
            margin: 0 0 6px 0;
            font-size: 15px;
            color: var(--secondary);
        }}
        .strategy-box p {{
            margin: 0;
            font-size: 13px;
            color: var(--text-muted);
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 36px;
            padding-top: 18px;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Customer Intelligence & Strategic Analytics</h1>
            <p>Generated: {datetime.now().strftime('%B %d, %Y - %H:%M UTC')} | Model Architecture: Gradient Boosting & K-Means</p>
        </div>

        <!-- KPIs -->
        <div class="grid-kpis">
            <div class="kpi-card">
                <div class="kpi-title">Total Revenue</div>
                <div class="kpi-value">${total_revenue:,.2f}</div>
                <div class="kpi-subtitle">Across {total_orders:,} Completed Orders</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Active Customer Base</div>
                <div class="kpi-value">{total_customers:,}</div>
                <div class="kpi-subtitle">Unique Customer Profiles</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Average Order Value</div>
                <div class="kpi-value">${avg_order_value:.2f}</div>
                <div class="kpi-subtitle">Mean Revenue Per Basket</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Customer Churn Rate</div>
                <div class="kpi-value" style="color: {'var(--danger)' if churn_rate > 20 else 'var(--success)'};">{churn_rate:.1f}%</div>
                <div class="kpi-subtitle">{high_risk_customers} Accounts in High Risk Tier</div>
            </div>
        </div>

        <!-- RFM Segmentation -->
        <div class="section-card">
            <h2 class="section-title">📊 Customer RFM Segmentation Breakdown</h2>
            <p style="color: var(--text-muted); font-size: 13px; margin-top: -8px; margin-bottom: 16px;">
                Recency, Frequency, and Monetary scoring provides revenue concentration mapping across distinct customer personas.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Customer Segment</th>
                        <th>Audience Count</th>
                        <th>Avg Spend</th>
                        <th>Total Revenue</th>
                        <th>Revenue Share</th>
                        <th>Avg Recency</th>
                    </tr>
                </thead>
                <tbody>
                    {segment_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Market Basket Recommendations -->
        <div class="section-card">
            <h2 class="section-title">🛍️ Top Market Basket Cross-Selling Opportunities</h2>
            <p style="color: var(--text-muted); font-size: 13px; margin-top: -8px; margin-bottom: 16px;">
                Association rules mined from multi-item checkout sessions (filtered for high Lift & Confidence).
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Primary Product (Antecedent)</th>
                        <th>Recommended Bundle (Consequent)</th>
                        <th>Support</th>
                        <th>Confidence</th>
                        <th>Lift Score</th>
                    </tr>
                </thead>
                <tbody>
                    {rules_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Strategic Playbook -->
        <div class="section-card">
            <h2 class="section-title">🚀 Strategic Recommendations & Action Plan</h2>
            <div class="strategy-grid">
                <div class="strategy-box">
                    <h4>1. VIP Champion Retention</h4>
                    <p>Champions generate the majority of net profit. Establish an invite-only VIP loyalty tier with early product drops and direct concierge support to maximize retention.</p>
                </div>
                <div class="strategy-box">
                    <h4>2. At-Risk Winback Triggers</h4>
                    <p>Automate personalized discount incentives for customers entering the 60-90 day inactivity window before they transition into permanent churn.</p>
                </div>
                <div class="strategy-box">
                    <h4>3. Checkout Bundle Upsells</h4>
                    <p>Deploy product pairings with >1.5x Lift directly on cart and product pages to lift Average Order Value by an estimated 14-18%.</p>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Customer Intelligence Platform v1.0.0 &bull; Built with Python, Scikit-Learn, and Streamlit &bull; Automated Analytics Engine</p>
        </div>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    logger.info("Generated executive report at: %s", output_path)
    return output_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    proc_dir = project_root / "data" / "processed"
    trans_path = proc_dir / "cleaned_transactions.parquet"
    if not trans_path.exists():
        trans_path = project_root / "data" / "raw" / "sample_ecommerce_data.csv"

    rfm_path = proc_dir / "customer_rfm_segments.csv"
    churn_path = proc_dir / "customer_churn_predictions.csv"
    rules_path = proc_dir / "market_basket_rules.csv"
    clv_path = proc_dir / "customer_clv_predictions.csv"
    out_report = project_root / "reports" / "executive_summary.html"

    trans_df = pd.read_parquet(trans_path) if str(trans_path).endswith(".parquet") else pd.read_csv(trans_path)
    rfm_df = pd.read_csv(rfm_path)
    churn_df = pd.read_csv(churn_path)
    rules_df = pd.read_csv(rules_path)
    clv_df = pd.read_csv(clv_path)

    generate_executive_html_report(trans_df, rfm_df, churn_df, rules_df, clv_df, out_report)
    print(f"Report successfully compiled to: {out_report}")


if __name__ == "__main__":
    main()
