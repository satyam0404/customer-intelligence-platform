"""
Customer Intelligence Platform - Streamlit Interactive Web Dashboard.
Includes KPI overviews, interactive RFM segment exploration, machine learning
churn simulation, market basket cross-selling, and CLV analysis.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 8px 8px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 3px solid #2563eb;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    project_root = Path(__file__).resolve().parent.parent
    proc_dir = project_root / "data" / "processed"
    raw_path = project_root / "data" / "raw" / "sample_ecommerce_data.csv"
    trans_parquet = proc_dir / "cleaned_transactions.parquet"

    # Transactions
    if trans_parquet.exists():
        trans_df = pd.read_parquet(trans_parquet)
    elif raw_path.exists():
        trans_df = pd.read_csv(raw_path)
    else:
        # Generate on the fly if needed
        from src.pipeline import generate_synthetic_data, clean_and_preprocess_data
        raw = generate_synthetic_data()
        trans_df = clean_and_preprocess_data(raw)

    trans_df["InvoiceDate"] = pd.to_datetime(trans_df["InvoiceDate"])

    # RFM
    rfm_path = proc_dir / "customer_rfm_segments.csv"
    if rfm_path.exists():
        rfm_df = pd.read_csv(rfm_path)
    else:
        from src.pipeline import compute_rfm_metrics, score_rfm_quintiles, map_rfm_segments
        rfm_df = map_rfm_segments(score_rfm_quintiles(compute_rfm_metrics(trans_df)))

    # Churn
    churn_path = proc_dir / "customer_churn_predictions.csv"
    if churn_path.exists():
        churn_df = pd.read_csv(churn_path)
    else:
        from src.pipeline import engineer_churn_features, train_and_evaluate_models
        _, churn_df = train_and_evaluate_models(engineer_churn_features(trans_df))

    # Market Basket
    rules_path = proc_dir / "market_basket_rules.csv"
    if rules_path.exists():
        rules_df = pd.read_csv(rules_path)
    else:
        from src.pipeline import extract_transaction_baskets, generate_association_rules
        baskets = extract_transaction_baskets(trans_df)
        rules_df = generate_association_rules(baskets)

    # CLV
    clv_path = proc_dir / "customer_clv_predictions.csv"
    if clv_path.exists():
        clv_df = pd.read_csv(clv_path)
    else:
        from src.pipeline import calculate_customer_lifetime_value
        clv_df = calculate_customer_lifetime_value(rfm_df, churn_df)

    return trans_df, rfm_df, churn_df, rules_df, clv_df


trans_df, rfm_df, churn_df, rules_df, clv_df = load_all_data()

# Sidebar
st.sidebar.title("🛒 Customer Intelligence")
st.sidebar.caption("E-Commerce Analytics & ML Platform")

countries = ["All"] + sorted(trans_df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Filter by Country", countries)

devices = ["All"] + sorted(trans_df["Device"].unique().tolist()) if "Device" in trans_df.columns else ["All"]
selected_device = st.sidebar.selectbox("Filter by Device", devices)

# Apply filters
filtered_trans = trans_df.copy()
if selected_country != "All":
    filtered_trans = filtered_trans[filtered_trans["Country"] == selected_country]
if selected_device != "All" and "Device" in filtered_trans.columns:
    filtered_trans = filtered_trans[filtered_trans["Device"] == selected_device]

# Header
st.title("📊 Customer Intelligence & E-Commerce Analytics")
st.markdown("Advanced customer behavioral segmentation, machine learning churn prediction, and market basket analysis.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Executive Overview",
    "🎯 RFM Segmentation",
    "🔮 Churn Prediction (ML)",
    "🛍️ Market Basket Analysis",
    "💎 Customer Lifetime Value (CLV)",
])

# ----------------- TAB 1: EXECUTIVE OVERVIEW -----------------
with tab1:
    tot_rev = filtered_trans["LineTotal"].sum()
    tot_orders = filtered_trans["InvoiceNo"].nunique()
    tot_cust = filtered_trans["CustomerID"].nunique()
    aov = tot_rev / max(tot_orders, 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">${tot_rev:,.2f}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Orders</div><div class="metric-value">{tot_orders:,}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Active Customers</div><div class="metric-value">{tot_cust:,}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg Order Value (AOV)</div><div class="metric-value">${aov:.2f}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Revenue Trend
    monthly = filtered_trans.set_index("InvoiceDate").resample("W")["LineTotal"].sum().reset_index()
    monthly["7D_MA"] = monthly["LineTotal"].rolling(4, min_periods=1).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=monthly["InvoiceDate"], y=monthly["LineTotal"], name="Weekly Revenue", marker_color="#93c5fd", opacity=0.7))
    fig_trend.add_trace(go.Scatter(x=monthly["InvoiceDate"], y=monthly["7D_MA"], name="4-Week Moving Avg", line=dict(color="#1d4ed8", width=3)))
    fig_trend.update_layout(title="Weekly Revenue Growth & Moving Trend", height=380, template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cat_rev = filtered_trans.groupby("Category")["LineTotal"].sum().reset_index().sort_values(by="LineTotal", ascending=True)
        fig_cat = px.bar(cat_rev, x="LineTotal", y="Category", orientation="h", title="Revenue by Product Category", color="LineTotal", color_continuous_scale="Blues")
        fig_cat.update_layout(height=340, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        pay_df = filtered_trans.groupby("PaymentMethod")["LineTotal"].sum().reset_index()
        fig_pay = px.pie(pay_df, values="LineTotal", names="PaymentMethod", title="Revenue Share by Payment Method", hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pay.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig_pay, use_container_width=True)


# ----------------- TAB 2: RFM SEGMENTATION -----------------
with tab2:
    st.subheader("Customer Personas & Segment Distribution")

    seg_counts = rfm_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        fig_seg = px.treemap(
            seg_counts,
            path=["Segment"],
            values="Count",
            title="Customer Base Breakdown (Treemap)",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig_seg.update_layout(height=380, margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_s2:
        seg_rev = rfm_df.groupby("Segment")["Monetary"].sum().reset_index().sort_values(by="Monetary", ascending=False)
        fig_seg_rev = px.bar(seg_rev, x="Segment", y="Monetary", title="Total Revenue Generated by Segment ($)", color="Segment", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_seg_rev.update_layout(height=380, template="plotly_white", showlegend=False)
        st.plotly_chart(fig_seg_rev, use_container_width=True)

    # 3D RFM Scatter
    st.subheader("3D Behavioral Space (Recency vs Frequency vs Monetary)")
    fig_3d = px.scatter_3d(
        rfm_df,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Segment",
        opacity=0.8,
        hover_name="CustomerID",
        title="3D RFM Clustering",
        height=520,
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Segment Table
    st.subheader("Segment Analytics Summary")
    summary_table = rfm_df.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgOrders=("Frequency", "mean"),
        AvgSpend=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).reset_index()
    summary_table["AvgRecency"] = summary_table["AvgRecency"].round(1)
    summary_table["AvgOrders"] = summary_table["AvgOrders"].round(1)
    summary_table["AvgSpend"] = summary_table["AvgSpend"].apply(lambda v: f"${v:,.2f}")
    summary_table["TotalRevenue"] = summary_table["TotalRevenue"].apply(lambda v: f"${v:,.2f}")
    st.dataframe(summary_table, use_container_width=True)


# ----------------- TAB 3: CHURN PREDICTION (ML) -----------------
with tab3:
    st.subheader("Predictive Churn Risk Engine")
    st.caption("Supervised Machine Learning classification trained to anticipate customer drop-off before it happens.")

    c_c1, c_c2 = st.columns([1, 2])
    with c_c1:
        tier_counts = churn_df["RiskTier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Customers"]
        fig_risk = px.pie(
            tier_counts,
            values="Customers",
            names="Risk Tier",
            title="Customer Portfolio Churn Risk Tiers",
            hole=0.5,
            color="Risk Tier",
            color_discrete_map={"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"},
        )
        fig_risk.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_risk, use_container_width=True)

    with c_c2:
        # What-If Churn Simulator
        st.markdown("#### ⚡ Real-Time Churn What-If Simulator")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_freq = st.slider("Past Order Count (Frequency)", 1, 25, 4)
            sim_spend = st.slider("Total Spend ($)", 20.0, 3000.0, 350.0)
            sim_recency = st.slider("Days Since Last Purchase (Recency)", 1, 365, 45)
        with sim_col2:
            sim_sat = st.slider("Satisfaction Score (1-5)", 1, 5, 4)
            sim_tenure = st.slider("Customer Tenure (Days)", 30, 730, 240)
            sim_returns = st.slider("Return Rate (%)", 0, 100, 5) / 100.0

        # Heuristic / tree simulation approximation
        sim_score = (
            (sim_recency / 365.0) * 0.45
            - (sim_freq / 25.0) * 0.25
            - (sim_sat / 5.0) * 0.20
            + (sim_returns * 0.15)
            + 0.10
        )
        sim_prob = float(np.clip(sim_score, 0.02, 0.98))

        st.markdown(f"**Predicted Churn Probability:** `{sim_prob * 100:.1f}%`")
        if sim_prob > 0.70:
            st.error("🚨 Status: High Churn Risk — Recommend Immediate Automated VIP Winback Offer")
        elif sim_prob > 0.35:
            st.warning("⚠️ Status: Medium Churn Risk — Send Engagement Survey & Product Recommendations")
        else:
            st.success("✅ Status: Healthy Account — Target with Loyalty Rewards & Upsell Bundles")

    # High-Risk Customer Inspection
    st.subheader("🚨 Priority Retention Accounts (High Churn Probability)")
    high_risk_list = churn_df[churn_df["RiskTier"] == "High Risk"].sort_values(by="TotalSpend", ascending=False).head(15)
    display_cols = ["CustomerID", "TotalSpend", "Frequency", "Recency", "Satisfaction", "ChurnProbability", "RiskTier"]
    st.dataframe(high_risk_list[[c for c in display_cols if c in high_risk_list.columns]], use_container_width=True)


# ----------------- TAB 4: MARKET BASKET ANALYSIS -----------------
with tab4:
    st.subheader("🛍️ Market Basket Association Rules & Cross-Sell Engine")
    st.caption("Identify items frequently purchased together to boost basket size and average order value (AOV).")

    min_lift = st.slider("Filter by Minimum Lift", min_value=1.0, max_value=3.0, value=1.05, step=0.05)
    filtered_rules = rules_df[rules_df["Lift"] >= min_lift] if not rules_df.empty else pd.DataFrame()

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.markdown("#### Association Rules Table")
        st.dataframe(filtered_rules, use_container_width=True, height=360)

    with col_m2:
        st.markdown("#### 🎯 Interactive Product Recommender")
        all_products = sorted(rules_df["Antecedent"].unique().tolist()) if not rules_df.empty else []
        if all_products:
            selected_prod = st.selectbox("Select Primary Item in Cart:", all_products)
            recs = rules_df[rules_df["Antecedent"] == selected_prod].sort_values(by="Lift", ascending=False).head(3)
            if not recs.empty:
                st.write("**Recommended Add-to-Cart Items:**")
                for _, r in recs.iterrows():
                    st.info(f"✨ **{r['Consequent']}**\n- Confidence: `{r['Confidence']*100:.1f}%`\n- Lift: `{r['Lift']}x`")
            else:
                st.write("No strong co-purchase rules found for this product.")


# ----------------- TAB 5: CUSTOMER LIFETIME VALUE (CLV) -----------------
with tab5:
    st.subheader("💎 Customer Lifetime Value (CLV) Projections")
    st.caption("Discounted cash-flow projections combining historical spend, order cadence, and survival curves.")

    clv_c1, clv_c2 = st.columns(2)
    with clv_c1:
        tier_clv = clv_df.groupby("CLV_Tier")["TotalEstimatedCLV"].sum().reset_index().sort_values(by="TotalEstimatedCLV", ascending=False)
        fig_clv = px.bar(
            tier_clv,
            x="CLV_Tier",
            y="TotalEstimatedCLV",
            title="Total Projected Value by CLV Tier ($)",
            color="CLV_Tier",
            color_discrete_map={"Platinum": "#8b5cf6", "Gold": "#f59e0b", "Silver": "#94a3b8", "Bronze": "#b45309"},
        )
        fig_clv.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_clv, use_container_width=True)

    with clv_c2:
        fig_scatter_clv = px.scatter(
            clv_df,
            x="HistoricalProfit",
            y="PredictiveCLV_2Year",
            color="CLV_Tier",
            hover_name="CustomerID",
            title="Historical Profit vs 2-Year Projected CLV",
            labels={"HistoricalProfit": "Historical Profit ($)", "PredictiveCLV_2Year": "2-Year Projected CLV ($)"},
        )
        fig_scatter_clv.update_layout(height=360, template="plotly_white")
        st.plotly_chart(fig_scatter_clv, use_container_width=True)

    st.subheader("🏆 Top 20 Most Valuable Accounts")
    top_clv = clv_df.sort_values(by="TotalEstimatedCLV", ascending=False).head(20)
    st.dataframe(top_clv[["CustomerID", "Frequency", "Monetary", "HistoricalProfit", "PredictiveCLV_2Year", "TotalEstimatedCLV", "CLV_Tier"]], use_container_width=True)
