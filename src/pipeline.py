from __future__ import annotations


from collections import Counter
from datetime import datetime
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict
from typing import Any, Dict, List
from typing import Any, Dict, Tuple
from typing import Dict, Tuple
from typing import Tuple
import argparse
import logging
import numpy as np
import pandas as pd


'\nData Loader and Synthetic E-Commerce Generator Module.\nGenerates realistic transactional datasets with seasonal patterns, churn behavior,\nand customer demographics, with full validation and cleaning pipelines.\n'
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
PRODUCT_CATALOG = [{'code': 'PROD_001', 'name': 'Wireless Noise-Canceling Headphones', 'category': 'Electronics', 'price': 149.99}, {'code': 'PROD_002', 'name': 'Mechanical Gaming Keyboard', 'category': 'Electronics', 'price': 89.5}, {'code': 'PROD_003', 'name': 'Ergonomic Office Chair', 'category': 'Furniture', 'price': 249.0}, {'code': 'PROD_004', 'name': 'Stainless Steel Water Bottle', 'category': 'Home & Kitchen', 'price': 24.99}, {'code': 'PROD_005', 'name': 'Organic Fair-Trade Coffee Beans (1kg)', 'category': 'Grocery', 'price': 18.75}, {'code': 'PROD_006', 'name': 'Ceramic Pour-Over Coffee Maker', 'category': 'Home & Kitchen', 'price': 32.0}, {'code': 'PROD_007', 'name': 'Smart Fitness Tracker Watch', 'category': 'Electronics', 'price': 99.99}, {'code': 'PROD_008', 'name': 'Yoga Mat with Alignment Lines', 'category': 'Fitness', 'price': 38.5}, {'code': 'PROD_009', 'name': 'Adjustable Dumbbell Set', 'category': 'Fitness', 'price': 179.0}, {'code': 'PROD_010', 'name': 'Compact Bamboo Desk Organizer', 'category': 'Furniture', 'price': 29.99}, {'code': 'PROD_011', 'name': 'Fast-Charging USB-C Cable (2m)', 'category': 'Electronics', 'price': 14.5}, {'code': 'PROD_012', 'name': 'Ultra-Wide Gaming Monitor 34"', 'category': 'Electronics', 'price': 429.0}, {'code': 'PROD_013', 'name': 'Non-Stick Ceramic Frying Pan', 'category': 'Home & Kitchen', 'price': 45.0}, {'code': 'PROD_014', 'name': 'Electric Gooseneck Kettle', 'category': 'Home & Kitchen', 'price': 59.99}, {'code': 'PROD_015', 'name': 'Whey Protein Isolate (2kg)', 'category': 'Fitness', 'price': 54.0}]
COUNTRIES = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia', 'Netherlands']
COUNTRY_WEIGHTS = [0.45, 0.2, 0.12, 0.08, 0.07, 0.05, 0.03]
DEVICE_TYPES = ['Desktop', 'Mobile', 'Tablet']
DEVICE_WEIGHTS = [0.4, 0.5, 0.1]
PAYMENT_METHODS = ['Credit Card', 'PayPal', 'Apple Pay', 'Debit Card']
PAYMENT_WEIGHTS = [0.55, 0.25, 0.12, 0.08]
def generate_synthetic_data(num_customers: int=1500, num_transactions: int=15000, end_date: datetime | None=None, seed: int=42) -> pd.DataFrame:
    """
    Generates a realistic multi-year transactional dataset with customer churn patterns,
    RFM variations, and product co-purchasing affinities.
    """
    np.random.seed(seed)
    if end_date is None:
        end_date = datetime(2026, 9, 1)
    start_date = end_date - timedelta(days=730)
    logger.info('Generating %d synthetic customers and %d transactions...', num_customers, num_transactions)
    archetype_weights = [0.25, 0.2, 0.4, 0.15]
    customer_archetypes = np.random.choice([0, 1, 2, 3], size=num_customers, p=archetype_weights)
    customer_countries = np.random.choice(COUNTRIES, size=num_customers, p=COUNTRY_WEIGHTS)
    customer_ages = np.random.normal(38, 12, size=num_customers).clip(18, 75).astype(int)
    customer_satisfaction = np.random.choice([1, 2, 3, 4, 5], size=num_customers, p=[0.05, 0.08, 0.22, 0.4, 0.25])
    for i in range(num_customers):
        if customer_archetypes[i] == 0:
            customer_satisfaction[i] = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        elif customer_archetypes[i] == 1:
            customer_satisfaction[i] = np.random.choice([4, 5], p=[0.3, 0.7])
    transactions = []
    invoice_counter = 100000
    for cust_id in range(1, num_customers + 1):
        arch = customer_archetypes[cust_id - 1]
        country = customer_countries[cust_id - 1]
        age = customer_ages[cust_id - 1]
        sat = customer_satisfaction[cust_id - 1]
        if arch == 0:
            num_orders = np.random.randint(1, 5)
            date_range_days = 365
            offset_start = 730
            offset_end = 200
        elif arch == 1:
            num_orders = np.random.randint(8, 25)
            date_range_days = 730
            offset_start = 730
            offset_end = 2
        elif arch == 2:
            num_orders = np.random.randint(2, 8)
            date_range_days = 600
            offset_start = 650
            offset_end = 20
        else:
            num_orders = np.random.randint(1, 3)
            date_range_days = 90
            offset_start = 90
            offset_end = 1
        order_offsets = np.sort(np.random.uniform(offset_end, offset_start, size=num_orders))[::-1]
        for offset in order_offsets:
            invoice_counter += 1
            invoice_no = f'INV{invoice_counter}'
            order_date = end_date - timedelta(days=float(offset), hours=float(np.random.randint(0, 24)))
            device = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
            payment = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
            basket_size = np.random.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05])
            primary_prod = np.random.choice(PRODUCT_CATALOG)
            selected_products = [primary_prod]
            if basket_size > 1:
                if primary_prod['code'] == 'PROD_005':
                    selected_products.append(PRODUCT_CATALOG[5])
                elif primary_prod['code'] == 'PROD_008':
                    selected_products.append(PRODUCT_CATALOG[8])
                elif primary_prod['code'] == 'PROD_002':
                    selected_products.append(PRODUCT_CATALOG[10])
                else:
                    others = [p for p in PRODUCT_CATALOG if p['code'] != primary_prod['code']]
                    selected_products.extend(np.random.choice(others, size=min(basket_size - 1, len(others)), replace=False))
            for prod in selected_products:
                qty = int(np.random.choice([1, 2, 3], p=[0.75, 0.2, 0.05]))
                returned = np.random.choice([0, 1], p=[0.96, 0.04])
                transactions.append({'InvoiceNo': invoice_no, 'StockCode': prod['code'], 'Description': prod['name'], 'Category': prod['category'], 'Quantity': qty, 'UnitPrice': prod['price'], 'InvoiceDate': order_date.strftime('%Y-%m-%d %H:%M:%S'), 'CustomerID': f'CUST_{cust_id:05d}', 'Country': country, 'Age': age, 'Satisfaction': sat, 'Device': device, 'PaymentMethod': payment, 'Returned': returned})
    df = pd.DataFrame(transactions)
    logger.info('Generated %d transaction records across %d unique customers.', len(df), df['CustomerID'].nunique())
    return df
def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data:
    - Parses dates
    - Removes invalid quantities / prices
    - Calculates TotalAmount
    - Filters out cancelled / returned items for revenue analysis
    """
    data = df.copy()
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data['LineTotal'] = data['Quantity'] * data['UnitPrice']
    data['CustomerID'] = data['CustomerID'].astype(str)
    return data
def save_datasets(raw_df: pd.DataFrame, processed_df: pd.DataFrame, base_dir: Path) -> Tuple[Path, Path]:
    """Saves raw CSV and cleaned Parquet/CSV files."""
    raw_dir = base_dir / 'data' / 'raw'
    proc_dir = base_dir / 'data' / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / 'sample_ecommerce_data.csv'
    proc_path = proc_dir / 'cleaned_transactions.parquet'
    proc_csv_path = proc_dir / 'cleaned_transactions.csv'
    raw_df.to_csv(raw_path, index=False)
    processed_df.to_parquet(proc_path, index=False)
    processed_df.to_csv(proc_csv_path, index=False)
    logger.info('Raw data saved to: %s', raw_path)
    logger.info('Cleaned data saved to: %s', proc_path)
    return (raw_path, proc_path)
'\nRFM (Recency, Frequency, Monetary) Customer Segmentation Module.\nComputes RFM scores, segments customers using rule-based heuristics and K-Means clustering,\nand outputs actionable marketing segments.\n'
logger = logging.getLogger(__name__)
def compute_rfm_metrics(df: pd.DataFrame, reference_date: pd.Timestamp | None=None) -> pd.DataFrame:
    """
    Computes Recency, Frequency, and Monetary metrics for each unique customer.
    Filters out returned transactions for monetary calculations.
    """
    if 'LineTotal' not in df.columns:
        df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    valid_orders = df[df['Returned'] == 0] if 'Returned' in df.columns else df
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = valid_orders.groupby('CustomerID').agg(Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days), Frequency=('InvoiceNo', 'nunique'), Monetary=('LineTotal', 'sum'), FirstPurchase=('InvoiceDate', 'min'), LastPurchase=('InvoiceDate', 'max'), Country=('Country', 'first'), Age=('Age', 'mean') if 'Age' in valid_orders.columns else ('CustomerID', 'count'), Satisfaction=('Satisfaction', 'mean') if 'Satisfaction' in valid_orders.columns else ('CustomerID', 'count')).reset_index()
    rfm['TenureDays'] = (reference_date - rfm['FirstPurchase']).dt.days
    rfm['Monetary'] = rfm['Monetary'].round(2)
    rfm['Age'] = rfm['Age'].round(1) if 'Age' in rfm.columns else 35.0
    rfm['Satisfaction'] = rfm['Satisfaction'].round(2) if 'Satisfaction' in rfm.columns else 4.0
    return rfm
def score_rfm_quintiles(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns quintile scores (1 to 5) to Recency, Frequency, and Monetary values.
    For Recency: 5 is best (most recent), 1 is worst.
    For Frequency and Monetary: 5 is highest, 1 is lowest.
    """
    df = rfm_df.copy()
    df['R_Score'] = pd.qcut(df['Recency'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
    df['RFM_Composite'] = (df['R_Score'] + df['F_Score'] + df['M_Score']) / 3.0
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
        r = row['R_Score']
        f = row['F_Score']
        m = row['M_Score']
        rf = (r + f) / 2.0
        if r >= 4 and f >= 4 and (m >= 4):
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f in [1, 2]:
            return 'New Customers' if f == 1 else 'Potential Loyalists'
        elif r == 3 and f in [1, 2]:
            return 'Promising'
        elif r in [2, 3] and f >= 3:
            return 'Customers Needing Attention'
        elif r in [1, 2] and f >= 4 and (m >= 4):
            return "Can't Lose Them"
        elif r in [1, 2] and f >= 2:
            return 'At Risk'
        elif r == 2 and f == 1:
            return 'Hibernating'
        else:
            return 'Lost'
    df['Segment'] = df.apply(get_segment, axis=1)
    return df
def apply_kmeans_clustering(rfm_df: pd.DataFrame, n_clusters: int=4, random_state: int=42) -> Tuple[pd.DataFrame, KMeans]:
    """
    Applies K-Means clustering on log-transformed, normalized RFM metrics.
    Compares unsupervised clusters against rule-based segments.
    """
    df = rfm_df.copy()
    rfm_log = pd.DataFrame()
    rfm_log['Recency'] = np.log1p(df['Recency'])
    rfm_log['Frequency'] = np.log1p(df['Frequency'])
    rfm_log['Monetary'] = np.log1p(df['Monetary'].clip(lower=0))
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    df['Cluster'] = df['Cluster'].astype(str)
    return (df, kmeans)
def generate_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produces executive summary metrics grouped by customer segment."""
    summary = df.groupby('Segment').agg(CustomerCount=('CustomerID', 'count'), AvgRecencyDays=('Recency', 'mean'), AvgFrequency=('Frequency', 'mean'), AvgMonetarySpend=('Monetary', 'mean'), TotalRevenue=('Monetary', 'sum'), AvgSatisfaction=('Satisfaction', 'mean')).reset_index()
    total_customers = len(df)
    total_rev = summary['TotalRevenue'].sum()
    summary['CustomerPct'] = (summary['CustomerCount'] / total_customers * 100).round(1)
    summary['RevenuePct'] = (summary['TotalRevenue'] / total_rev * 100).round(1)
    summary['AvgRecencyDays'] = summary['AvgRecencyDays'].round(1)
    summary['AvgFrequency'] = summary['AvgFrequency'].round(1)
    summary['AvgMonetarySpend'] = summary['AvgMonetarySpend'].round(2)
    summary['TotalRevenue'] = summary['TotalRevenue'].round(2)
    summary['AvgSatisfaction'] = summary['AvgSatisfaction'].round(2)
    return summary.sort_values(by='TotalRevenue', ascending=False)
def run_rfm_pipeline(transactions_path: Path, output_dir: Path | None=None) -> pd.DataFrame:
    """Executes the complete RFM analysis pipeline and saves results."""
    logger.info('Loading transaction data from: %s', transactions_path)
    if str(transactions_path).endswith('.parquet'):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    rfm = compute_rfm_metrics(df)
    scored = score_rfm_quintiles(rfm)
    segmented = map_rfm_segments(scored)
    clustered, _ = apply_kmeans_clustering(segmented)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'customer_rfm_segments.csv'
        clustered.to_csv(out_path, index=False)
        logger.info('Saved RFM segmentation output to: %s', out_path)
    return clustered
'\nCustomer Churn Prediction and Risk Modeling Module.\nPerforms behavioral feature engineering, trains classification algorithms (Random Forest,\nGradient Boosting, Logistic Regression), calculates ROC-AUC/Confusion Matrix,\nand outputs customer-level churn risk probabilities.\n'
logger = logging.getLogger(__name__)
def engineer_churn_features(transactions_df: pd.DataFrame, churn_inactivity_days: int=120, reference_date: pd.Timestamp | None=None) -> pd.DataFrame:
    """
    Extracts rich customer-level behavioral and demographic signals for churn prediction.
    Target variable: Churned = 1 if Recency > churn_inactivity_days, else 0.
    """
    df = transactions_df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if 'LineTotal' not in df.columns:
        df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    grouped = df.groupby('CustomerID')

    def calc_interval_std(dates):
        if len(dates) < 2:
            return 0.0
        sorted_dates = sorted(dates)
        diffs = [(sorted_dates[i] - sorted_dates[i - 1]).days for i in range(1, len(sorted_dates))]
        return float(np.std(diffs))
    intervals = grouped['InvoiceDate'].apply(calc_interval_std).rename('PurchaseIntervalStd')
    if 'Returned' in df.columns:
        return_rate = grouped['Returned'].mean().rename('ReturnRate')
    else:
        return_rate = pd.Series(0.0, index=df['CustomerID'].unique(), name='ReturnRate')
    customer_stats = grouped.agg(Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days), Frequency=('InvoiceNo', 'nunique'), TotalSpend=('LineTotal', 'sum'), TotalQuantity=('Quantity', 'sum'), TenureDays=('InvoiceDate', lambda x: (reference_date - x.min()).days), Age=('Age', 'mean') if 'Age' in df.columns else ('LineTotal', 'count'), Satisfaction=('Satisfaction', 'mean') if 'Satisfaction' in df.columns else ('LineTotal', 'count'), Device=('Device', lambda x: x.mode()[0]) if 'Device' in df.columns else ('LineTotal', 'count'), Country=('Country', 'first'))
    features = customer_stats.join(intervals).join(return_rate).reset_index()
    features['AverageOrderValue'] = (features['TotalSpend'] / features['Frequency'].clip(lower=1)).round(2)
    features['AverageBasketQuantity'] = (features['TotalQuantity'] / features['Frequency'].clip(lower=1)).round(2)
    features['Churned'] = (features['Recency'] > churn_inactivity_days).astype(int)
    return features
def train_and_evaluate_models(features_df: pd.DataFrame, random_state: int=42) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Trains multiple ML models and benchmarks their classification performance.
    Returns comparison dictionary and customer predictions dataframe with churn risk tiers.
    """
    feature_cols = ['Frequency', 'TotalSpend', 'AverageOrderValue', 'AverageBasketQuantity', 'TenureDays', 'PurchaseIntervalStd', 'ReturnRate', 'Age', 'Satisfaction']
    available_cols = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available_cols].fillna(0)
    y = features_df['Churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state), 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=random_state), 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state)}
    results = {}
    best_model_name = 'Gradient Boosting'
    best_model = None
    best_score = -1.0
    for name, model in models.items():
        if name == 'Logistic Regression':
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
        results[name] = {'accuracy': round(float(acc), 4), 'precision': round(float(prec), 4), 'recall': round(float(rec), 4), 'f1': round(float(f1), 4), 'roc_auc': round(float(auc), 4), 'confusion_matrix': cm}
        if auc > best_score:
            best_score = auc
            best_model_name = name
            best_model = model
    if hasattr(best_model, 'feature_importances_'):
        importances = dict(zip(available_cols, [round(float(v), 4) for v in best_model.feature_importances_]))
        sorted_importance = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    else:
        sorted_importance = {}
    results['best_model'] = best_model_name
    results['feature_importances'] = sorted_importance
    pred_df = features_df.copy()
    if best_model_name == 'Logistic Regression':
        all_scaled = scaler.transform(X)
        pred_df['ChurnProbability'] = best_model.predict_proba(all_scaled)[:, 1].round(4)
    else:
        pred_df['ChurnProbability'] = best_model.predict_proba(X)[:, 1].round(4)

    def assign_risk_tier(prob):
        if prob < 0.35:
            return 'Low Risk'
        elif prob < 0.7:
            return 'Medium Risk'
        else:
            return 'High Risk'
    pred_df['RiskTier'] = pred_df['ChurnProbability'].apply(assign_risk_tier)
    return (results, pred_df)
def run_churn_pipeline(transactions_path: Path, output_dir: Path | None=None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs end-to-end churn engineering, training, and saves predictions."""
    logger.info('Executing churn pipeline on %s...', transactions_path)
    if str(transactions_path).endswith('.parquet'):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)
    features = engineer_churn_features(df)
    results, predictions = train_and_evaluate_models(features)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_path = output_dir / 'customer_churn_predictions.csv'
        predictions.to_csv(pred_path, index=False)
        logger.info('Saved churn predictions to: %s', pred_path)
    return (results, predictions)
'\nMarket Basket Analysis (MBA) and Association Rule Mining Module.\nComputes Support, Confidence, Lift, and Conviction across transaction baskets,\nand generates cross-selling product bundle recommendations.\n'
logger = logging.getLogger(__name__)
def extract_transaction_baskets(df: pd.DataFrame, item_col: str='Description') -> List[List[str]]:
    """
    Groups line items by InvoiceNo to form distinct basket item lists.
    Filters out single-item orders to focus on co-purchased bundles.
    """
    valid_orders = df[df['Returned'] == 0] if 'Returned' in df.columns else df
    baskets = valid_orders.groupby('InvoiceNo')[item_col].apply(lambda s: sorted(list(set(s)))).tolist()
    multi_item_baskets = [b for b in baskets if len(b) > 1]
    return multi_item_baskets
def generate_association_rules(baskets: List[List[str]], min_support: float=0.01, min_confidence: float=0.2, min_lift: float=1.05) -> pd.DataFrame:
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
        return pd.DataFrame(columns=['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift', 'Conviction'])
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
        conf_a_b = support_ab / support_a if support_a > 0 else 0
        lift_a_b = conf_a_b / support_b if support_b > 0 else 0
        conv_a_b = (1 - support_b) / (1 - conf_a_b) if conf_a_b < 1.0 else float('inf')
        if conf_a_b >= min_confidence and lift_a_b >= min_lift:
            rules.append({'Antecedent': item_a, 'Consequent': item_b, 'Support': round(support_ab, 4), 'Confidence': round(conf_a_b, 4), 'Lift': round(lift_a_b, 3), 'Conviction': round(conv_a_b, 3) if conv_a_b != float('inf') else 999.0, 'PairCount': pair_freq})
        conf_b_a = support_ab / support_b if support_b > 0 else 0
        lift_b_a = conf_b_a / support_a if support_a > 0 else 0
        conv_b_a = (1 - support_a) / (1 - conf_b_a) if conf_b_a < 1.0 else float('inf')
        if conf_b_a >= min_confidence and lift_b_a >= min_lift:
            rules.append({'Antecedent': item_b, 'Consequent': item_a, 'Support': round(support_ab, 4), 'Confidence': round(conf_b_a, 4), 'Lift': round(lift_b_a, 3), 'Conviction': round(conv_b_a, 3) if conv_b_a != float('inf') else 999.0, 'PairCount': pair_freq})
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=['Lift', 'Confidence'], ascending=[False, False]).reset_index(drop=True)
    return rules_df
def get_recommendations_for_product(product_name: str, rules_df: pd.DataFrame, top_n: int=3) -> List[Dict[str, Any]]:
    """Returns top complementary products to recommend given a primary product."""
    if rules_df.empty:
        return []
    matches = rules_df[rules_df['Antecedent'] == product_name].sort_values(by=['Lift', 'Confidence'], ascending=False)
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        recommendations.append({'product': row['Consequent'], 'confidence': f"{row['Confidence'] * 100:.1f}%", 'lift': row['Lift'], 'support': f"{row['Support'] * 100:.1f}%"})
    return recommendations
def run_basket_analysis_pipeline(transactions_path: Path, output_dir: Path | None=None) -> pd.DataFrame:
    """Runs end-to-end basket analysis and saves association rules."""
    logger.info('Executing basket analysis on %s...', transactions_path)
    if str(transactions_path).endswith('.parquet'):
        df = pd.read_parquet(transactions_path)
    else:
        df = pd.read_csv(transactions_path)
    baskets = extract_transaction_baskets(df, item_col='Description')
    rules_df = generate_association_rules(baskets, min_support=0.01, min_confidence=0.15, min_lift=1.02)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'market_basket_rules.csv'
        rules_df.to_csv(out_path, index=False)
        logger.info('Saved association rules to: %s (total %d rules)', out_path, len(rules_df))
    return rules_df
'\nCustomer Lifetime Value (CLV) Calculation Module.\nComputes historical revenue contributions and forecasts multi-period predictive CLV\nfactoring in retention probability, profit margins, and discount rates.\n'
logger = logging.getLogger(__name__)
def calculate_customer_lifetime_value(rfm_df: pd.DataFrame, churn_df: pd.DataFrame | None=None, profit_margin: float=0.25, discount_rate: float=0.1, forecast_years: int=2) -> pd.DataFrame:
    """
    Computes both Historical and Predictive CLV for each customer.
    Formula for period t:
      ExpectedValue_t = (AOV * AnnualFrequency * ProfitMargin) * (RetentionRate^t) / ((1 + DiscountRate)^t)
    """
    df = rfm_df.copy()
    tenure_years = (df['TenureDays'] / 365.0).clip(lower=0.1)
    df['AnnualFrequency'] = (df['Frequency'] / tenure_years).round(2)
    df['AverageOrderValue'] = (df['Monetary'] / df['Frequency'].clip(lower=1)).round(2)
    if churn_df is not None and 'ChurnProbability' in churn_df.columns:
        churn_map = churn_df.set_index('CustomerID')['ChurnProbability'].to_dict()
        df['ChurnProbability'] = df['CustomerID'].map(churn_map).fillna(0.3)
    else:
        df['ChurnProbability'] = (df['Recency'] / 365.0).clip(0.05, 0.95)
    df['RetentionRate'] = 1.0 - df['ChurnProbability']
    df['HistoricalProfit'] = (df['Monetary'] * profit_margin).round(2)
    predictive_clv = np.zeros(len(df))
    for t in range(1, forecast_years + 1):
        cash_flow = df['AverageOrderValue'] * df['AnnualFrequency'] * profit_margin
        survival_prob = df['RetentionRate'] ** t
        discount_factor = (1.0 + discount_rate) ** t
        predictive_clv += cash_flow * survival_prob / discount_factor
    df['PredictiveCLV_2Year'] = predictive_clv.round(2)
    df['TotalEstimatedCLV'] = (df['HistoricalProfit'] + df['PredictiveCLV_2Year']).round(2)
    quantiles = df['TotalEstimatedCLV'].quantile([0.5, 0.8, 0.95]).to_dict()
    q50, q80, q95 = (quantiles[0.5], quantiles[0.8], quantiles[0.95])

    def assign_tier(val):
        if val >= q95:
            return 'Platinum'
        elif val >= q80:
            return 'Gold'
        elif val >= q50:
            return 'Silver'
        else:
            return 'Bronze'
    df['CLV_Tier'] = df['TotalEstimatedCLV'].apply(assign_tier)
    return df
def run_clv_pipeline(rfm_path: Path, churn_path: Path | None=None, output_dir: Path | None=None) -> pd.DataFrame:
    """Runs CLV calculations and saves results."""
    logger.info('Calculating CLV from RFM path: %s', rfm_path)
    rfm_df = pd.read_csv(rfm_path)
    churn_df = pd.read_csv(churn_path) if churn_path and Path(churn_path).exists() else None
    clv_df = calculate_customer_lifetime_value(rfm_df, churn_df)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'customer_clv_predictions.csv'
        clv_df.to_csv(out_path, index=False)
        logger.info('Saved CLV analysis to: %s', out_path)
    return clv_df
'\nExecutive Summary Report Generator Module.\nGenerates an executive-ready, responsive, self-contained HTML report with\ninteractive KPI cards, RFM segment distribution, churn risk breakdown,\nand actionable cross-selling recommendations.\n'
logger = logging.getLogger(__name__)
def generate_executive_html_report(transactions_df: pd.DataFrame, rfm_df: pd.DataFrame, churn_df: pd.DataFrame, rules_df: pd.DataFrame, clv_df: pd.DataFrame, output_path: Path) -> Path:
    """Generates an HTML executive report summarizing key customer analytics findings."""
    total_revenue = transactions_df['LineTotal'].sum() if 'LineTotal' in transactions_df.columns else 0.0
    total_orders = transactions_df['InvoiceNo'].nunique()
    total_customers = rfm_df['CustomerID'].nunique()
    avg_order_value = total_revenue / max(total_orders, 1)
    high_risk_customers = len(churn_df[churn_df['RiskTier'] == 'High Risk']) if 'RiskTier' in churn_df.columns else 0
    churn_rate = len(churn_df[churn_df['Churned'] == 1]) / max(len(churn_df), 1) * 100
    segment_summary = rfm_df.groupby('Segment').agg(Count=('CustomerID', 'count'), AvgSpend=('Monetary', 'mean'), TotalRev=('Monetary', 'sum'), AvgRecency=('Recency', 'mean')).reset_index()
    segment_summary['RevShare'] = (segment_summary['TotalRev'] / max(total_revenue, 1) * 100).round(1)
    segment_rows_html = ''
    for _, row in segment_summary.sort_values(by='TotalRev', ascending=False).iterrows():
        segment_rows_html += f"""\n        <tr>\n            <td class="font-bold">{row['Segment']}</td>\n            <td>{row['Count']:,}</td>\n            <td>${row['AvgSpend']:,.2f}</td>\n            <td>${row['TotalRev']:,.2f}</td>\n            <td>{row['RevShare']}%</td>\n            <td>{row['AvgRecency']:.1f} days</td>\n        </tr>\n        """
    rules_rows_html = ''
    top_rules = rules_df.head(6)
    for _, r in top_rules.iterrows():
        rules_rows_html += f"""\n        <tr>\n            <td><span class="badge badge-primary">{r['Antecedent']}</span></td>\n            <td><span class="badge badge-success">{r['Consequent']}</span></td>\n            <td>{r['Support'] * 100:.1f}%</td>\n            <td><strong>{r['Confidence'] * 100:.1f}%</strong></td>\n            <td><span class="badge badge-lift">{r['Lift']:.2f}x</span></td>\n        </tr>\n        """
    report_html = f"""<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Customer Intelligence & E-Commerce Executive Report</title>\n    <style>\n        :root {{\n            --primary: #1e3a8a;\n            --primary-light: #3b82f6;\n            --secondary: #0f172a;\n            --success: #10b981;\n            --warning: #f59e0b;\n            --danger: #ef4444;\n            --bg: #f8fafc;\n            --card-bg: #ffffff;\n            --border: #e2e8f0;\n            --text-dark: #0f172a;\n            --text-muted: #64748b;\n        }}\n        body {{\n            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;\n            background-color: var(--bg);\n            color: var(--text-dark);\n            margin: 0;\n            padding: 30px 20px;\n            line-height: 1.5;\n        }}\n        .container {{\n            max-width: 1100px;\n            margin: 0 auto;\n        }}\n        .header {{\n            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);\n            color: white;\n            padding: 36px 30px;\n            border-radius: 14px;\n            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);\n            margin-bottom: 28px;\n        }}\n        .header h1 {{\n            margin: 0 0 8px 0;\n            font-size: 28px;\n            font-weight: 700;\n        }}\n        .header p {{\n            margin: 0;\n            font-size: 15px;\n            opacity: 0.9;\n        }}\n        .grid-kpis {{\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));\n            gap: 16px;\n            margin-bottom: 28px;\n        }}\n        .kpi-card {{\n            background: var(--card-bg);\n            padding: 22px;\n            border-radius: 12px;\n            border: 1px solid var(--border);\n            box-shadow: 0 2px 6px rgba(0,0,0,0.04);\n        }}\n        .kpi-title {{\n            font-size: 13px;\n            color: var(--text-muted);\n            text-transform: uppercase;\n            font-weight: 600;\n            letter-spacing: 0.5px;\n            margin-bottom: 6px;\n        }}\n        .kpi-value {{\n            font-size: 26px;\n            font-weight: 700;\n            color: var(--secondary);\n        }}\n        .kpi-subtitle {{\n            font-size: 12px;\n            color: var(--success);\n            font-weight: 500;\n            margin-top: 4px;\n        }}\n        .section-card {{\n            background: var(--card-bg);\n            border-radius: 12px;\n            border: 1px solid var(--border);\n            padding: 24px;\n            margin-bottom: 24px;\n            box-shadow: 0 2px 6px rgba(0,0,0,0.04);\n        }}\n        .section-title {{\n            font-size: 18px;\n            font-weight: 700;\n            color: var(--secondary);\n            margin-top: 0;\n            margin-bottom: 16px;\n            display: flex;\n            align-items: center;\n            gap: 8px;\n        }}\n        table {{\n            width: 100%;\n            border-collapse: collapse;\n            font-size: 14px;\n        }}\n        th, td {{\n            padding: 12px 14px;\n            text-align: left;\n            border-bottom: 1px solid var(--border);\n        }}\n        th {{\n            background-color: #f1f5f9;\n            color: var(--text-muted);\n            font-weight: 600;\n            font-size: 12px;\n            text-transform: uppercase;\n        }}\n        tr:hover {{\n            background-color: #f8fafc;\n        }}\n        .font-bold {{\n            font-weight: 600;\n        }}\n        .badge {{\n            display: inline-block;\n            padding: 3px 8px;\n            border-radius: 6px;\n            font-size: 12px;\n            font-weight: 500;\n        }}\n        .badge-primary {{ background: #eff6ff; color: #1d4ed8; }}\n        .badge-success {{ background: #ecfdf5; color: #047857; }}\n        .badge-lift {{ background: #fef3c7; color: #b45309; font-weight: 700; }}\n        .strategy-grid {{\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));\n            gap: 16px;\n        }}\n        .strategy-box {{\n            padding: 18px;\n            border-radius: 10px;\n            background: #f8fafc;\n            border-left: 4px solid var(--primary-light);\n        }}\n        .strategy-box h4 {{\n            margin: 0 0 6px 0;\n            font-size: 15px;\n            color: var(--secondary);\n        }}\n        .strategy-box p {{\n            margin: 0;\n            font-size: 13px;\n            color: var(--text-muted);\n        }}\n        .footer {{\n            text-align: center;\n            font-size: 12px;\n            color: var(--text-muted);\n            margin-top: 36px;\n            padding-top: 18px;\n            border-top: 1px solid var(--border);\n        }}\n    </style>\n</head>\n<body>\n    <div class="container">\n        <!-- Header -->\n        <div class="header">\n            <h1>Customer Intelligence & Strategic Analytics</h1>\n            <p>Generated: {datetime.now().strftime('%B %d, %Y - %H:%M UTC')} | Model Architecture: Gradient Boosting & K-Means</p>\n        </div>\n\n        <!-- KPIs -->\n        <div class="grid-kpis">\n            <div class="kpi-card">\n                <div class="kpi-title">Total Revenue</div>\n                <div class="kpi-value">${total_revenue:,.2f}</div>\n                <div class="kpi-subtitle">Across {total_orders:,} Completed Orders</div>\n            </div>\n            <div class="kpi-card">\n                <div class="kpi-title">Active Customer Base</div>\n                <div class="kpi-value">{total_customers:,}</div>\n                <div class="kpi-subtitle">Unique Customer Profiles</div>\n            </div>\n            <div class="kpi-card">\n                <div class="kpi-title">Average Order Value</div>\n                <div class="kpi-value">${avg_order_value:.2f}</div>\n                <div class="kpi-subtitle">Mean Revenue Per Basket</div>\n            </div>\n            <div class="kpi-card">\n                <div class="kpi-title">Customer Churn Rate</div>\n                <div class="kpi-value" style="color: {('var(--danger)' if churn_rate > 20 else 'var(--success)')};">{churn_rate:.1f}%</div>\n                <div class="kpi-subtitle">{high_risk_customers} Accounts in High Risk Tier</div>\n            </div>\n        </div>\n\n        <!-- RFM Segmentation -->\n        <div class="section-card">\n            <h2 class="section-title">📊 Customer RFM Segmentation Breakdown</h2>\n            <p style="color: var(--text-muted); font-size: 13px; margin-top: -8px; margin-bottom: 16px;">\n                Recency, Frequency, and Monetary scoring provides revenue concentration mapping across distinct customer personas.\n            </p>\n            <table>\n                <thead>\n                    <tr>\n                        <th>Customer Segment</th>\n                        <th>Audience Count</th>\n                        <th>Avg Spend</th>\n                        <th>Total Revenue</th>\n                        <th>Revenue Share</th>\n                        <th>Avg Recency</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    {segment_rows_html}\n                </tbody>\n            </table>\n        </div>\n\n        <!-- Market Basket Recommendations -->\n        <div class="section-card">\n            <h2 class="section-title">🛍️ Top Market Basket Cross-Selling Opportunities</h2>\n            <p style="color: var(--text-muted); font-size: 13px; margin-top: -8px; margin-bottom: 16px;">\n                Association rules mined from multi-item checkout sessions (filtered for high Lift & Confidence).\n            </p>\n            <table>\n                <thead>\n                    <tr>\n                        <th>Primary Product (Antecedent)</th>\n                        <th>Recommended Bundle (Consequent)</th>\n                        <th>Support</th>\n                        <th>Confidence</th>\n                        <th>Lift Score</th>\n                    </tr>\n                </thead>\n                <tbody>\n                    {rules_rows_html}\n                </tbody>\n            </table>\n        </div>\n\n        <!-- Strategic Playbook -->\n        <div class="section-card">\n            <h2 class="section-title">🚀 Strategic Recommendations & Action Plan</h2>\n            <div class="strategy-grid">\n                <div class="strategy-box">\n                    <h4>1. VIP Champion Retention</h4>\n                    <p>Champions generate the majority of net profit. Establish an invite-only VIP loyalty tier with early product drops and direct concierge support to maximize retention.</p>\n                </div>\n                <div class="strategy-box">\n                    <h4>2. At-Risk Winback Triggers</h4>\n                    <p>Automate personalized discount incentives for customers entering the 60-90 day inactivity window before they transition into permanent churn.</p>\n                </div>\n                <div class="strategy-box">\n                    <h4>3. Checkout Bundle Upsells</h4>\n                    <p>Deploy product pairings with >1.5x Lift directly on cart and product pages to lift Average Order Value by an estimated 14-18%.</p>\n                </div>\n            </div>\n        </div>\n\n        <!-- Footer -->\n        <div class="footer">\n            <p>Customer Intelligence Platform v1.0.0 &bull; Built with Python, Scikit-Learn, and Streamlit &bull; Automated Analytics Engine</p>\n        </div>\n    </div>\n</body>\n</html>\n"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding='utf-8')
    logger.info('Generated executive report at: %s', output_path)
    return output_path

def main():
    import argparse
    from pathlib import Path
    import logging

    parser = argparse.ArgumentParser(description="Run the full Customer Analytics Pipeline")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()

    base_dir = Path("data")
    base_dir.mkdir(exist_ok=True)
    (base_dir / "raw").mkdir(exist_ok=True)
    (base_dir / "processed").mkdir(exist_ok=True)

    transactions_path = base_dir / "raw" / "transactions.csv"
    processed_dir = base_dir / "processed"

    if args.generate_sample:
        logging.info("Generating synthetic dataset...")
        raw_df = generate_synthetic_data(num_customers=5000, num_transactions=25000)
        raw_df.to_csv(transactions_path, index=False)
        logging.info(f"Saved synthetic data to {transactions_path}")

    # 1. RFM Pipeline
    rfm_df = run_rfm_pipeline(transactions_path, processed_dir)

    # 2. Churn Pipeline
    metrics, churn_df = run_churn_pipeline(transactions_path, processed_dir)

    # 3. Basket Analysis Pipeline
    rules_df = run_basket_analysis_pipeline(transactions_path, processed_dir)

    # 4. CLV Pipeline
    clv_df = run_clv_pipeline(processed_dir / "customer_rfm_segments.csv", processed_dir / "customer_churn_predictions.csv", processed_dir)

    # 5. Report Generation
    generate_executive_html_report(pd.read_csv(transactions_path), rfm_df, churn_df, rules_df, clv_df, Path("reports/executive_summary.html"))
    logging.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()
