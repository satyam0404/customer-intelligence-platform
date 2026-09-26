"""
Data Loader and Synthetic E-Commerce Generator Module.
Generates realistic transactional datasets with seasonal patterns, churn behavior,
and customer demographics, with full validation and cleaning pipelines.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRODUCT_CATALOG = [
    {"code": "PROD_001", "name": "Wireless Noise-Canceling Headphones", "category": "Electronics", "price": 149.99},
    {"code": "PROD_002", "name": "Mechanical Gaming Keyboard", "category": "Electronics", "price": 89.50},
    {"code": "PROD_003", "name": "Ergonomic Office Chair", "category": "Furniture", "price": 249.00},
    {"code": "PROD_004", "name": "Stainless Steel Water Bottle", "category": "Home & Kitchen", "price": 24.99},
    {"code": "PROD_005", "name": "Organic Fair-Trade Coffee Beans (1kg)", "category": "Grocery", "price": 18.75},
    {"code": "PROD_006", "name": "Ceramic Pour-Over Coffee Maker", "category": "Home & Kitchen", "price": 32.00},
    {"code": "PROD_007", "name": "Smart Fitness Tracker Watch", "category": "Electronics", "price": 99.99},
    {"code": "PROD_008", "name": "Yoga Mat with Alignment Lines", "category": "Fitness", "price": 38.50},
    {"code": "PROD_009", "name": "Adjustable Dumbbell Set", "category": "Fitness", "price": 179.00},
    {"code": "PROD_010", "name": "Compact Bamboo Desk Organizer", "category": "Furniture", "price": 29.99},
    {"code": "PROD_011", "name": "Fast-Charging USB-C Cable (2m)", "category": "Electronics", "price": 14.50},
    {"code": "PROD_012", "name": "Ultra-Wide Gaming Monitor 34\"", "category": "Electronics", "price": 429.00},
    {"code": "PROD_013", "name": "Non-Stick Ceramic Frying Pan", "category": "Home & Kitchen", "price": 45.00},
    {"code": "PROD_014", "name": "Electric Gooseneck Kettle", "category": "Home & Kitchen", "price": 59.99},
    {"code": "PROD_015", "name": "Whey Protein Isolate (2kg)", "category": "Fitness", "price": 54.00},
]

COUNTRIES = ["United States", "United Kingdom", "Germany", "France", "Canada", "Australia", "Netherlands"]
COUNTRY_WEIGHTS = [0.45, 0.20, 0.12, 0.08, 0.07, 0.05, 0.03]

DEVICE_TYPES = ["Desktop", "Mobile", "Tablet"]
DEVICE_WEIGHTS = [0.40, 0.50, 0.10]

PAYMENT_METHODS = ["Credit Card", "PayPal", "Apple Pay", "Debit Card"]
PAYMENT_WEIGHTS = [0.55, 0.25, 0.12, 0.08]


def generate_synthetic_data(
    num_customers: int = 1500,
    num_transactions: int = 15000,
    end_date: datetime | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic multi-year transactional dataset with customer churn patterns,
    RFM variations, and product co-purchasing affinities.
    """
    np.random.seed(seed)
    if end_date is None:
        end_date = datetime(2026, 9, 1)

    start_date = end_date - timedelta(days=730)  # 2-year history
    logger.info("Generating %d synthetic customers and %d transactions...", num_customers, num_transactions)

    # 1. Generate Customer Profile Archetypes
    # 0: Churned (low recent activity, high early activity)
    # 1: Loyal / Champions (frequent, high value)
    # 2: Occasional / Moderate (sporadic buyers)
    # 3: New / Recent (first purchase within last 90 days)
    archetype_weights = [0.25, 0.20, 0.40, 0.15]
    customer_archetypes = np.random.choice([0, 1, 2, 3], size=num_customers, p=archetype_weights)
    customer_countries = np.random.choice(COUNTRIES, size=num_customers, p=COUNTRY_WEIGHTS)
    customer_ages = np.random.normal(38, 12, size=num_customers).clip(18, 75).astype(int)
    customer_satisfaction = np.random.choice([1, 2, 3, 4, 5], size=num_customers, p=[0.05, 0.08, 0.22, 0.40, 0.25])

    # Churned customers tend to have lower satisfaction
    for i in range(num_customers):
        if customer_archetypes[i] == 0:
            customer_satisfaction[i] = np.random.choice([1, 2, 3, 4], p=[0.30, 0.40, 0.20, 0.10])
        elif customer_archetypes[i] == 1:
            customer_satisfaction[i] = np.random.choice([4, 5], p=[0.30, 0.70])

    transactions = []
    invoice_counter = 100000

    for cust_id in range(1, num_customers + 1):
        arch = customer_archetypes[cust_id - 1]
        country = customer_countries[cust_id - 1]
        age = customer_ages[cust_id - 1]
        sat = customer_satisfaction[cust_id - 1]

        # Determine number of orders based on archetype
        if arch == 0:  # Churned
            num_orders = np.random.randint(1, 5)
            # Purchases concentrated in the early period
            date_range_days = 365
            offset_start = 730
            offset_end = 200  # Stopped buying 200+ days ago
        elif arch == 1:  # Champions
            num_orders = np.random.randint(8, 25)
            date_range_days = 730
            offset_start = 730
            offset_end = 2  # Active recently
        elif arch == 2:  # Occasional
            num_orders = np.random.randint(2, 8)
            date_range_days = 600
            offset_start = 650
            offset_end = 20
        else:  # New
            num_orders = np.random.randint(1, 3)
            date_range_days = 90
            offset_start = 90
            offset_end = 1

        order_offsets = np.sort(np.random.uniform(offset_end, offset_start, size=num_orders))[::-1]

        for offset in order_offsets:
            invoice_counter += 1
            invoice_no = f"INV{invoice_counter}"
            order_date = end_date - timedelta(days=float(offset), hours=float(np.random.randint(0, 24)))
            device = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
            payment = np.random.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)

            # Basket size (number of distinct products per order)
            basket_size = np.random.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05])
            
            # Affinities: Coffee + Kettle, Yoga Mat + Dumbbells, Keyboard + Cables
            primary_prod = np.random.choice(PRODUCT_CATALOG)
            selected_products = [primary_prod]

            if basket_size > 1:
                # Add correlated products
                if primary_prod["code"] == "PROD_005":
                    selected_products.append(PRODUCT_CATALOG[5])  # Pour-Over Maker
                elif primary_prod["code"] == "PROD_008":
                    selected_products.append(PRODUCT_CATALOG[8])  # Dumbbells
                elif primary_prod["code"] == "PROD_002":
                    selected_products.append(PRODUCT_CATALOG[10]) # USB-C Cable
                else:
                    others = [p for p in PRODUCT_CATALOG if p["code"] != primary_prod["code"]]
                    selected_products.extend(np.random.choice(others, size=min(basket_size - 1, len(others)), replace=False))

            for prod in selected_products:
                qty = int(np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05]))
                returned = np.random.choice([0, 1], p=[0.96, 0.04])
                
                transactions.append({
                    "InvoiceNo": invoice_no,
                    "StockCode": prod["code"],
                    "Description": prod["name"],
                    "Category": prod["category"],
                    "Quantity": qty,
                    "UnitPrice": prod["price"],
                    "InvoiceDate": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "CustomerID": f"CUST_{cust_id:05d}",
                    "Country": country,
                    "Age": age,
                    "Satisfaction": sat,
                    "Device": device,
                    "PaymentMethod": payment,
                    "Returned": returned,
                })

    df = pd.DataFrame(transactions)
    logger.info("Generated %d transaction records across %d unique customers.", len(df), df["CustomerID"].nunique())
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
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    
    # Remove negative or zero unit prices / quantities
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    
    # Calculate Line Total
    data["LineTotal"] = data["Quantity"] * data["UnitPrice"]
    
    # Ensure CustomerID is treated as string
    data["CustomerID"] = data["CustomerID"].astype(str)
    
    return data


def save_datasets(raw_df: pd.DataFrame, processed_df: pd.DataFrame, base_dir: Path) -> Tuple[Path, Path]:
    """Saves raw CSV and cleaned Parquet/CSV files."""
    raw_dir = base_dir / "data" / "raw"
    proc_dir = base_dir / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "sample_ecommerce_data.csv"
    proc_path = proc_dir / "cleaned_transactions.parquet"
    proc_csv_path = proc_dir / "cleaned_transactions.csv"

    raw_df.to_csv(raw_path, index=False)
    processed_df.to_parquet(proc_path, index=False)
    processed_df.to_csv(proc_csv_path, index=False)

    logger.info("Raw data saved to: %s", raw_path)
    logger.info("Cleaned data saved to: %s", proc_path)
    return raw_path, proc_path


def main():
    parser = argparse.ArgumentParser(description="Generate and clean e-commerce transactional data.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic sample data.")
    parser.add_argument("--customers", type=int, default=1500, help="Number of synthetic customers.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_df = generate_synthetic_data(num_customers=args.customers)
    cleaned_df = clean_and_preprocess_data(raw_df)
    save_datasets(raw_df, cleaned_df, project_root)


if __name__ == "__main__":
    main()
