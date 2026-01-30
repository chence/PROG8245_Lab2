"""
Lab2 - Data Collection and Pre-processing
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

path_primary_data = Path("data/primary_transactions_1000.csv")
path_secondary_data = Path("data/secondary_product_catalog.csv")

# Step 1 - Hello, Data!primary_path = Path("data/primary_transactions_1000.csv")
secondary_path = Path("data/secondary_product_catalog.csv")

df_txn_raw = pd.read_csv(path_primary_data)[:500]
df_meta_raw = pd.read_csv(path_secondary_data)

df_txn_raw.head(3)


# Step 2 - Pick the Right Container


# Step 3 - Implement Functions and Data structure
class TransactionProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def total(self) -> float:
        # total gross revenue (price * quantity)
        return float((self.df["price"] * self.df["quantity"]).sum())

    def clean(self) -> dict:
        """Apply cleaning rules and return before/after counts."""
        df = self.df.copy()

        before = {
            "rows": int(len(df)),
            "missing_price": int(df["price"].isna().sum()),
            "missing_quantity": int(df["quantity"].isna().sum()),
            "missing_shipping_city": int(df["shipping_city"].isna().sum()),
            "bad_price_nonpositive": int((pd.to_numeric(df["price"], errors="coerce") <= 0).sum()),
            "bad_quantity_nonpositive": int((pd.to_numeric(df["quantity"], errors="coerce") <= 0).sum()),
            "bad_date_parse": int(pd.to_datetime(df["date"], errors="coerce").isna().sum()),
            "coupon_blank": int(df["coupon_code"].astype(str).str.strip().eq("").sum()),
        }

        # Rule 1) Parse date; drop rows where date can't be parsed
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

        # Rule 2) Coerce price/quantity to numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Rule 3) Fix non-positive or missing price -> fill with median price
        df.loc[df["price"] <= 0, "price"] = np.nan
        df["price"] = df["price"].fillna(df["price"].median())

        # Rule 4) Fix non-positive or missing quantity -> fill with 1
        df.loc[df["quantity"] <= 0, "quantity"] = np.nan
        df["quantity"] = df["quantity"].fillna(1).astype(int)

        # Rule 5) Clean shipping_city -> strip; fill missing/blank with 'Unknown'
        df["shipping_city"] = df["shipping_city"].astype(str).str.strip()
        df.loc[df["shipping_city"].isin(["nan", "None", ""]), "shipping_city"] = "Unknown"

        # Rule 6) Clean coupon_code -> strip + upper; fill missing/blank with 'NONE'
        df["coupon_code"] = df["coupon_code"].astype(str).str.strip().str.upper()
        df.loc[df["coupon_code"].isin(["nan", "None", ""]), "coupon_code"] = "NONE"

        after = {
            "rows": int(len(df)),
            "missing_price": int(df["price"].isna().sum()),
            "missing_quantity": int(df["quantity"].isna().sum()),
            "missing_shipping_city": int(df["shipping_city"].isna().sum()),
            "bad_price_nonpositive": int((df["price"] <= 0).sum()),
            "bad_quantity_nonpositive": int((df["quantity"] <= 0).sum()),
            "bad_date_parse": int(df["date"].isna().sum()),
            "coupon_blank": int(df["coupon_code"].astype(str).str.strip().eq("").sum()),
        }

        self.df = df
        return {"before": before, "after": after}

proc = TransactionProcessor(df_txn_raw)
print(proc.df.head(3))


# Step 4 - Bulk Loaded
# Example: build a lookup dict from product_sku -> product metadata
meta_by_sku = df_meta_raw.set_index("product_sku").to_dict(orient="index")

# Show 2 example entries
sample_keys = list(meta_by_sku.keys())[:2]
print({k: meta_by_sku[k] for k in sample_keys})

# Step 5 - Quick Profiling
price_num = pd.to_numeric(df_txn_raw["price"], errors="coerce")
cities_set = set(df_txn_raw["shipping_city"].astype(str).str.strip())

profiling = {
    "min_price": float(np.nanmin(price_num)),
    "mean_price": float(np.nanmean(price_num)),
    "max_price": float(np.nanmax(price_num)),
    "unique_city_count": int(len(cities_set)),
}
print(profiling)


# Step 6 - Spot the Grime
grime_checks = {
    "unparseable_dates": int(pd.to_datetime(df_txn_raw["date"], errors="coerce").isna().sum()),
    "nonpositive_prices": int((pd.to_numeric(df_txn_raw["price"], errors="coerce") <= 0).sum()),
    "nonpositive_quantities": int((pd.to_numeric(df_txn_raw["quantity"], errors="coerce") <= 0).sum()),
    "missing_or_blank_city": int(df_txn_raw["shipping_city"].isna().sum() + df_txn_raw["shipping_city"].astype(str).str.strip().eq("").sum()),
    "missing_or_blank_coupon": int(df_txn_raw["coupon_code"].isna().sum() + df_txn_raw["coupon_code"].astype(str).str.strip().eq("").sum()),
}
print(grime_checks)

# Step 7- Cleaning Rules
clean_summary = proc.clean()
print(clean_summary)

# Step 8 - Transformations
def parse_discount_percent(code: str) -> int:
    """Transform coupon_code into numeric discount percent."""
    if code is None:
        return 0
    code = str(code).strip().upper()
    if code in ("NONE", ""):
        return 0
    if "FREE" in code:  # FREESHIP etc.
        return 0
    m = re.search(r"(\d+)$", code)  # trailing digits
    return int(m.group(1)) if m else 0

proc.df["discount_percent"] = proc.df["coupon_code"].apply(parse_discount_percent).astype(int)
print(proc.df[["coupon_code", "discount_percent"]].head(10))

# Step 9 – Feature Engineering
# days_since_purchase relative to the most recent purchase date in the dataset
reference_date = proc.df["date"].max()
proc.df["days_since_purchase"] = (reference_date - proc.df["date"]).dt.days.astype(int)

# revenue columns
proc.df["gross_revenue"] = (proc.df["price"] * proc.df["quantity"]).round(2)
proc.df["net_revenue"] = (proc.df["gross_revenue"] * (1 - proc.df["discount_percent"] / 100.0)).round(2)

print(proc.df[["date", "price", "quantity", "discount_percent", "gross_revenue", "net_revenue", "days_since_purchase"]].head(5))

# Step 10 - Mini-Aggregation
revenue_per_city = proc.df.groupby("shipping_city")["net_revenue"].sum().round(2).sort_values(ascending=False)

# Show top 10 cities
print(revenue_per_city.head(10))

# Also demonstrate a dict result (as required option)
revenue_city_dict = revenue_per_city.to_dict()
print(list(revenue_city_dict.items())[:5])


# Step 11 - Serialization Checkpoint
out_json = Path("output/cleaned_transactions.json")
out_csv = Path("output/cleaned_transactions.csv")

out_json.parent.mkdir(parents=True, exist_ok=True)

proc.df.to_json(out_json, orient="records", indent=2, date_format="iso")
proc.df.to_csv(out_csv, index=False)

print(out_json.as_posix(), out_csv.as_posix())


# Data Dictionary
def infer_type(series: pd.Series) -> str:
    t = str(series.dtype)
    if "datetime" in t:
        return "datetime"
    if "int" in t:
        return "int"
    if "float" in t:
        return "float"
    return "string"

dd_rows = []

# Primary fields
for col in proc.df.columns:
    dd_rows.append({
        "Field": col,
        "Type": infer_type(proc.df[col]),
        "Source": "Primary",
        "Description": f"Transaction field '{col}' from the primary transactions file.",
    })

# Secondary fields
for col in df_meta_raw.columns:
    dd_rows.append({
        "Field": col,
        "Type": infer_type(df_meta_raw[col]),
        "Source": "Secondary",
        "Description": f"Metadata field '{col}' from the product catalog.",
    })

data_dictionary = pd.DataFrame(dd_rows)

# Remove duplicate field names (prefer Primary CSV if same name exists)
data_dictionary = (
    data_dictionary
    .sort_values(by=["Field", "Source"])
    .drop_duplicates(subset=["Field"], keep="first")
    .reset_index(drop=True)
)

print(data_dictionary.head(50))


# Concise Analytical Insight
top_city = revenue_per_city.index[0]
top_city_revenue = float(revenue_per_city.iloc[0])

# insight
print(
    "Insight: "
    f"The city with the highest net revenue is {top_city}, "
    f"with a total net revenue of ${top_city_revenue:.2f}."
)