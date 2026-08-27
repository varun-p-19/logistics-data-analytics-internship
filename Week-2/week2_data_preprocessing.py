import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_PATH = "data"

orders_file = os.path.join(DATA_PATH, "olist_orders_dataset.csv")
items_file = os.path.join(DATA_PATH, "olist_order_items_dataset.csv")
customers_file = os.path.join(DATA_PATH, "olist_customers_dataset.csv")
sellers_file = os.path.join(DATA_PATH, "olist_sellers_dataset.csv")
products_file = os.path.join(DATA_PATH, "olist_products_dataset.csv")

orders = pd.read_csv(orders_file)
items = pd.read_csv(items_file)
customers = pd.read_csv(customers_file)
sellers = pd.read_csv(sellers_file)
products = pd.read_csv(products_file)

print("========== DATASET SHAPES ==========")
print("Orders:", orders.shape)
print("Items:", items.shape)
print("Customers:", customers.shape)
print("Sellers:", sellers.shape)
print("Products:", products.shape)

print("\n========== ORDERS DATA TYPES ==========")
print(orders.dtypes)

print("\n========== ORDERS SUMMARY ==========")
print(orders.describe(include="all"))

print("\n========== ORDER STATUS ==========")
print(orders["order_status"].value_counts(dropna=False))

print("\n========== MISSING VALUES ==========")

missing_count = orders.isnull().sum()
missing_percentage = (
    orders.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percentage": missing_percentage
}).sort_values(
    "missing_percentage",
    ascending=False
)

print(missing_report)

print("\n========== DUPLICATE CHECK ==========")

duplicate_count = orders.duplicated().sum()

print("Duplicate rows:", duplicate_count)

orders = orders.drop_duplicates()

print(
    "Rows after duplicate removal:",
    len(orders)
)

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )

print("\n========== UPDATED DATE TYPES ==========")
print(orders[date_columns].dtypes)

print("\n========== CATEGORICAL VALUES ==========")

print(
    orders["order_status"]
    .value_counts(dropna=False)
)

print(
    "\nCustomer states:"
)

print(
    customers["customer_state"]
    .value_counts(dropna=False)
    .head(30)
)

for column in ["order_status"]:
    orders[column] = (
        orders[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )


order_values = (
    items.groupby("order_id", as_index=False)
    .agg(
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        number_of_items=("order_item_id", "count")
    )
)

df = orders.merge(
    order_values,
    on="order_id",
    how="left"
)

df = df.merge(
    customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ]
    ],
    on="customer_id",
    how="left"
)


numeric_columns = [
    "order_value",
    "freight_value",
    "number_of_items"
]

print("\n========== NUMERICAL MISSING VALUES BEFORE IMPUTATION ==========")
print(df[numeric_columns].isnull().sum())

imputer = SimpleImputer(strategy="median")

df[numeric_columns] = imputer.fit_transform(
    df[numeric_columns]
)

print("\n========== NUMERICAL MISSING VALUES AFTER IMPUTATION ==========")
print(df[numeric_columns].isnull().sum())

df["delivery_days"] = (
    df["order_delivered_customer_date"]
    - df["order_purchase_timestamp"]
).dt.total_seconds() / 86400

df["delay_days"] = (
    df["order_delivered_customer_date"]
    - df["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

df["is_late"] = (
    df["delay_days"] > 0
).astype(int)

df["freight_ratio"] = (
    df["freight_value"]
    / df["order_value"].replace(0, np.nan)
)

df["purchase_month"] = (
    df["order_purchase_timestamp"].dt.month
)

df["purchase_year"] = (
    df["order_purchase_timestamp"].dt.year
)

delivered_df = df[
    (df["order_status"] == "delivered")
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
].copy()

print("\nValid delivered orders:", len(delivered_df))

print("\n========== INVALID VALUE CHECK ==========")

negative_freight = (
    delivered_df["freight_value"] < 0
).sum()

negative_price = (
    delivered_df["order_value"] < 0
).sum()

negative_delivery_time = (
    delivered_df["delivery_days"] < 0
).sum()

print("Negative freight values:", negative_freight)
print("Negative order values:", negative_price)
print("Negative delivery times:", negative_delivery_time)

def find_iqr_outliers(dataframe, column):
    """Return IQR limits and rows outside those limits."""
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = dataframe[
        (dataframe[column] < lower_limit)
        | (dataframe[column] > upper_limit)
    ]

    return lower_limit, upper_limit, outliers


for column in ["freight_value", "order_value", "delivery_days"]:
    lower, upper, outliers = find_iqr_outliers(
        delivered_df,
        column
    )

    print(f"\n{column}")
    print("Lower limit:", round(lower, 2))
    print("Upper limit:", round(upper, 2))
    print("Potential outliers:", len(outliers))

def cap_iqr(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    dataframe[column] = dataframe[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )

    return dataframe


for column in [
    "freight_value",
    "order_value"
]:
    delivered_df = cap_iqr(
        delivered_df,
        column
    )

minmax_scaler = MinMaxScaler()

delivered_df[
    [
        "order_value_normalized",
        "freight_value_normalized"
    ]
] = minmax_scaler.fit_transform(
    delivered_df[
        [
            "order_value",
            "freight_value"
        ]
    ]
)

print("\n========== MIN-MAX NORMALIZATION ==========")
print(
    delivered_df[
        [
            "order_value_normalized",
            "freight_value_normalized"
        ]
    ].head()
)


standard_scaler = StandardScaler()

delivered_df[
    [
        "delivery_days_standardized",
        "freight_standardized"
    ]
] = standard_scaler.fit_transform(
    delivered_df[
        [
            "delivery_days",
            "freight_value"
        ]
    ]
)

print("\n========== STANDARDIZATION ==========")
print(
    delivered_df[
        [
            "delivery_days_standardized",
            "freight_standardized"
        ]
    ].head()
)

print("\n========== FINAL VALIDATION ==========")

print(
    "Final rows:",
    delivered_df.shape[0]
)

print(
    "Final columns:",
    delivered_df.shape[1]
)

print(
    "\nRemaining missing values:"
)

print(
    delivered_df.isnull().sum()
    .sort_values(ascending=False)
    .head(15)
)

print(
    "\nDuplicate rows:",
    delivered_df.duplicated().sum()
)

print(
    "\nFinal numerical summary:"
)

print(
    delivered_df[
        [
            "order_value",
            "freight_value",
            "delivery_days",
            "delay_days",
            "freight_ratio"
        ]
    ].describe()
)

output_path = "cleaned_logistics_data.csv"

delivered_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nCleaned dataset saved to: {output_path}"
)

print("\n========== DATA QUALITY SUMMARY ==========")

print("The preprocessing pipeline completed the following:")
print("1. Loaded the Olist logistics data.")
print("2. Profiled data types and missing values.")
print("3. Removed exact duplicate records.")
print("4. Converted timestamp columns to datetime.")
print("5. Checked categorical values.")
print("6. Imputed selected numerical missing values.")
print("7. Created delivery and delay features.")
print("8. Checked invalid numerical values.")
print("9. Detected potential outliers using IQR.")
print("10. Capped selected extreme numerical values.")
print("11. Applied Min-Max normalization.")
print("12. Applied StandardScaler standardization.")
print("13. Performed final validation.")
print("14. Saved the cleaned dataset.")