import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

DATA_PATH = "data"

orders = pd.read_csv(
    os.path.join(DATA_PATH, "olist_orders_dataset.csv")
)

items = pd.read_csv(
    os.path.join(DATA_PATH, "olist_order_items_dataset.csv")
)

customers = pd.read_csv(
    os.path.join(DATA_PATH, "olist_customers_dataset.csv")
)

sellers = pd.read_csv(
    os.path.join(DATA_PATH, "olist_sellers_dataset.csv")
)

print("Orders:", orders.shape)
print("Items:", items.shape)
print("Customers:", customers.shape)
print("Sellers:", sellers.shape)

print("\n--- Orders Information ---")
print(orders.info())

print("\n--- Missing Values ---")
print(orders.isnull().sum())

print("\n--- Order Status ---")
print(orders["order_status"].value_counts())

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


order_values = (
    items.groupby("order_id", as_index=False)
    .agg(
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        number_of_items=("order_item_id", "count"),
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
    / df["order_value"].replace(0, pd.NA)
)

df["purchase_month"] = (
    df["order_purchase_timestamp"].dt.month
)

delivered_df = df[
    (df["order_status"] == "delivered")
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
].copy()

print("\nEligible delivered orders:", len(delivered_df))

on_time_delivery_rate = (
    1 - delivered_df["is_late"].mean()
) * 100

late_delivery_rate = (
    delivered_df["is_late"].mean()
) * 100

average_delivery_days = (
    delivered_df["delivery_days"].mean()
)

average_freight_cost = (
    delivered_df["freight_value"].mean()
)

freight_to_order_value_ratio = (
    delivered_df["freight_ratio"].mean()
)

print("\n========== LOGISTICS KPI SUMMARY ==========")
print(f"On-Time Delivery Rate: {on_time_delivery_rate:.2f}%")
print(f"Late Delivery Rate: {late_delivery_rate:.2f}%")
print(f"Average Delivery Lead Time: {average_delivery_days:.2f} days")
print(f"Average Freight Cost: {average_freight_cost:.2f}")
print(
    "Average Freight-to-Order Value Ratio: "
    f"{freight_to_order_value_ratio:.4f}"
)

print("\n--- Delivery Time Statistics ---")
print(delivered_df["delivery_days"].describe())

print("\n--- Late Delivery Rate by State ---")

late_by_state = (
    delivered_df.groupby("customer_state")["is_late"]
    .mean()
    .sort_values(ascending=False)
)

print(late_by_state.head(10))
plt.figure(figsize=(8, 5))
plt.hist(
    delivered_df["delivery_days"].dropna(),
    bins=40
)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
late_by_state.head(10).plot(kind="bar")
plt.title("Top States by Late Delivery Rate")
plt.xlabel("State")
plt.ylabel("Late Delivery Rate")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(
    delivered_df["freight_value"].dropna(),
    bins=40
)
plt.title("Distribution of Freight Cost")
plt.xlabel("Freight Cost")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

model_data = delivered_df[
    [
        "freight_value",
        "order_value",
        "number_of_items",
        "customer_state",
        "purchase_month",
        "delivery_days",
    ]
].dropna()

X = model_data[
    [
        "freight_value",
        "order_value",
        "number_of_items",
        "customer_state",
        "purchase_month",
    ]
]

y = model_data["delivery_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

numeric_features = [
    "freight_value",
    "order_value",
    "number_of_items",
    "purchase_month",
]

categorical_features = [
    "customer_state"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

regression_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

regression_model.fit(X_train, y_train)

predictions = regression_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print("\n========== REGRESSION RESULTS ==========")
print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")

seller_orders = items.merge(
    orders[
        [
            "order_id",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "order_purchase_timestamp",
            "order_status",
        ]
    ],
    on="order_id",
    how="left"
)

seller_orders["delivery_days"] = (
    seller_orders["order_delivered_customer_date"]
    - seller_orders["order_purchase_timestamp"]
).dt.total_seconds() / 86400

seller_orders["delay_days"] = (
    seller_orders["order_delivered_customer_date"]
    - seller_orders["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

seller_orders["is_late"] = (
    seller_orders["delay_days"] > 0
).astype(int)

seller_orders = seller_orders[
    (seller_orders["order_status"] == "delivered")
    & seller_orders["delivery_days"].notna()
].copy()


seller_features = (
    seller_orders.groupby("seller_id")
    .agg(
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("is_late", "mean"),
        avg_freight=("freight_value", "mean"),
        order_count=("order_id", "nunique"),
    )
    .dropna()
)

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    seller_features
)

# Four clusters are used as an initial baseline.
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

seller_features["cluster"] = (
    kmeans.fit_predict(scaled_features)
)

print("\n========== SELLER CLUSTERS ==========")
print(
    seller_features.groupby("cluster").mean(
        numeric_only=True
    )
)


print("\n========== INITIAL BUSINESS INSIGHTS ==========")

print(
    "1. The KPI results provide a baseline for delivery performance."
)

print(
    "2. States with higher late-delivery rates can be investigated "
    "for geographic or operational causes."
)

print(
    "3. The regression model provides a starting point for estimating "
    "delivery time."
)

print(
    "4. Seller clustering can help identify groups with similar "
    "delivery and freight behaviour."
)

print(
    "5. A future optimization model can use these insights for "
    "route planning, capacity allocation, and cost reduction."
)



