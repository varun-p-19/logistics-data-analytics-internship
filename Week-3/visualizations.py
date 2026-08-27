import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_NAME = "cleaned_logistics_data.csv"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"'{FILE_NAME}' was not found. Place the cleaned Week 2 CSV "
        "in the same folder as this Python file."
    )

df = pd.read_csv(FILE_NAME)

print("=" * 60)
print("WEEK 3 - LOGISTICS EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nFirst five rows:")
print(df.head())

# Convert purchase timestamp to datetime
if "order_purchase_timestamp" in df.columns:
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce"
    )

if (
    "delivery_days" not in df.columns
    and "order_purchase_timestamp" in df.columns
    and "order_delivered_customer_date" in df.columns
):
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"],
        errors="coerce"
    )

    df["delivery_days"] = (
        df["order_delivered_customer_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 60 * 60)

if (
    "delay_days" not in df.columns
    and "order_delivered_customer_date" in df.columns
    and "order_estimated_delivery_date" in df.columns
):
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce"
    )

    df["delay_days"] = (
        df["order_delivered_customer_date"]
        - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 60 * 60)

# Create is_late if it does not already exist
if "is_late" not in df.columns and "delay_days" in df.columns:
    df["is_late"] = (df["delay_days"] > 0).astype(int)

numeric_columns = [
    column
    for column in [
        "order_value",
        "freight_value",
        "number_of_items",
        "delivery_days",
        "delay_days"
    ]
    if column in df.columns
]

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

if numeric_columns:
    print(df[numeric_columns].describe())
else:
    print("No expected numerical columns were found.")


if "delivery_days" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["delivery_days"].dropna(),
        bins=40,
        kde=True
    )

    plt.title("Distribution of Delivery Time")
    plt.xlabel("Delivery Time (Days)")
    plt.ylabel("Number of Orders")
    plt.tight_layout()

    plt.savefig(
        "delivery_time_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if "freight_value" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        df["freight_value"].dropna(),
        bins=40,
        kde=True
    )

    plt.title("Distribution of Freight Cost")
    plt.xlabel("Freight Cost")
    plt.ylabel("Number of Orders")
    plt.tight_layout()

    plt.savefig(
        "freight_cost_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if "delivery_days" in df.columns:

    plt.figure(figsize=(9, 4))

    sns.boxplot(
        x=df["delivery_days"].dropna()
    )

    plt.title("Delivery Time Box Plot")
    plt.xlabel("Delivery Time (Days)")
    plt.tight_layout()

    plt.savefig(
        "delivery_time_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if (
    "order_purchase_timestamp" in df.columns
    and "order_id" in df.columns
):

    df["purchase_date"] = df["order_purchase_timestamp"]

    monthly_orders = (
        df.set_index("purchase_date")
        .resample("MS")["order_id"]
        .nunique()
    )

    plt.figure(figsize=(10, 5))

    monthly_orders.plot()

    plt.title("Monthly Order Volume")
    plt.xlabel("Month")
    plt.ylabel("Number of Orders")
    plt.tight_layout()

    plt.savefig(
        "monthly_order_volume.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if (
    "order_purchase_timestamp" in df.columns
    and "delivery_days" in df.columns
):

    monthly_delivery = (
        df.dropna(subset=["delivery_days"])
        .set_index("order_purchase_timestamp")
        .resample("MS")["delivery_days"]
        .mean()
    )

    plt.figure(figsize=(10, 5))

    monthly_delivery.plot()

    plt.title("Average Delivery Time Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average Delivery Time (Days)")
    plt.tight_layout()

    plt.savefig(
        "average_delivery_time_over_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if (
    "customer_state" in df.columns
    and "order_id" in df.columns
):

    state_orders = (
        df.groupby("customer_state")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=state_orders.values,
        y=state_orders.index
    )

    plt.title("Top 10 States by Order Volume")
    plt.xlabel("Number of Orders")
    plt.ylabel("Customer State")
    plt.tight_layout()

    plt.savefig(
        "top_states_by_order_volume.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if (
    "customer_state" in df.columns
    and "is_late" in df.columns
):

    late_rate = (
        df.groupby("customer_state")["is_late"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=late_rate.values * 100,
        y=late_rate.index
    )

    plt.title("States with Higher Late Delivery Rates")
    plt.xlabel("Late Delivery Rate (%)")
    plt.ylabel("Customer State")
    plt.tight_layout()

    plt.savefig(
        "late_delivery_rate_by_state.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if (
    "freight_value" in df.columns
    and "delivery_days" in df.columns
):

    relationship_data = df[
        ["freight_value", "delivery_days"]
    ].dropna()

    # Limit plotted observations only for readability.
    # The original dataframe is not modified.
    if len(relationship_data) > 5000:
        relationship_data = relationship_data.sample(
            5000,
            random_state=42
        )

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=relationship_data,
        x="freight_value",
        y="delivery_days",
        alpha=0.4
    )

    plt.title("Freight Cost vs Delivery Time")
    plt.xlabel("Freight Cost")
    plt.ylabel("Delivery Time (Days)")
    plt.tight_layout()

    plt.savefig(
        "freight_cost_vs_delivery_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if (
    "order_value" in df.columns
    and "freight_value" in df.columns
):

    relationship_data = df[
        ["order_value", "freight_value"]
    ].dropna()

    if len(relationship_data) > 5000:
        relationship_data = relationship_data.sample(
            5000,
            random_state=42
        )

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=relationship_data,
        x="order_value",
        y="freight_value",
        alpha=0.4
    )

    plt.title("Order Value vs Freight Cost")
    plt.xlabel("Order Value")
    plt.ylabel("Freight Cost")
    plt.tight_layout()

    plt.savefig(
        "order_value_vs_freight_cost.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

correlation_columns = [
    column
    for column in [
        "order_value",
        "freight_value",
        "number_of_items",
        "delivery_days",
        "delay_days"
    ]
    if column in df.columns
]

if len(correlation_columns) >= 2:

    correlation_matrix = df[
        correlation_columns
    ].corr()

    plt.figure(figsize=(9, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap of Logistics Variables")
    plt.tight_layout()

    plt.savefig(
        "logistics_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("""
The Week 3 visualization script has completed the following tasks:

1. Loaded the cleaned logistics dataset.
2. Displayed descriptive statistics.
3. Visualized delivery-time distribution.
4. Visualized freight-cost distribution.
5. Created a delivery-time box plot.
6. Analyzed monthly order volume.
7. Analyzed average delivery time over time.
8. Compared order volume across states.
9. Compared late-delivery rates across states.
10. Examined freight cost vs delivery time.
11. Examined order value vs freight cost.
12. Created a correlation heatmap.

The generated PNG files can be added to the Week-3 GitHub folder
if screenshots/visualization outputs are required for submission.
""")
