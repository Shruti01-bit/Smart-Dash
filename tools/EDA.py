
import pandas as pd

# Load dataset
df = pd.read_csv("data/sales.csv")

# Convert date
df["order_date"] = pd.to_datetime(df["order_date"])

print("========== SALES DATA ANALYSIS ==========")

# 1. Product-wise sales
print("\n1. PRODUCT-WISE SALES")
print(
    df.groupby("product")["sales"]
    .sum()
    .sort_values(ascending=False)
)

# 2. Region-wise sales
print("\n2. REGION-WISE SALES")
print(
    df.groupby("region")["sales"]
    .sum()
    .sort_values(ascending=False)
)

# 3. Monthly sales
print("\n3. MONTHLY SALES")
monthly_sales = (
    df.groupby(df["order_date"].dt.to_period("M"))["sales"]
    .sum()
    .sort_index()
)
print(monthly_sales)

# 4. Product order count
print("\n4. PRODUCT ORDER COUNT")
print(df["product"].value_counts())

# 5. Product summary
print("\n5. PRODUCT SUMMARY")
print(
    df.groupby("product").agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        total_profit=("profit", "sum"),
        total_quantity=("quantity", "sum")
    )
)

# 6. Region and product pivot table
print("\n6. PIVOT TABLE")
print(
    pd.pivot_table(
        df,
        values="sales",
        index="region",
        columns="product",
        aggfunc="sum",
        fill_value=0
    )
)

# 7. Average order value
aov = df["sales"].sum() / df["order_id"].nunique()
print("\n7. AVERAGE ORDER VALUE:", aov)

# 8. Business insights
print("\n8. BUSINESS INSIGHTS")

best_product = df.groupby("product")["sales"].sum().idxmax()
best_region = df.groupby("region")["sales"].sum().idxmax()
best_profit_product = df.groupby("product")["profit"].sum().idxmax()

print("Best Selling Product:", best_product)
print("Best Performing Region:", best_region)
print("Most Profitable Product:", best_profit_product)

print("\n========== ANALYSIS COMPLETED ==========")
