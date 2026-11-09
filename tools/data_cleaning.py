
import pandas as pd

# Load dataset
df = pd.read_csv("data/sales.csv")

print("Original Dataset Shape:", df.shape)

# 1. Clean column names
df.columns = df.columns.str.strip()

# 2. Remove duplicate records
df = df.drop_duplicates()

# 3. Handle missing values
df["region"] = df["region"].fillna("Unknown")
df["sales"] = df["sales"].fillna(df["sales"].median())

# 4. Remove unwanted spaces
df["product"] = df["product"].str.strip().str.title()

# 5. Convert date
df["order_date"] = pd.to_datetime(df["order_date"])

# 6. Check invalid sales
df = df[df["sales"] >= 0]

# 7. Check final missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 8. Check final dataset
print("\nCleaned Dataset Shape:", df.shape)

# 9. Save cleaned dataset
df.to_csv("outputs/cleaned_sales.csv", index=False)

print("\nData Cleaning Completed!")
