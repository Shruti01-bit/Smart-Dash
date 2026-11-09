
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sales.csv"
)


# 1. Load dataset

def load_sales_data(df=None):

    if df is not None:
        return df.copy()

    return pd.read_csv(DATA_PATH)


# 2. Dataset overview

def get_dataset_overview(df=None):

    data = load_sales_data(df)

    return {
        "total_rows": int(data.shape[0]),
        "total_columns": int(data.shape[1]),
        "columns": data.columns.tolist(),
        "data_types": data.dtypes.astype(str).to_dict()
    }


# 3. Total sales

def get_total_sales(df=None):

    data = load_sales_data(df)

    return float(data["sales"].sum())


# 4. Average sales

def get_average_sales(df=None):

    data = load_sales_data(df)

    return float(data["sales"].mean())


# 5. Maximum sales

def get_maximum_sales(df=None):

    data = load_sales_data(df)

    return float(data["sales"].max())


# 6. Product-wise sales

def get_product_sales(df=None):

    data = load_sales_data(df)

    result = (
        data.groupby("product")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        str(product): float(sales)
        for product, sales in result.items()
    }


# 7. Missing values

def get_missing_values(df=None):

    data = load_sales_data(df)

    return {
        str(column): int(count)
        for column, count in data.isnull().sum().items()
    }


# 8. Run analysis directly

if __name__ == "__main__":

    print("DATASET OVERVIEW")
    print(get_dataset_overview())

    print("\nTOTAL SALES")
    print(get_total_sales())

    print("\nAVERAGE SALES")
    print(get_average_sales())

    print("\nMAXIMUM SALES")
    print(get_maximum_sales())

    print("\nPRODUCT-WISE SALES")
    print(get_product_sales())

    print("\nMISSING VALUES")
    print(get_missing_values())
