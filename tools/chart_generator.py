
import pandas as pd
import plotly.express as px


# ==========================================
# COMMON DATA VALIDATION
# ==========================================

def validate_dataframe(df, required_columns):

    if df is None:
        raise ValueError("Dataset is None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return df.copy()


# ==========================================
# 1. BAR CHART
# ==========================================

def create_bar_chart(df, category_column, value_column):

    if category_column == value_column:
        raise ValueError(
            "Please select different category and numeric columns."
        )

    data = validate_dataframe(
        df,
        [category_column, value_column]
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[category_column, value_column]
    )

    if data.empty:
        raise ValueError("No valid data available for bar chart.")

    chart_data = (
        data.groupby(category_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
        .head(10)
    )

    fig = px.bar(
        chart_data,
        x=category_column,
        y=value_column,
        title=f"{value_column} by {category_column}",
        color=value_column,
        template="plotly_dark"
    )

    return fig


# ==========================================
# 2. LINE CHART
# ==========================================

def create_line_chart(df, date_column, value_column):

    data = validate_dataframe(
        df,
        [date_column, value_column]
    )

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column, value_column]
    )

    if data.empty:
        raise ValueError("No valid data available for line chart.")

    chart_data = (
        data.groupby(date_column, as_index=False)[value_column]
        .sum()
        .sort_values(date_column)
    )

    fig = px.line(
        chart_data,
        x=date_column,
        y=value_column,
        title=f"{value_column} Trend",
        markers=True,
        template="plotly_dark"
    )

    return fig


# ==========================================
# 3. PIE CHART
# ==========================================

def create_pie_chart(df, category_column, value_column):

    data = validate_dataframe(
        df,
        [category_column, value_column]
    )

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[category_column, value_column]
    )

    if data.empty:
        raise ValueError("No valid data available for pie chart.")

    chart_data = (
        data.groupby(category_column, as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
        .head(10)
    )

    if chart_data[value_column].sum() <= 0:
        raise ValueError(
            "Pie chart requires a positive total value."
        )

    fig = px.pie(
        chart_data,
        names=category_column,
        values=value_column,
        title=f"{value_column} Contribution",
        template="plotly_dark"
    )

    return fig


# ==========================================
# 4. HISTOGRAM
# ==========================================

def create_histogram(df, column):

    data = validate_dataframe(df, [column])

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    data = data.dropna(subset=[column])

    if data.empty:
        raise ValueError(
            "No valid numeric data available for histogram."
        )

    fig = px.histogram(
        data,
        x=column,
        title=f"Distribution of {column}",
        template="plotly_dark"
    )

    return fig


# ==========================================
# 5. SCATTER PLOT
# ==========================================

def create_scatter_plot(df, x_column, y_column):

    data = validate_dataframe(
        df,
        [x_column, y_column]
    )

    data[x_column] = pd.to_numeric(
        data[x_column],
        errors="coerce"
    )

    data[y_column] = pd.to_numeric(
        data[y_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[x_column, y_column]
    )

    if data.empty:
        raise ValueError(
            "No valid data available for scatter plot."
        )

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
        template="plotly_dark"
    )

    return fig
