
import pandas as pd
import plotly.express as px


# 1. Bar Chart


def create_bar_chart(df, category_column, value_column):

    if category_column == value_column:
        raise ValueError(
            "Please select different category and numeric columns."
        )

    data = df.copy()

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[category_column, value_column]
    )

    chart_data = (
        data.groupby(category_column)[value_column]
        .sum()
        .reset_index()
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


# 2. Line Chart

def create_line_chart(df, date_column, value_column):

    data = df.copy()

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

    chart_data = (
        data.groupby(date_column)[value_column]
        .sum()
        .reset_index()
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


# 3. Pie Chart

def create_pie_chart(df, category_column, value_column):

    data = df.copy()

    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[category_column, value_column]
    )

    chart_data = (
        data.groupby(category_column)[value_column]
        .sum()
        .reset_index()
        .sort_values(value_column, ascending=False)
        .head(10)
    )

    fig = px.pie(
        chart_data,
        names=category_column,
        values=value_column,
        title=f"{value_column} Contribution",
        template="plotly_dark"
    )

    return fig


# 4. Histogram

def create_histogram(df, column):

    data = df.copy()

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    data = data.dropna(subset=[column])

    fig = px.histogram(
        data,
        x=column,
        title=f"Distribution of {column}",
        template="plotly_dark"
    )

    return fig


# 5. Scatter Plot

def create_scatter_plot(df, x_column, y_column):

    data = df.copy()

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

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
        template="plotly_dark"
    )

    return fig
