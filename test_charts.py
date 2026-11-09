from tools.data_analysis import load_sales_data
from tools.chart_generator import create_bar_chart

df = load_sales_data()

fig = create_bar_chart(
    df,
    "product",
    "sales"
)

fig.show()