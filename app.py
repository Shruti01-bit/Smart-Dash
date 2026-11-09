
import streamlit as st
import pandas as pd

from agents.analyst_agent import ask_agent
from tools.data_analysis import load_sales_data

from tools.chart_generator import (
    create_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_histogram,
    create_scatter_plot
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Dash | Data Analyst",
    page_icon="0_0",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101426;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #171D33;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.main-title {
    font-size: 36px;
    font-weight: 700;
    color: #FFFFFF;
}

.subtitle {
    color: #AAB2CC;
    font-size: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background-color: #1B2340;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #303A5C;
}

.metric-label {
    color: #AAB2CC;
    font-size: 14px;
}

.metric-value {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #FFFFFF;
    margin-top: 20px;
    margin-bottom: 15px;
}

.stButton > button {
    background-color: #7657E8;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 9px 18px;
}

.stButton > button:hover {
    background-color: #6242D0;
    color: white;
}

div[data-testid="stChatMessage"] {
    background-color: #1B2340;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. CSV READING FUNCTION
# ==========================================

def read_uploaded_csv(uploaded_file):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1"
    ]

    last_error = None

    for encoding in encodings:

        try:
            uploaded_file.seek(0)

            data = pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

            return data

        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        f"Could not read CSV encoding: {last_error}"
    )


# ==========================================
# 4. SESSION STATE
# ==========================================

if "active_df" not in st.session_state:
    st.session_state.active_df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_error" not in st.session_state:
    st.session_state.upload_error = None


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("Smart Dash")

    st.caption("Your AI-powered Data Analyst")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AI Chat",
            "Data Explorer"
        ]
    )

    st.divider()

    st.subheader(" Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload any CSV file",
        type=["csv"],
        help="Upload a CSV containing category and numeric columns."
    )

    st.session_state.upload_error = None

    if uploaded_file is not None:

        try:

            uploaded_df = read_uploaded_csv(uploaded_file)

            if uploaded_df.empty:

                st.session_state.active_df = None
                st.session_state.original_df = None

                st.session_state.upload_error = (
                    "The uploaded CSV is empty."
                )

            else:

                st.write("**Detected columns:**")

                st.write(uploaded_df.columns.tolist())

                # Find numeric columns
                numeric_columns = uploaded_df.select_dtypes(
                    include="number"
                ).columns.tolist()

                # Find text/category columns
                category_columns = [
                    col for col in uploaded_df.columns
                    if col not in numeric_columns
                ]

                if not numeric_columns:

                    st.session_state.active_df = None
                    st.session_state.original_df = None

                    st.session_state.upload_error = (
                        "No numeric columns found. "
                        "Please upload a CSV with a numeric value column."
                    )

                elif not category_columns:

                    st.session_state.active_df = None
                    st.session_state.original_df = None

                    st.session_state.upload_error = (
                        "No text/category column found. "
                        "Please upload a dataset with a category column."
                    )

                else:

                    # Select category column
                    category_column = st.selectbox(
                        "Select category/product column",
                        options=category_columns,
                        key=f"category_{uploaded_file.name}"
                    )

                    # Select numeric column
                    value_column = st.selectbox(
                        "Select sales/value column",
                        options=numeric_columns,
                        key=f"value_{uploaded_file.name}"
                    )

                    # Create analysis dataframe
                    analysis_df = uploaded_df.copy()

                    # Standard columns for AI tools
                    analysis_df["product"] = (
                        uploaded_df[category_column]
                    )

                    analysis_df["sales"] = pd.to_numeric(
                        uploaded_df[value_column],
                        errors="coerce"
                    )

                    # Remove invalid rows
                    analysis_df = analysis_df.dropna(
                        subset=["product", "sales"]
                    )

                    if analysis_df.empty:

                        st.session_state.active_df = None
                        st.session_state.original_df = None

                        st.session_state.upload_error = (
                            "No valid records found. "
                            "Please check your selected columns."
                        )

                    else:

                        st.session_state.active_df = analysis_df
                        st.session_state.original_df = uploaded_df

                        st.success("CSV uploaded successfully!")

                        st.caption(
                            f"Rows: {len(analysis_df)} | "
                            f"Columns: {len(uploaded_df.columns)}"
                        )

        except Exception as error:

            st.session_state.active_df = None
            st.session_state.original_df = None

            st.session_state.upload_error = (
                f"CSV upload error: {error}"
            )

    else:

        st.session_state.active_df = None
        st.session_state.original_df = None

    if st.session_state.upload_error:

        st.error(st.session_state.upload_error)

    st.divider()

    if st.button(
        "Reset Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


# ==========================================
# 6. LOAD DATASET
# ==========================================

if st.session_state.upload_error:

    st.warning(
        "Please upload a valid CSV dataset to continue."
    )

    st.stop()


if st.session_state.active_df is not None:

    df = st.session_state.active_df.copy()

    display_df = st.session_state.original_df.copy()

else:

    try:

        df = load_sales_data()

        display_df = df.copy()

    except Exception as error:

        st.error(f"Could not load default dataset: {error}")
        st.stop()


# ==========================================
# 7. DATA VALIDATION
# ==========================================

required_columns = {"product", "sales"}

if not required_columns.issubset(df.columns):

    st.error(
        "Default dataset must contain 'product' and 'sales' columns."
    )

    st.stop()


df["sales"] = pd.to_numeric(
    df["sales"],
    errors="coerce"
)

df = df.dropna(
    subset=["sales", "product"]
)

if df.empty:

    st.error("No valid records available for analysis.")

    st.stop()


# ==========================================
# 8. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Smart Dash Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyse your business data with the power of AI.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 9. KPI CALCULATIONS
# ==========================================

total_sales = df["sales"].sum()

average_sales = df["sales"].mean()

maximum_sales = df["sales"].max()

total_records = len(df)


# ==========================================
# 10. KPI CARDS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Sales</div>
            <div class="metric-value">₹{total_sales:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Average Sales</div>
            <div class="metric-value">₹{average_sales:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Highest Sales</div>
            <div class="metric-value">₹{maximum_sales:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{total_records:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 11. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Sales Analytics</div>',
        unsafe_allow_html=True
    )

    # BAR CHART AND HISTOGRAM

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top Products by Sales")

        bar_fig = create_bar_chart(
            df,
            "product",
            "sales"
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Sales Distribution")

        histogram_fig = create_histogram(
            df,
            "sales"
        )

        st.plotly_chart(
            histogram_fig,
            use_container_width=True
        )

    # PIE CHART

    st.subheader("Product Sales Contribution")

    pie_fig = create_pie_chart(
        df,
        "product",
        "sales"
    )

    st.plotly_chart(
        pie_fig,
        use_container_width=True
    )

    # CUSTOM BAR CHART

    st.divider()

    st.subheader("Explore Your Data")

    original_columns = display_df.columns.tolist()

    numeric_columns = display_df.select_dtypes(
        include="number"
    ).columns.tolist()

    category_columns = [
        col for col in original_columns
        if col not in numeric_columns
    ]

    if category_columns and numeric_columns:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            selected_category = st.selectbox(
                "Choose category column",
                category_columns,
                key="custom_category"
            )

        with chart_col2:

            selected_value = st.selectbox(
                "Choose numeric column",
                numeric_columns,
                key="custom_value"
            )

        custom_fig = create_bar_chart(
            display_df,
            selected_category,
            selected_value
        )

        st.plotly_chart(
            custom_fig,
            use_container_width=True
        )

    else:

        st.info(
            "Your dataset needs at least one text/category "
            "column and one numeric column."
        )

    # LINE CHART

    st.divider()

    st.subheader("Time-Based Analysis")

    if numeric_columns:

        line_col1, line_col2 = st.columns(2)

        with line_col1:

            date_column = st.selectbox(
                "Select date column",
                ["None"] + original_columns,
                key="line_date"
            )

        with line_col2:

            line_value = st.selectbox(
                "Select numeric column for line chart",
                numeric_columns,
                key="line_value"
            )

        if date_column != "None":

            try:

                line_fig = create_line_chart(
                    display_df,
                    date_column,
                    line_value
                )

                st.plotly_chart(
                    line_fig,
                    use_container_width=True
                )

            except Exception as error:

                st.warning(
                    f"Could not create line chart: {error}"
                )

        else:

            st.info("Select a date column to create a line chart.")

    else:

        st.info("No numeric columns available for line chart.")

    # SCATTER PLOT

    st.divider()

    st.subheader("Relationship Analysis")

    if len(numeric_columns) >= 2:

        scatter_col1, scatter_col2 = st.columns(2)

        with scatter_col1:

            x_column = st.selectbox(
                "Select X-axis",
                numeric_columns,
                key="scatter_x"
            )

        with scatter_col2:

            y_column = st.selectbox(
                "Select Y-axis",
                numeric_columns,
                index=1,
                key="scatter_y"
            )

        scatter_fig = create_scatter_plot(
            display_df,
            x_column,
            y_column
        )

        st.plotly_chart(
            scatter_fig,
            use_container_width=True
        )

    else:

        st.info(
            "Upload a dataset with at least two numeric columns "
            "to create a scatter plot."
        )

    # TOP PRODUCTS TABLE

    st.divider()

    st.subheader("Top Products")

    product_sales = (
        df.groupby("product")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        product_sales,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 12. AI CHAT PAGE
# ==========================================

elif page == "AI Chat":

    st.markdown(
        '<div class="section-title">'
        'Chat with your AI Analyst'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Ask questions about your selected dataset."
    )

    # DISPLAY CHAT HISTORY

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # CHAT INPUT

    question = st.chat_input(
        "Ask something about your data..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("AI is analysing your data..."):

                try:

                    answer = ask_agent(
                        question,
                        df=df
                    )

                    if answer is None:

                        answer = "I couldn't generate an answer."

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as error:

                    error_message = f"Analysis error: {error}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


# ==========================================
# 13. DATA EXPLORER PAGE
# ==========================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="section-title">'
        'Dataset Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        display_df.head(20),
        use_container_width=True
    )

    # DATASET INFORMATION

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Rows:**", display_df.shape[0])

        st.write("**Columns:**", display_df.shape[1])

        st.write("**Column Names:**")

        st.write(display_df.columns.tolist())

    with col2:

        st.write("**Data Types:**")

        st.write(
            display_df.dtypes.astype(str).to_dict()
        )

    # MISSING VALUES

    st.subheader("Missing Values")

    missing_values = display_df.isnull().sum()

    missing_df = missing_values.rename(
        "Missing Count"
    ).reset_index()

    missing_df.columns = [
        "Column",
        "Missing Count"
    ]

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    # STATISTICAL SUMMARY

    st.subheader("Statistical Summary")

    st.dataframe(
        display_df.describe(include="all").T,
        use_container_width=True
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "InsightAI | Built with Python, Pandas, "
    "Plotly, Streamlit and Groq AI"
)
