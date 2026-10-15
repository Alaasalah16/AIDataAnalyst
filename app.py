"""
AI Data Analyst - Streamlit app.

Run with:  streamlit run app.py

The page goes from top to bottom:
   1. Upload                 2. Overview              3. Data quality
   4. Cleaning               5. Exploratory analysis  6. KPIs
   7. Charts                 8. Dates                 9. Categories
  10. Dataset profile       11. AI insights          12. Ask a question
"""

import plotly.express as px
import streamlit as st

from src import analysis, ai
from src.data_loader import ENCODINGS, clean_data, load_file
from src.questions import answer_question


st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analyst")
st.write("Upload your dataset and let the application perform end-to-end data analysis.")


# Cache = do the slow work once, not on every click
@st.cache_data
def read_file(file_bytes, file_name, encoding):
    return load_file(file_bytes, file_name, encoding)


@st.cache_data
def clean(df):
    return clean_data(df)


def show_chart(figure):
    st.plotly_chart(figure, use_container_width=True)


def show_table(table, **options):
    st.dataframe(table, use_container_width=True, **options)


# =====================================================================
# 1. UPLOAD
# =====================================================================

st.divider()
st.header("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset to get started.")
    st.stop()

encoding = "Auto"
if uploaded_file.name.lower().endswith(".csv"):
    encoding = st.selectbox("File encoding", list(ENCODINGS))

# When a different file is uploaded, forget everything from the old one
file_id = f"{uploaded_file.name}-{uploaded_file.size}-{encoding}"
if st.session_state.get("file_id") != file_id:
    st.session_state.clear()
    st.session_state["file_id"] = file_id

try:
    df, warning = read_file(uploaded_file.getvalue(), uploaded_file.name, encoding)
except Exception as error:
    st.error(f"Unable to read the file: {error}")
    st.stop()

st.success(f"Successfully loaded: {uploaded_file.name}")
if warning:
    st.warning(warning)


# =====================================================================
# 2. OVERVIEW
# =====================================================================

st.divider()
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Missing Values", f"{df.isna().sum().sum():,}")
col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")


# =====================================================================
# 3. DATA QUALITY
# =====================================================================

st.divider()
st.header("🛡️ Data Quality Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Missing Values", f"{df.isna().sum().sum():,}")
col2.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
col3.metric("Numeric Columns", len(analysis.get_number_columns(df)))


# =====================================================================
# 4. CLEANING
# =====================================================================

st.divider()
st.header("🧹 Automatic Data Cleaning")

if st.button("🧹 Clean Dataset", type="primary", use_container_width=True):
    st.session_state["clean_df"], st.session_state["report"] = clean(df)

if "clean_df" not in st.session_state:
    st.info("Click **Clean Dataset** to continue the analysis.")
    st.stop()

data = st.session_state["clean_df"]
report = st.session_state["report"]

st.success("✅ Dataset cleaned successfully!")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Original Rows", f"{report['rows_before']:,}")
col2.metric("Cleaned Rows", f"{report['rows_after']:,}")
col3.metric("Rows Removed", f"{report['rows_before'] - report['rows_after']:,}")
col4.metric("Columns Removed", report["columns_before"] - report["columns_after"])

if report["number_columns"]:
    st.info("Converted to numbers: " + ", ".join(report["number_columns"]))
if report["date_columns"]:
    st.info("Converted to dates: " + ", ".join(report["date_columns"]))

st.subheader("📥 Download Cleaned Dataset")
st.download_button(
    "📥 Download Cleaned CSV",
    data.to_csv(index=False).encode("utf-8-sig"),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
    use_container_width=True,
)

# Column groups used by the rest of the page
number_columns = analysis.get_number_columns(data)
metric_columns = analysis.get_metric_columns(data)
date_columns = analysis.get_date_columns(data)
text_columns = analysis.get_text_columns(data)


# =====================================================================
# 5. EXPLORATORY DATA ANALYSIS
# =====================================================================

st.divider()
st.header("📈 Exploratory Data Analysis")

if number_columns:
    st.subheader("📊 Numerical Statistics")
    show_table(analysis.numbers_summary(data))
else:
    st.info("No numerical columns detected.")

if len(number_columns) >= 2:
    st.subheader("🔗 Correlation Analysis")
    show_table(data[number_columns].corr().round(2))


# =====================================================================
# 6. KPIs
# =====================================================================

st.divider()
st.header("🎯 Automatic KPI Detection")

kpis = analysis.kpi_table(data)

if kpis.empty:
    st.info("No numerical metrics detected.")
else:
    show_table(kpis, hide_index=True)
    st.caption("Columns like IDs, codes and years are skipped because their totals mean nothing.")

    st.subheader("⭐ Key Metrics")
    first_four = kpis.head(4)
    for box, (_, row) in zip(st.columns(len(first_four)), first_four.iterrows()):
        box.metric(row["Metric"], f"{row['Total']:,.2f}")


# =====================================================================
# 7. CHARTS
# =====================================================================

st.divider()
st.header("📊 Automatic Visualizations")

if number_columns:
    st.subheader("📈 Numerical Distribution")
    column = st.selectbox("Select a numerical column", number_columns)
    show_chart(px.histogram(data, x=column, marginal="box", title=f"Distribution of {column}"))

if text_columns:
    st.subheader("📊 Basic Category Analysis")
    column = st.selectbox("Select a categorical column", text_columns)
    counts = data[column].value_counts().head(15).reset_index()
    counts.columns = [column, "Count"]
    show_chart(px.bar(counts, x=column, y="Count", title=f"Top Categories — {column}"))

if len(number_columns) >= 2:
    st.subheader("🔵 Relationship Analysis")
    x_column = st.selectbox("X-axis", number_columns)
    y_column = st.selectbox("Y-axis", [c for c in number_columns if c != x_column])
    show_chart(px.scatter(data, x=x_column, y=y_column, title=f"{x_column} vs {y_column}"))


# =====================================================================
# 8. DATES
# =====================================================================

st.divider()
st.header("📅 Automatic Date & Time Analysis")

if not date_columns:
    st.info("No date columns were automatically detected.")
else:
    st.success(f"Detected {len(date_columns)} date column(s).")
    date_column = st.selectbox("Select Date Column", date_columns)
    summary = analysis.date_summary(data, date_column)

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Earliest Date", f"{summary['earliest']:%Y-%m-%d}")
        col2.metric("Latest Date", f"{summary['latest']:%Y-%m-%d}")
        col3.metric("Date Range", f"{summary['days']:,} days")
        col4.metric("Valid Date Records", f"{summary['records']:,}")

        if metric_columns:
            st.subheader("📈 Monthly Trend")
            metric = st.selectbox("Select Metric", metric_columns)
            trend = analysis.monthly_trend(data, date_column, metric)
            show_chart(px.line(trend, x="Month", y=metric, markers=True, title=f"Monthly {metric}"))


# =====================================================================
# 9. CATEGORIES
# =====================================================================

st.divider()
st.header("🏷️ Automatic Categorical Analysis")

if not text_columns:
    st.info("No categorical columns were detected.")
else:
    st.success(f"Detected {len(text_columns)} categorical column(s).")
    category = st.selectbox("Select Categorical Column", text_columns)

    st.subheader("📊 Category Distribution")
    counts = data[category].value_counts().head(15).reset_index()
    counts.columns = [category, "Count"]
    show_chart(px.bar(counts, x=category, y="Count", title=f"Top Categories — {category}"))
    show_table(counts, hide_index=True)

    if metric_columns:
        st.subheader("📈 Category Performance")
        metric = st.selectbox("Select Numeric Metric", metric_columns)
        table = analysis.category_summary(data, category, metric)

        if not table.empty:
            show_table(table, hide_index=True)

            best, worst = table.iloc[0], table.iloc[-1]
            st.success(f"🏆 Top {category}: {best[category]} with total {metric} of {best['Total']:,.2f}")
            st.info(f"📉 Lowest {category}: {worst[category]} with total {metric} of {worst['Total']:,.2f}")

            show_chart(px.bar(table.head(15), x=category, y="Total", title=f"{metric} by {category}"))


# =====================================================================
# 10. DATASET PROFILE
# =====================================================================

st.divider()
st.header("🔍 Advanced Dataset Profile")
st.success("✅ Dataset profile generated successfully.")

with st.expander("🔎 View Dataset Profile"):
    st.json(analysis.profile_dataset(data))

# Short text description of the data - this is all the AI sees
description = analysis.describe_for_ai(data)


# =====================================================================
# 11. AI BUSINESS INSIGHTS
# =====================================================================

st.divider()
st.header("💡 Automatic Business Insights")
st.write("Let the AI identify important patterns and potential business insights.")

if st.button("💡 Generate Business Insights", use_container_width=True):
    with st.spinner("AI is analyzing the dataset..."):
        st.session_state["insights"] = ai.business_insights(description)

if "insights" in st.session_state:
    st.success("Business insights generated successfully.")
    st.markdown(st.session_state["insights"])


# =====================================================================
# 12. ASK A QUESTION
# =====================================================================

st.divider()
st.header("🤖 AI Data Analyst")
st.write("Ask questions about the analyzed dataset.")

question = st.text_area(
    "Ask the AI Analyst",
    placeholder=(
        "Examples:\n"
        "What is the total Sales?\n"
        "What is the average Profit?\n"
        "What are the biggest problems in this dataset?\n"
        "Give me 5 important business insights.\n"
        "What additional analysis should I perform?"
    ),
)

if st.button("🤖 Analyze", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Try 1: simple question -> answer directly with pandas
        answer = answer_question(data, question)

        # Try 2: the AI chooses a calculation, pandas does the math
        if answer is None:
            with st.spinner("Analyzing your question..."):
                answer = ai.plan_and_calculate(data, question, number_columns, text_columns)

        if answer is not None:
            text, table = answer
            st.subheader("📊 Data Result")
            st.success(text)
            if table is not None:
                show_table(table)

        # Try 3: open question -> the AI answers from the description
        else:
            with st.spinner("AI is analyzing the dataset..."):
                st.subheader("🤖 AI Response")
                st.markdown(ai.free_answer(description, question))


st.divider()
st.subheader("📋 Cleaned Dataset Preview")
show_table(data.head(100))
