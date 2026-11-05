import hashlib
from io import StringIO

import pandas as pd
import streamlit as st

from utils.config import APP_NAME, APP_ICON
from agents.cleaning import show_cleaning
from agents.eda import show_eda
from agents.visualization import show_visualization
from agents.ml_agent import show_ml
from agents.insight_agent import show_ai_insights
from agents.report_agent import show_report
from agents.chat_with_data import show_chat

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")

for key, value in {
    "original_df": None,
    "cleaned_df": None,
    "uploaded_file_id": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def read_uploaded_file(uploaded_file):
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if extension == "csv":
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode the CSV file.")

    if extension in ("xlsx", "xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    if extension == "json":
        uploaded_file.seek(0)
        return pd.read_json(StringIO(uploaded_file.getvalue().decode("utf-8")))

    raise ValueError("Unsupported file format.")


def show_dashboard():
    st.header("🏠 Dashboard")
    df = st.session_state.cleaned_df

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    st.divider()
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, height=400)
    st.caption(f"Showing first 50 rows out of {len(df)} total rows.")

    st.divider()
    with st.expander("📊 Basic Statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)


st.sidebar.title("🤖 AI Data Analyst Agent")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧹 Data Cleaning",
        "📈 EDA",
        "📊 Visualization",
        "🤖 Machine Learning",
        "💬 Chat With Your Data",
        "🧠 AI Insights",
        "📄 Report"
    ]
)

st.title("🤖 AI Data Analyst Agent")
st.caption(
    "Upload a dataset to perform data cleaning, EDA, visualization, "
    "machine learning and AI-powered analysis."
)
st.divider()

st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls", "json"]
)

if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    file_id = getattr(uploaded_file, "file_id", None) or (
        uploaded_file.name,
        uploaded_file.size,
        hashlib.sha256(file_content).hexdigest()
    )

    if file_id != st.session_state.uploaded_file_id:
        st.session_state.original_df = None
        st.session_state.cleaned_df = None
        st.session_state.uploaded_file_id = None

        try:
            df = read_uploaded_file(uploaded_file)

            if df.empty or len(df.columns) == 0:
                raise ValueError("The uploaded dataset is empty or contains no columns.")

            if df.columns.duplicated().any():
                raise ValueError("The dataset contains duplicate column names. Rename them before uploading.")

            st.session_state.original_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.uploaded_file_id = file_id

            st.success(
                f"Dataset loaded successfully • {df.shape[0]} rows • {df.shape[1]} columns"
            )

        except Exception as e:
            st.error(f"Unable to read file: {e}")
            st.stop()

st.divider()

if st.session_state.cleaned_df is None:
    st.info("Supported formats: CSV, Excel (.xlsx, .xls) and JSON.")
    st.stop()

if page == "🏠 Dashboard":
    show_dashboard()
elif page == "🧹 Data Cleaning":
    show_cleaning()
elif page == "📈 EDA":
    show_eda()
elif page == "📊 Visualization":
    show_visualization()
elif page == "🤖 Machine Learning":
    show_ml()
elif page == "💬 Chat With Your Data":
    show_chat()
elif page == "🧠 AI Insights":
    show_ai_insights()
elif page == "📄 Report":
    show_report()