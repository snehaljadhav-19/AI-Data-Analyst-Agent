import streamlit as st
import pandas as pd
from utils.helpers import load_dataset, dataset_summary

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent")
st.markdown(
    "Upload any **CSV dataset** and instantly explore its structure, statistics, and data quality."
)

st.sidebar.title("Navigation")
st.sidebar.success("📂 Upload a dataset to begin.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df, error = load_dataset(uploaded_file)

    if error:
        st.error(error)
    else:
        st.success("Dataset uploaded successfully!")

        summary = dataset_summary(df)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Rows", summary["Rows"])
        col2.metric("Columns", summary["Columns"])
        col3.metric("Missing", summary["Missing Values"])
        col4.metric("Duplicates", summary["Duplicate Rows"])
        col5.metric("Memory (KB)", summary["Memory Usage (KB)"])

        st.divider()

        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.divider()

        st.subheader("Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info_df, use_container_width=True)

        st.divider()

        st.subheader("Basic Statistics")

        st.dataframe(df.describe(include="all").T, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to start analyzing your data.")