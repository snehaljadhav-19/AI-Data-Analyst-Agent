import streamlit as st
import pandas as pd


def show_cleaning():
    st.header("🧹 Data Cleaning")

    if st.session_state.cleaned_df is None:
        st.info("Upload a dataset first.")
        return

    df = st.session_state.cleaned_df.copy()

    st.subheader("📋 Dataset Quality Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    st.divider()
    st.subheader("❓ Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing (%)": (df.isnull().mean().values * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⚙️ Cleaning Options")

    strategy = st.selectbox(
        "Missing Value Strategy",
        [
            "None",
            "Drop Missing Rows",
            "Fill Numeric With Mean",
            "Fill Numeric With Median",
            "Fill Categorical With Mode"
        ],
        key="cleaning_strategy"
    )

    remove_duplicates = st.checkbox(
        "Remove Duplicate Rows",
        key="remove_duplicates"
    )

    if st.button("Apply Cleaning", type="primary", key="apply_cleaning"):
        cleaned_df = df.copy()

        if strategy == "Drop Missing Rows":
            cleaned_df = cleaned_df.dropna()

        elif strategy in ("Fill Numeric With Mean", "Fill Numeric With Median"):
            numeric_cols = cleaned_df.select_dtypes(include="number").columns
            fill_values = (
                cleaned_df[numeric_cols].mean()
                if strategy == "Fill Numeric With Mean"
                else cleaned_df[numeric_cols].median()
            )
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(fill_values)

        elif strategy == "Fill Categorical With Mode":
            categorical_cols = cleaned_df.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns

            for col in categorical_cols:
                mode = cleaned_df[col].mode(dropna=True)
                if not mode.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])

        if remove_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()

        st.session_state.cleaned_df = cleaned_df
        st.success(
            f"Cleaning completed. {len(df) - len(cleaned_df)} rows removed; "
            f"{int(cleaned_df.isnull().sum().sum())} missing values remain."
        )
        st.rerun()

    st.divider()
    st.subheader("📄 Cleaned Dataset Preview")
    st.dataframe(
        st.session_state.cleaned_df,
        use_container_width=True,
        height=400
    )

    st.download_button(
        "⬇ Download Cleaned Dataset",
        data=st.session_state.cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key="download_cleaned_dataset"
    )