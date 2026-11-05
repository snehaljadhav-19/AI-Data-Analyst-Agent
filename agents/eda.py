import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda():
    st.header("📈 Exploratory Data Analysis")

    if st.session_state.cleaned_df is None:
        st.info("Upload a dataset first.")
        return

    df = st.session_state.cleaned_df

    st.subheader("📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()
    st.subheader("📝 Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique(dropna=True).values
    })
    st.dataframe(info_df, use_container_width=True, height=300, hide_index=True)

    st.divider()
    st.subheader("📊 Statistical Summary")
    try:
        summary = df.describe(include="all").T
        st.dataframe(summary, use_container_width=True, height=350)
    except (TypeError, ValueError):
        st.warning("Statistical summary could not be generated for this dataset.")

    st.divider()
    st.subheader("❌ Missing Values Analysis")

    missing_df = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Missing (%)": (df.isnull().mean().values * 100).round(2)
    }).sort_values("Missing Values", ascending=False)

    st.dataframe(missing_df, use_container_width=True, height=250, hide_index=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) > 1:
        correlation = df[numeric_cols].corr()
        valid_cols = correlation.columns[
            correlation.notna().any(axis=0)
        ].tolist()

        if len(valid_cols) > 1:
            st.divider()
            st.subheader("🔥 Correlation Heatmap")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(
                correlation.loc[valid_cols, valid_cols],
                annot=False,
                cmap="coolwarm",
                center=0,
                ax=ax
            )
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if numeric_cols:
        st.divider()
        st.subheader("📈 Numerical Analysis")

        selected_col = st.selectbox(
            "Select Numerical Column",
            numeric_cols,
            key="eda_numeric_column"
        )

        valid_values = pd.to_numeric(df[selected_col], errors="coerce").dropna()

        if valid_values.empty:
            st.info("No valid numerical values are available for this column.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distribution Analysis**")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(valid_values, kde=len(valid_values) > 1, ax=ax)
                ax.set_title(f"Distribution of {selected_col}", fontsize=10)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col2:
                st.markdown("**Box Plot**")
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(x=valid_values, ax=ax)
                ax.set_title(f"Box Plot of {selected_col}", fontsize=10)
                ax.tick_params(labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    categorical_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if categorical_cols:
        st.divider()
        st.subheader("📊 Category Analysis")

        selected_cat = st.selectbox(
            "Select Categorical Column",
            categorical_cols,
            key="eda_categorical_column"
        )

        value_counts = df[selected_cat].astype("string").fillna("Missing").value_counts().head(10)

        if not value_counts.empty:
            left, center, right = st.columns([1, 2, 1])

            with center:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                value_counts.plot(kind="bar", ax=ax)
                ax.set_title(f"Top Categories in {selected_cat}", fontsize=10)
                ax.set_xlabel(selected_cat)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", labelrotation=30, labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.divider()
    st.success("EDA completed successfully.")