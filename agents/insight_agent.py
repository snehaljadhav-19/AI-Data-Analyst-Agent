import streamlit as st
import pandas as pd
import numpy as np


def show_ai_insights():
    st.header("🧠 AI Insights")

    df = st.session_state.get("cleaned_df")

    if df is None:
        st.info("Upload a dataset first.")
        return

    if df.empty or len(df.columns) == 0:
        st.warning("The dataset is empty.")
        return

    rows, cols = df.shape
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    total_cells = rows * cols
    missing_percentage = (missing / total_cells * 100) if total_cells else 0

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    st.subheader("📊 Dataset Health")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", cols)
    col3.metric("Missing Values", f"{missing:,}")
    col4.metric("Duplicates", f"{duplicates:,}")

    st.divider()
    st.subheader("🏥 Dataset Health Score")

    health_score = max(0, 100 - missing_percentage)
    st.progress(int(health_score))
    st.success(f"Dataset Completeness Score: {health_score:.1f}/100")
    st.caption(
        "This score measures completeness only. It does not evaluate "
        "duplicates, validity, consistency, or suitability for modeling."
    )

    st.divider()
    st.subheader("📌 Column Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Numeric Columns ({len(numeric_cols)})**")
        st.write(numeric_cols if numeric_cols else "No numeric columns found.")

    with col2:
        st.markdown(f"**Categorical Columns ({len(categorical_cols)})**")
        st.write(categorical_cols if categorical_cols else "No categorical columns found.")

    st.divider()
    st.subheader("🔍 Dataset Insights")

    insights = [
        f"The dataset contains {rows:,} records and {cols:,} features."
    ]

    if missing:
        insights.append(
            f"{missing:,} values are missing ({missing_percentage:.2f}% of all cells)."
        )
    else:
        insights.append("No missing values were detected.")

    if duplicates:
        insights.append(f"The dataset contains {duplicates:,} duplicate rows.")
    else:
        insights.append("No duplicate rows were detected.")

    if numeric_cols:
        numeric_data = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        means = numeric_data.mean().dropna()

        if not means.empty:
            highest_mean_col = means.idxmax()
            insights.append(
                f"'{highest_mean_col}' has the highest numerical mean among the numeric columns."
            )

        varying_cols = numeric_data.nunique(dropna=True)
        varying_cols = varying_cols[varying_cols > 1].index.tolist()

        if len(varying_cols) >= 2:
            corr_matrix = numeric_data[varying_cols].corr()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            corr_pairs = upper_triangle.stack().dropna()

            if not corr_pairs.empty:
                strongest_pair = corr_pairs.abs().idxmax()
                correlation = corr_pairs.loc[strongest_pair]
                strength = abs(correlation)

                if strength >= 0.7:
                    description = "strong"
                elif strength >= 0.4:
                    description = "moderate"
                else:
                    description = "weak"

                direction = "positive" if correlation > 0 else "negative"

                insights.append(
                    f"The strongest observed linear relationship is between "
                    f"'{strongest_pair[0]}' and '{strongest_pair[1]}' "
                    f"({description} {direction} correlation: {correlation:.2f})."
                )
        else:
            insights.append(
                "Not enough varying numerical columns are available for correlation analysis."
            )
    else:
        insights.append("No numerical columns are available for numerical analysis.")

    for insight in insights:
        st.info(insight)

    st.divider()
    st.subheader("🎯 Recommendations")

    recommendations = []

    if missing:
        recommendations.append(
            "Investigate missing values and select an appropriate imputation or removal strategy."
        )

    if duplicates:
        recommendations.append(
            "Review duplicate records and remove them if they are unintended."
        )

    if numeric_cols and len(numeric_cols) >= 2:
        recommendations.append(
            "Examine numerical relationships and potential multicollinearity before modeling."
        )

    if categorical_cols:
        recommendations.append(
            "Encode categorical variables when preparing data for machine learning."
        )

    high_cardinality = [
        col for col in categorical_cols
        if df[col].nunique(dropna=True) > 100
    ]

    if high_cardinality:
        recommendations.append(
            "Review high-cardinality categorical columns: "
            + ", ".join(map(str, high_cardinality))
            + ". Consider suitable encoding or feature reduction."
        )

    if not recommendations:
        recommendations.append(
            "No basic completeness or duplication issues were detected. "
            "Further validation is still recommended before modeling."
        )

    for recommendation in recommendations:
        st.write(f"✅ {recommendation}")

    st.divider()
    st.subheader("📋 Dataset Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Values",
            "Missing Percentage",
            "Duplicate Rows",
            "Numeric Columns",
            "Categorical Columns",
            "Completeness Score"
        ],
        "Value": [
            rows,
            cols,
            missing,
            f"{missing_percentage:.2f}%",
            duplicates,
            len(numeric_cols),
            len(categorical_cols),
            f"{health_score:.1f}/100"
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)