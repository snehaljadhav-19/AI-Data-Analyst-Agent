import re
import difflib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_column(question, columns):
    q = normalize(question)
    normalized = {normalize(col): col for col in columns}

    for name, col in normalized.items():
        if name and re.search(rf"\b{re.escape(name)}\b", q):
            return col

    words = q.split()

    for size in range(min(6, len(words)), 0, -1):
        for i in range(len(words) - size + 1):
            candidate = " ".join(words[i:i + size])
            match = difflib.get_close_matches(
                candidate, normalized, n=1, cutoff=0.88
            )
            if match:
                return normalized[match[0]]

    return None


def show_result_chart(data, title, kind="bar", xlabel=None, ylabel=None):
    if data is None or len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))

    try:
        if kind == "bar":
            data.plot(kind="bar", ax=ax)
        elif kind == "hist":
            sns.histplot(data, kde=True, ax=ax)
        elif kind == "box":
            sns.boxplot(x=data, ax=ax)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    finally:
        plt.close(fig)


def analyze_question(question, df):
    q = normalize(question)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    if df.empty:
        return "The dataset is empty. Upload a dataset containing records.", None

    if any(word in q for word in ["overview", "summary", "describe dataset", "about dataset"]):
        result = pd.DataFrame({
            "Metric": [
                "Rows", "Columns", "Numerical Columns",
                "Categorical Columns", "Missing Values", "Duplicate Rows"
            ],
            "Value": [
                len(df), len(df.columns), len(numeric_cols),
                len(categorical_cols), int(df.isna().sum().sum()),
                int(df.duplicated().sum())
            ]
        })
        return "Here is an overview of your dataset.", result

    if "row" in q and any(word in q for word in ["how many", "number", "count", "total"]):
        return f"Your dataset contains **{len(df):,} rows**.", None

    if "column" in q and any(word in q for word in ["how many", "number", "count", "total"]):
        return f"Your dataset contains **{len(df.columns)} columns**.", None

    if any(word in q for word in ["column names", "list columns", "what columns", "column list"]):
        return "These are the columns in your dataset.", pd.DataFrame({"Columns": df.columns})

    if "missing" in q or "null values" in q or "empty values" in q:
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        if missing.empty:
            return "No missing values were found in the dataset.", None

        result = missing.rename("Missing Values").to_frame()
        result["Missing %"] = (result["Missing Values"] / len(df) * 100).round(2)

        return (
            f"Found **{int(missing.sum()):,} missing values** across {len(missing)} columns.",
            result
        )

    if "duplicate" in q:
        count = int(df.duplicated().sum())
        return f"Your dataset contains **{count:,} duplicate rows**.", None

    if any(word in q for word in ["correlation", "correlated", "relationship between"]):
        if len(numeric_cols) < 2:
            return "At least two numerical columns are required for correlation analysis.", None

        numeric_data = df[numeric_cols].replace([float("inf"), float("-inf")], float("nan"))
        col = find_column(question, numeric_cols)

        if col:
            correlations = numeric_data.corr()[col].drop(col).dropna()
            result = correlations.reindex(
                correlations.abs().sort_values(ascending=False).index
            )
            return (
                f"Correlation of **{col}** with other numerical columns.",
                result.rename("Correlation").to_frame()
            )

        result = numeric_data.corr()
        return "Here is the numerical correlation matrix.", result

    if any(word in q for word in ["top", "most frequent", "most common", "popular"]):
        col = find_column(question, df.columns)

        if not col:
            if not categorical_cols:
                return "No categorical columns are available for category analysis.", None

            return (
                f"Specify a column name. Available categorical columns: "
                f"{', '.join(map(str, categorical_cols[:10]))}.",
                None
            )

        limit_match = re.search(r"\btop\s+(\d+)\b", q)
        limit = min(int(limit_match.group(1)), 50) if limit_match else 10
        result = df[col].value_counts(dropna=False).head(limit)

        return f"Top {limit} most frequent values in **{col}**.", result.rename("Count").to_frame()

    aggregate_keywords = [
        "average", "mean", "median", "sum", "total", "minimum",
        "maximum", "highest", "lowest", "smallest", "largest",
        "standard deviation", "std"
    ]

    if any(word in q for word in aggregate_keywords):
        col = find_column(question, numeric_cols)

        if not col:
            return (
                f"Please specify a numerical column. Available columns: "
                f"{', '.join(map(str, numeric_cols[:15]))}.",
                None
            )

        series = pd.to_numeric(df[col], errors="coerce")
        series = series.replace([float("inf"), float("-inf")], float("nan")).dropna()

        if series.empty:
            return f"Column **{col}** has no valid numerical values.", None

        if "standard deviation" in q or "std" in q:
            return f"Standard deviation of **{col}**: **{series.std():,.3f}**", None

        if "median" in q:
            return f"Median of **{col}**: **{series.median():,.3f}**", None

        if "average" in q or "mean" in q:
            return f"Average of **{col}**: **{series.mean():,.3f}**", None

        if "sum" in q or "total" in q:
            return f"Total of **{col}**: **{series.sum():,.3f}**", None

        if any(word in q for word in ["minimum", "lowest", "smallest"]):
            idx = series.idxmin()
            return f"Minimum **{col}**: **{series.min():,.3f}**", df.loc[[idx]]

        if any(word in q for word in ["maximum", "highest", "largest"]):
            idx = series.idxmax()
            return f"Maximum **{col}**: **{series.max():,.3f}**", df.loc[[idx]]

    if "distribution" in q or "histogram" in q:
        col = find_column(question, numeric_cols)

        if not col:
            return "Specify a numerical column for distribution analysis.", None

        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([float("inf"), float("-inf")], float("nan")).dropna()

        return f"Distribution of **{col}**.", {
            "chart": "hist",
            "data": values
        }

    if "box plot" in q or "boxplot" in q:
        col = find_column(question, numeric_cols)

        if not col:
            return "Specify a numerical column for the box plot.", None

        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([float("inf"), float("-inf")], float("nan")).dropna()

        return f"Box plot of **{col}**.", {
            "chart": "box",
            "data": values
        }

    if any(word in q for word in ["show data", "preview", "sample records", "display records", "first rows"]):
        match = re.search(r"\b(\d+)\b", q)
        limit = min(int(match.group(1)), 100) if match else 10
        return f"Showing the first {limit} records.", df.head(limit)

    return (
        "I couldn't identify the required analysis. Try asking about dataset statistics, "
        "missing values, duplicates, averages, highest or lowest values, top categories, "
        "correlations, distributions or sample records.",
        None
    )


def show_chat():
    st.header("💬 Chat With Your Data")
    st.caption("Ask questions about your uploaded dataset and receive data-backed answers.")

    df = st.session_state.get("cleaned_df")

    if df is None:
        st.info("Upload a dataset first.")
        return

    if df.empty or len(df.columns) == 0:
        st.warning("The uploaded dataset contains no usable data.")
        return

    dataset_id = st.session_state.get("uploaded_file_id")

    if st.session_state.get("chat_dataset_id") != dataset_id:
        st.session_state.chat_history = []
        st.session_state.chat_dataset_id = dataset_id

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        - Give me an overview of the dataset.
        - How many rows and columns are there?
        - Which columns contain missing values?
        - Show the top 5 categories in Type.
        - What is the average Sales?
        - Find the highest Profit.
        - Show the correlation matrix.
        - Show the distribution of Sales.
        - Display the first 20 records.
        """)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            result = message.get("data")

            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif isinstance(result, pd.Series):
                st.dataframe(result.to_frame(), use_container_width=True)
            elif isinstance(result, dict) and "chart" in result:
                show_result_chart(
                    result["data"],
                    message["content"],
                    result["chart"]
                )

    question = st.chat_input("Ask anything about your dataset...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer, result = analyze_question(question, df)
                st.markdown(answer)

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                elif isinstance(result, pd.Series):
                    st.dataframe(result.to_frame(), use_container_width=True)
                elif isinstance(result, dict) and "chart" in result:
                    show_result_chart(result["data"], answer, result["chart"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "data": result
                })

            except Exception:
                answer = "Unable to analyze this question. Please check the column name and try again."
                st.error(answer)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

    if st.session_state.chat_history and st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()