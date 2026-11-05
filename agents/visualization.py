import streamlit as st
import pandas as pd
import plotly.express as px

CHARTS = [
    "Histogram", "Box Plot", "Bar Chart", "Line Chart",
    "Scatter Plot", "Pie Chart", "Correlation Heatmap",
    "Area Chart", "Violin Plot", "Treemap", "Stacked Bar Chart",
    "Bubble Chart", "Density Heatmap", "Count Plot",
    "Sunburst Chart", "Pair Plot"
]

MAX_POINTS = 10000
CHART_HEIGHT = 290


def detect_columns(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    dates = []

    for col in df.columns:
        if col in numeric:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dates.append(col)
        elif any(token in str(col).lower() for token in ("date", "time", "timestamp")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                dates.append(col)

    categorical = [
        col for col in df.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns if col not in dates
    ]

    return numeric, categorical, dates


def sample_data(df, limit=MAX_POINTS):
    return df.sample(n=limit, random_state=42) if len(df) > limit else df


def chart_layout(fig, height=CHART_HEIGHT):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=20, t=50, b=40),
        title_font=dict(size=14),
        font=dict(size=10),
        legend=dict(orientation="h", y=-0.25),
        hoverlabel=dict(font_size=11)
    )
    return fig


def render_chart(fig, height=CHART_HEIGHT):
    st.plotly_chart(
        chart_layout(fig, height),
        use_container_width=True,
        config={"responsive": True, "displayModeBar": True}
    )


def get_recommendations(df, numeric, categorical, dates):
    recommendations = []

    if numeric:
        recommendations.append(("Distribution", "Histogram", numeric[0]))

    if categorical:
        recommendations.append(("Category Analysis", "Bar Chart", categorical[0]))

    if len(numeric) >= 2:
        recommendations.append(("Feature Relationship", "Scatter Plot", (numeric[0], numeric[1])))
        recommendations.append(("Correlation Analysis", "Correlation Heatmap", numeric[:min(10, len(numeric))]))

    if numeric and categorical:
        recommendations.append(("Category Distribution", "Box Plot", (numeric[0], categorical[0])))

    if dates and numeric:
        recommendations.append(("Time Trend", "Line Chart", (dates[0], numeric[0])))

    return recommendations


def show_automatic_dashboard(df, numeric, categorical, dates):
    st.subheader("✨ Automatic Visualization Dashboard")
    st.caption("Charts are selected automatically using the available dataset columns.")

    recommendations = get_recommendations(df, numeric, categorical, dates)

    if not recommendations:
        st.info("Not enough compatible columns to generate automatic charts.")
        return

    plot_df = sample_data(df)
    chart_columns = st.columns(2)

    for i, (title, chart_type, columns) in enumerate(recommendations):
        with chart_columns[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")

                try:
                    if chart_type == "Histogram":
                        fig = px.histogram(plot_df, x=columns, nbins=30, title=f"Distribution of {columns}")

                    elif chart_type == "Bar Chart":
                        counts = (
                            df[columns].astype("string").fillna("Missing")
                            .value_counts().head(10)
                            .rename_axis(columns).reset_index(name="Count")
                        )
                        fig = px.bar(counts, x=columns, y="Count", title=f"Top 10 {columns}")

                    elif chart_type == "Scatter Plot":
                        x, y = columns
                        fig = px.scatter(plot_df, x=x, y=y, opacity=0.55, title=f"{x} vs {y}")

                    elif chart_type == "Correlation Heatmap":
                        corr = df[columns].corr()
                        valid_cols = corr.columns[corr.notna().any()].tolist()

                        if len(valid_cols) < 2:
                            st.info("Insufficient variation for correlation analysis.")
                            continue

                        fig = px.imshow(
                            corr.loc[valid_cols, valid_cols],
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title="Numerical Correlations"
                        )

                    elif chart_type == "Box Plot":
                        y, category = columns
                        box_df = plot_df[[y, category]].dropna()
                        top_categories = box_df[category].value_counts().head(8).index
                        box_df = box_df[box_df[category].isin(top_categories)]

                        if box_df.empty:
                            st.info("No valid data available for this chart.")
                            continue

                        fig = px.box(box_df, x=category, y=y, title=f"{y} by {category}")

                    elif chart_type == "Line Chart":
                        x, y = columns
                        trend = df[[x, y]].copy()
                        trend[x] = pd.to_datetime(trend[x], errors="coerce")
                        trend[y] = pd.to_numeric(trend[y], errors="coerce")
                        trend = (
                            trend.dropna()
                            .groupby(x, as_index=False)[y].mean()
                            .sort_values(x)
                        )

                        if trend.empty:
                            st.info("No valid time-series data available.")
                            continue

                        fig = px.line(trend, x=x, y=y, title=f"{y} over time")

                    render_chart(fig, 270)

                except Exception:
                    st.warning(f"Unable to generate {title} for the selected dataset.")


def show_manual_builder(df, numeric, categorical, dates):
    st.subheader("🛠️ Manual Chart Builder")
    st.caption("Choose a chart, configure its columns and generate an interactive visualization.")

    chart_type = st.selectbox("Select Chart Type", CHARTS, key="manual_chart_type")
    plot_df = sample_data(df)
    fig = None

    if chart_type == "Histogram":
        if not numeric:
            st.warning("No numerical columns found.")
            return

        col = st.selectbox("Numerical Column", numeric, key="hist_column")
        fig = px.histogram(plot_df, x=col, nbins=40, marginal="box", title=f"Distribution of {col}")

    elif chart_type == "Box Plot":
        if not numeric:
            st.warning("No numerical columns found.")
            return

        y = st.selectbox("Numerical Column", numeric, key="box_numeric")
        category = st.selectbox("Group By (Optional)", ["None"] + categorical, key="box_category")
        fig = px.box(
            plot_df, x=category if category != "None" else None,
            y=y, color=category if category != "None" else None,
            title=f"Box Plot - {y}"
        )

    elif chart_type in ["Bar Chart", "Count Plot", "Pie Chart"]:
        if not categorical:
            st.warning("No categorical columns found.")
            return

        col = st.selectbox("Categorical Column", categorical, key="category_column")
        counts = (
            df[col].astype("string").fillna("Missing")
            .value_counts().head(10)
            .rename_axis(col).reset_index(name="Count")
        )

        if chart_type == "Pie Chart":
            fig = px.pie(counts, names=col, values="Count", hole=0.35, title=f"Distribution of {col}")
        else:
            fig = px.bar(counts, x=col, y="Count", title=f"{chart_type} - {col}", color="Count")

    elif chart_type in ["Line Chart", "Area Chart"]:
        if not numeric:
            st.warning("No numerical columns found.")
            return

        x_options = dates + numeric
        x = st.selectbox("X Axis", x_options, key="trend_x")
        y = st.selectbox("Y Axis", numeric, key="trend_y")

        if x == y:
            st.warning("Select different X and Y columns.")
            return

        trend = df[[x, y]].copy()
        trend[y] = pd.to_numeric(trend[y], errors="coerce")

        if x in dates:
            trend[x] = pd.to_datetime(trend[x], errors="coerce")
            trend = trend.dropna().groupby(x, as_index=False)[y].mean().sort_values(x)
        else:
            trend = trend.dropna().sort_values(x)

        if trend.empty:
            st.warning("No valid data available for this chart.")
            return

        fig = (
            px.area(trend, x=x, y=y, title=f"{y} over {x}")
            if chart_type == "Area Chart"
            else px.line(trend, x=x, y=y, title=f"{y} over {x}")
        )

    elif chart_type in ["Scatter Plot", "Bubble Chart", "Density Heatmap"]:
        required = 3 if chart_type == "Bubble Chart" else 2

        if len(numeric) < required:
            st.warning("Not enough numerical columns for this chart.")
            return

        x = st.selectbox("X Axis", numeric, key="manual_x")
        y = st.selectbox("Y Axis", [col for col in numeric if col != x], key="manual_y")

        if chart_type == "Scatter Plot":
            color = st.selectbox("Color By (Optional)", ["None"] + categorical, key="scatter_color")
            fig = px.scatter(
                plot_df, x=x, y=y,
                color=color if color != "None" else None,
                title=f"{x} vs {y}", opacity=0.65
            )

        elif chart_type == "Bubble Chart":
            size = st.selectbox("Bubble Size", [col for col in numeric if col not in [x, y]], key="bubble_size")
            bubble_df = plot_df[[x, y, size]].copy()
            bubble_df[size] = pd.to_numeric(bubble_df[size], errors="coerce")
            bubble_df = bubble_df.dropna()
            bubble_df = bubble_df[bubble_df[size] > 0]

            if bubble_df.empty:
                st.warning("No positive bubble-size values are available.")
                return

            fig = px.scatter(bubble_df, x=x, y=y, size=size, title=f"{x} vs {y}", size_max=35)

        else:
            fig = px.density_heatmap(
                plot_df, x=x, y=y,
                nbinsx=30, nbinsy=30,
                title=f"Density: {x} vs {y}"
            )

    elif chart_type == "Correlation Heatmap":
        if len(numeric) < 2:
            st.warning("At least two numerical columns are required.")
            return

        selected = st.multiselect(
            "Select Numerical Columns", numeric,
            default=numeric[:min(12, len(numeric))],
            key="correlation_columns"
        )

        if len(selected) < 2:
            st.warning("Select at least two columns.")
            return

        corr = df[selected].corr()
        valid_cols = corr.columns[corr.notna().any()].tolist()

        if len(valid_cols) < 2:
            st.warning("Selected columns do not contain enough variation for correlation analysis.")
            return

        fig = px.imshow(
            corr.loc[valid_cols, valid_cols],
            text_auto=".2f" if len(valid_cols) <= 15 else False,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Heatmap"
        )

    elif chart_type == "Violin Plot":
        if not numeric or not categorical:
            st.warning("This chart requires numerical and categorical columns.")
            return

        y = st.selectbox("Numerical Column", numeric, key="violin_numeric")
        category = st.selectbox("Group By", categorical, key="violin_category")
        violin_df = plot_df[[category, y]].dropna()
        top = violin_df[category].value_counts().head(10).index
        violin_df = violin_df[violin_df[category].isin(top)]

        if violin_df.empty:
            st.warning("No valid data available for this chart.")
            return

        fig = px.violin(
            violin_df, x=category, y=y,
            color=category, box=True, points=False,
            title=f"{y} by {category}"
        )

    elif chart_type in ["Treemap", "Sunburst Chart"]:
        if len(categorical) < 2:
            st.warning("At least two categorical columns are required.")
            return

        levels = st.multiselect(
            "Hierarchy Levels", categorical,
            default=categorical[:min(3, len(categorical))],
            max_selections=5,
            key="hierarchy_levels"
        )

        if len(levels) < 2:
            st.warning("Select at least two hierarchy levels.")
            return

        hierarchy = df.groupby(levels, observed=True, dropna=False).size().reset_index(name="Count")
        hierarchy = hierarchy.dropna(subset=levels)

        if hierarchy.empty:
            st.warning("No valid hierarchy data is available.")
            return

        fig = (
            px.treemap(hierarchy, path=levels, values="Count", title="Category Treemap")
            if chart_type == "Treemap"
            else px.sunburst(hierarchy, path=levels, values="Count", title="Category Sunburst")
        )

    elif chart_type == "Stacked Bar Chart":
        if len(categorical) < 2:
            st.warning("At least two categorical columns are required.")
            return

        x = st.selectbox("X Axis", categorical, key="stacked_x")
        color = st.selectbox("Stack By", [col for col in categorical if col != x], key="stacked_color")
        stacked = df.groupby([x, color], observed=True, dropna=False).size().reset_index(name="Count")
        top_x = df[x].value_counts().head(12).index
        top_color = df[color].value_counts().head(10).index
        stacked = stacked[stacked[x].isin(top_x) & stacked[color].isin(top_color)]

        if stacked.empty:
            st.warning("No valid category combinations are available.")
            return

        fig = px.bar(stacked, x=x, y="Count", color=color, barmode="stack", title=f"{x} by {color}")

    elif chart_type == "Pair Plot":
        if len(numeric) < 2:
            st.warning("At least two numerical columns are required.")
            return

        selected = st.multiselect(
            "Select Numerical Columns (Maximum 5)",
            numeric, default=numeric[:min(3, len(numeric))],
            max_selections=5,
            key="pairplot_columns"
        )

        if len(selected) < 2:
            st.warning("Select at least two columns.")
            return

        pair_df = plot_df[selected].dropna()

        if pair_df.empty:
            st.warning("No complete numerical rows are available.")
            return

        fig = px.scatter_matrix(pair_df, dimensions=selected, title="Feature Relationships")

    if fig is not None:
        render_chart(fig, 430)
        st.download_button(
            "⬇️ Download Chart (HTML)",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="data_visualization.html",
            mime="text/html",
            key="download_visualization"
        )


def show_visualization():
    st.header("📊 Data Visualization")
    st.caption("Explore your dataset through automatic visual analytics and interactive charts.")

    if st.session_state.get("cleaned_df") is None:
        st.info("Upload a dataset first.")
        return

    df = st.session_state.cleaned_df

    if df.empty:
        st.warning("The dataset is empty.")
        return

    numeric, categorical, dates = detect_columns(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Numerical Columns", len(numeric))
    col3.metric("Categorical Columns", len(categorical))

    st.divider()
    show_automatic_dashboard(df, numeric, categorical, dates)

    st.divider()
    show_manual_builder(df, numeric, categorical, dates)

    st.divider()
    st.success("Visualization generated successfully.")