import re
from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)


NAVY = colors.HexColor("#17365D")
LIGHT_BLUE = colors.HexColor("#EAF1F8")
DARK = colors.HexColor("#263445")
GREY = colors.HexColor("#64748B")


def safe_text(value):
    return escape(str(value))


def format_number(value):
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def is_sensitive_column(column):
    name = str(column).lower()
    return bool(re.search(
        r"(^|[\s_-])(email|password|phone|mobile|fname|lname|first name|last name|street|address)([\s_-]|$)",
        name
    ))


def is_identifier_column(column):
    name = str(column).lower()
    return bool(re.search(
        r"(^|[\s_-])(id|identifier|zipcode|zip|latitude|longitude)([\s_-]|$)",
        name
    ))


def create_chart(series, title, chart_type="hist"):
    values = series.dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 2.5))

    try:
        if chart_type == "hist":
            values = pd.to_numeric(values, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()

            if values.empty:
                plt.close(fig)
                return None

            if len(values) > 10000:
                values = values.sample(10000, random_state=42)

            ax.hist(values, bins=25)
            ax.set_ylabel("Frequency")
        else:
            counts = values.astype(str).value_counts().head(8).sort_values()
            ax.barh(counts.index, counts.values)
            ax.set_xlabel("Records")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y" if chart_type == "hist" else "x", alpha=0.2)
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        buffer.seek(0)
        return buffer

    finally:
        plt.close(fig)


def create_pdf(df):
    if df is None or df.empty or len(df.columns) == 0:
        raise ValueError("The dataset is empty. Upload a valid dataset first.")

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="Dataset Analysis Report",
        author="AI Data Analyst"
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=GREY,
        alignment=TA_CENTER,
        spaceAfter=16
    ))

    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=NAVY,
        spaceBefore=9,
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=DARK,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
        textColor=DARK,
        wordWrap="CJK"
    ))

    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["TableCell"],
        fontName="Helvetica-Bold",
        textColor=colors.white
    ))

    def cell(value, header=False):
        style = styles["TableHeader"] if header else styles["TableCell"]
        return Paragraph(safe_text(value), style)

    def make_table(data, widths):
        formatted = [
            [cell(value, header=(row_index == 0)) for value in row]
            for row_index, row in enumerate(data)
        ]

        table = Table(
            formatted,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT"
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
        ]))

        return table

    rows, columns = df.shape
    total_cells = max(rows * columns, 1)
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_pct = missing / total_cells * 100
    completeness = 100 - missing_pct

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if not is_identifier_column(col)
        and not is_sensitive_column(col)
        and df[col].nunique(dropna=True) > 1
    ]

    categorical_cols = [
        col for col in df.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
        if not is_sensitive_column(col)
        and not is_identifier_column(col)
        and 1 < df[col].nunique(dropna=True) <= 100
    ]

    missing_table = df.isna().sum().sort_values(ascending=False)
    missing_table = missing_table[missing_table > 0].head(5)

    quality_rows = [["Column", "Missing Values", "Missing (%)"]]

    for col, count in missing_table.items():
        quality_rows.append([
            col,
            f"{int(count):,}",
            f"{count / max(rows, 1) * 100:.1f}%"
        ])

    if len(quality_rows) == 1:
        quality_rows.append(["No missing values detected", "0", "0.0%"])

    numeric_stats = []

    for col in numeric_cols[:6]:
        values = pd.to_numeric(df[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()

        if not values.empty:
            numeric_stats.append([
                col,
                format_number(values.mean()),
                format_number(values.median()),
                format_number(values.min()),
                format_number(values.max())
            ])

    category_stats = []

    for col in categorical_cols[:5]:
        counts = df[col].dropna().astype(str).value_counts()

        if not counts.empty:
            category_stats.append([
                col,
                counts.index[0],
                f"{counts.iloc[0]:,}",
                f"{counts.iloc[0] / counts.sum() * 100:.1f}%"
            ])

    recommendations = []

    if missing_pct > 0:
        recommendations.append(
            "Review columns with high missing-value percentages and decide whether to impute or exclude them."
        )

    if duplicates:
        recommendations.append(
            f"Investigate the {duplicates:,} duplicate rows before analysis."
        )

    if numeric_cols:
        recommendations.append(
            "Inspect numerical distributions and potential outliers before selecting statistical or machine-learning methods."
        )

    if categorical_cols:
        recommendations.append(
            "Review category frequencies and consolidate rare or inconsistent labels where appropriate."
        )

    if not recommendations:
        recommendations.append(
            "No major issues were identified by these basic checks. Validate data meaning and suitability for the intended analysis."
        )

    content = [
        Spacer(1, 12),
        Paragraph("DATASET ANALYSIS REPORT", styles["ReportTitle"]),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Subtitle"]
        ),
        Paragraph(
            "Essential dataset overview and analytical findings",
            styles["Subtitle"]
        )
    ]

    overview = [
        ["DATASET INDICATOR", "RESULT"],
        ["Records", f"{rows:,}"],
        ["Columns", f"{columns:,}"],
        ["Numerical variables", f"{len(df.select_dtypes(include='number').columns):,}"],
        ["Categorical variables", f"{len(df.select_dtypes(include=['object', 'category', 'bool', 'string']).columns):,}"],
        ["Missing cells", f"{missing:,} ({missing_pct:.2f}%)"],
        ["Duplicate rows", f"{duplicates:,}"],
        ["Cell completeness", f"{completeness:.2f}%"]
    ]

    content += [
        Paragraph("1. Executive Summary", styles["Section"]),
        Paragraph(
            f"The dataset contains <b>{rows:,} records</b> and "
            f"<b>{columns:,} attributes</b>. Overall cell completeness is "
            f"<b>{completeness:.2f}%</b>. The following indicators summarize "
            "its basic structure and completeness.",
            styles["BodySmall"]
        ),
        make_table(overview, [85 * mm, 85 * mm]),
        Paragraph("2. Data Quality", styles["Section"]),
        Paragraph(
            "The following columns have the highest number of missing values. "
            "Investigate these fields before using them in analysis or modelling.",
            styles["BodySmall"]
        ),
        make_table(quality_rows, [85 * mm, 42 * mm, 43 * mm]),
        Paragraph("3. Recommendations", styles["Section"])
    ]

    for recommendation in recommendations[:4]:
        content.append(
            Paragraph(f"• {safe_text(recommendation)}", styles["BodySmall"])
        )

    content.append(PageBreak())

    content += [
        Spacer(1, 8),
        Paragraph("KEY ANALYTICAL FINDINGS", styles["ReportTitle"]),
        Paragraph(
            "Selected descriptive statistics and category patterns",
            styles["Subtitle"]
        ),
        Paragraph("4. Numerical Summary", styles["Section"])
    ]

    if numeric_stats:
        content.append(make_table(
            [["Variable", "Mean", "Median", "Minimum", "Maximum"]] + numeric_stats,
            [58 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm]
        ))
    else:
        content.append(
            Paragraph(
                "No suitable numerical variables were available.",
                styles["BodySmall"]
            )
        )

    content.append(Paragraph("5. Categorical Summary", styles["Section"]))

    if category_stats:
        content.append(make_table(
            [["Variable", "Most Frequent Category", "Count", "Share"]] + category_stats,
            [42 * mm, 65 * mm, 30 * mm, 33 * mm]
        ))
    else:
        content.append(
            Paragraph(
                "No suitable categorical variables were available.",
                styles["BodySmall"]
            )
        )

    if numeric_cols:
        chart = create_chart(
            df[numeric_cols[0]],
            f"Distribution of {numeric_cols[0]}"
        )

        if chart:
            content.append(Paragraph("6. Distribution", styles["Section"]))
            content.append(Image(chart, width=165 * mm, height=59 * mm))

    if categorical_cols:
        chart = create_chart(
            df[categorical_cols[0]],
            f"Most frequent categories in {categorical_cols[0]}",
            "bar"
        )

        if chart:
            content.append(Paragraph("Category Frequency", styles["Section"]))
            content.append(Image(chart, width=165 * mm, height=59 * mm))

    content += [
        Spacer(1, 8),
        Paragraph(
            "Interpretation: These results are descriptive summaries of the uploaded data. "
            "They do not establish causation or guarantee that the dataset is suitable "
            "for a particular predictive task.",
            styles["BodySmall"]
        )
    ]

    def add_page_number(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(colors.HexColor("#D5DEE8"))
        canvas.line(17 * mm, 13 * mm, width - 17 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(17 * mm, 8 * mm, "AI Data Analyst | Confidential")
        canvas.drawRightString(
            width - 17 * mm,
            8 * mm,
            f"Page {document.page}"
        )
        canvas.restoreState()

    doc.build(
        content,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def show_report():
    st.header("📄 Dataset Report")
    st.caption(
        "Generate a concise PDF with essential dataset indicators and analytical findings."
    )

    df = st.session_state.get("cleaned_df")

    if df is None:
        st.info("Upload a dataset first.")
        return

    if df.empty:
        st.warning("The dataset is empty.")
        return

    cols = st.columns(3)
    cols[0].metric("Records", f"{len(df):,}")
    cols[1].metric("Columns", f"{len(df.columns):,}")
    cols[2].metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    if st.button(
        "Generate Professional PDF",
        type="primary",
        use_container_width=True
    ):
        try:
            with st.spinner("Preparing your report..."):
                pdf = create_pdf(df)

            st.success("Your report is ready.")

            st.download_button(
                "⬇️ Download Dataset Report",
                data=pdf,
                file_name="AI_Data_Analyst_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        except Exception as error:
            st.error(f"Report generation failed: {error}")