# AI Data Analyst Agent

An interactive data analysis application built with **Python and Streamlit** that simplifies data cleaning, exploratory data analysis, visualization, automated insight generation, and machine learning through a single interface.

<https://ai-data-analyst-agent-sj.streamlit.app/>

## Overview

AI Data Analyst Agent helps users explore and analyze structured datasets without writing extensive code. Upload a dataset, perform preprocessing, discover patterns, generate visualizations, train machine learning models, and export analytical reports through an interactive application.

## Features

- **Dataset Upload:** Supports CSV, Excel, and JSON files.
- **Data Understanding:** Displays dataset structure, dimensions, data types, and statistical summaries.
- **Data Cleaning:** Identifies missing values, duplicate records, and supports preprocessing.
- **Exploratory Data Analysis:** Explores distributions, relationships, and statistical patterns.
- **Data Visualization:** Generates interactive and statistical visualizations.
- **Automated Insights:** Extracts useful observations and patterns from datasets.
- **Machine Learning:** Supports classification and regression with multiple algorithms.
- **Model Evaluation:** Provides performance metrics, confusion matrices, and feature importance.
- **Chat With Data:** Allows users to ask supported questions about their datasets.
- **Report Generation:** Generates downloadable analytical reports.

## Machine Learning Algorithms

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Decision Tree Classifier | Decision Tree Regressor |
| Random Forest Classifier | Random Forest Regressor |

**Evaluation metrics:** Accuracy, Balanced Accuracy, Precision, Recall, F1-score, Confusion Matrix, MAE, RMSE, and R².

The ML module also supports feature selection, preprocessing, configurable training samples, model comparison, and model and prediction downloads.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Report Generation | ReportLab |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## Project Structure

```text
AI-Data-Analyst-Agent/
│
├── agents/
│   ├── cleaning.py
│   ├── eda.py
│   ├── insight_agent.py
│   ├── ml_agent.py
│   ├── report_agent.py
│   └── visualization.py
│
├── pages/
├── utils/
├── assets/
├── data/
├── models/
├── reports/
├── uploads/
├── charts/
├── exports/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/snehaljadhav-19/AI-Data-Analyst-Agent.git
cd AI-Data-Analyst-Agent
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

## Live Demo

**Try the application:**  
https://ai-data-analyst-agent-sj.streamlit.app/

## Author

**Snehal Jadhav**  
B.Tech – Artificial Intelligence and Data Science

[GitHub](https://github.com/snehaljadhav-19)

---

*AI Data Analyst Agent — Turning raw data into meaningful insights through interactive analytics and machine learning.*
