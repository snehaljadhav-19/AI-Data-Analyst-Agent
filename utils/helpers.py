import pandas as pd


def load_dataset(uploaded_file):
    """Load uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def dataset_summary(df):
    """Return basic dataset information."""
    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }
    return summary