
import pandas as pd


def load_dataset(uploaded_file):
    """Load a CSV, Excel, or JSON dataset."""
    try:
        file_name = getattr(uploaded_file, "name", "").lower()
        extension = file_name.rsplit(".", 1)[-1] if "." in file_name else ""

        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if extension == "csv":
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig", low_memory=False)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)
        elif extension in {"xlsx", "xls"}:
            df = pd.read_excel(uploaded_file)
        elif extension == "json":
            df = pd.read_json(uploaded_file)
        else:
            return None, f"Unsupported file format: {extension or 'unknown'}"

        if df.empty:
            return None, "The uploaded file contains no data."

        if df.columns.duplicated().any():
            return None, "The dataset contains duplicate column names."

        return df, None

    except Exception as e:
        return None, str(e)


def dataset_summary(df):
    """Return basic dataset information."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    memory_kb = df.memory_usage(index=True, deep=True).sum() / 1024

    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isna().to_numpy().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": round(memory_kb, 2)
    }
