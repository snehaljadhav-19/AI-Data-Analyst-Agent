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
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
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

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }