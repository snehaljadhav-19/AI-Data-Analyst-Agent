
import pandas as pd


def load_dataset(uploaded_file):
    """Load an uploaded CSV file with automatic encoding detection."""
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1")

    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            if df.empty or len(df.columns) == 0:
                return None, "The uploaded CSV file is empty or contains no valid columns."
            return df, None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The uploaded CSV file is empty."
        except pd.errors.ParserError as e:
            return None, f"Unable to parse the CSV file: {e}"
        except Exception as e:
            return None, str(e)

    return None, "Unable to decode the CSV file. Please check its encoding."


def dataset_summary(df):
    """Return basic dataset information."""
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }
