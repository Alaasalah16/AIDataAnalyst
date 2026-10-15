"""
Step 1: read the uploaded file and clean it.
"""

import io

import numpy as np
import pandas as pd


# Encodings the user can choose from (None = try UTF-8 automatically)
ENCODINGS = {
    "Auto": None,
    "Arabic Windows (cp1256)": "cp1256",
    "Latin-1": "latin1",
}


def load_file(file_bytes, file_name, encoding_name="Auto"):
    """Read a CSV or Excel file. Returns (dataframe, warning_or_None)."""

    if file_name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes)), None

    encoding = ENCODINGS.get(encoding_name)

    if encoding:
        return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding), None

    try:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig"), None
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")
        warning = (
            "The file is not UTF-8. If the text looks broken, "
            "choose another encoding (cp1256 for Arabic)."
        )
        return df, warning


def text_to_number(column):
    """
    Turn text like "$1,200" or "15%" into numbers.
    Returns None if the column is not really numbers.
    """

    text = column.astype(str).str.replace(r"[,$€£%\s]", "", regex=True)
    numbers = pd.to_numeric(text, errors="coerce")

    # Accept only if at least 90% of the filled values are numbers
    filled = column.notna()
    if filled.sum() == 0 or numbers[filled].notna().mean() < 0.9:
        return None

    return numbers


def text_to_date(column):
    """
    Turn a text column into dates.
    Returns None if the column is not really dates.
    """

    dates = pd.to_datetime(column, errors="coerce", format="mixed")

    filled = column.notna()
    if filled.sum() == 0 or dates[filled].notna().mean() < 0.8:
        return None

    return dates


def clean_data(df):
    """Clean the dataset. Returns (clean_df, report)."""

    df = df.copy()
    report = {"rows_before": len(df), "columns_before": len(df.columns)}

    # 1. Remove spaces around column names
    df.columns = df.columns.astype(str).str.strip()

    # 2. Remove columns that are completely empty
    empty_columns = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_columns)
    report["empty_columns"] = empty_columns

    # 3. Fix text columns
    report["number_columns"] = []
    report["date_columns"] = []

    for column in df.select_dtypes(include=["object", "string"]).columns:

        # Remove extra spaces and turn fake empty values into real ones
        values = df[column].astype("string").str.strip()
        values = values.replace(["", "nan", "None", "null", "N/A"], pd.NA)

        numbers = text_to_number(values)
        if numbers is not None:
            df[column] = numbers
            report["number_columns"].append(column)
            continue

        dates = text_to_date(values)
        if dates is not None:
            df[column] = dates
            report["date_columns"].append(column)
            continue

        df[column] = values

    # 4. Remove infinite values and duplicate rows
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates().reset_index(drop=True)

    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)

    return df, report
