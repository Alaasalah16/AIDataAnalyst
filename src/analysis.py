"""
Step 2: find the column types and calculate the analysis tables.
"""

import pandas as pd


# Numbers in these columns are labels, not amounts (adding them is meaningless)
ID_WORDS = ["id", "code", "zip", "postal", "phone", "year"]


def get_number_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def get_date_columns(df):
    return df.select_dtypes(include="datetime").columns.tolist()


def get_text_columns(df):
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def get_metric_columns(df):
    """Number columns that make sense to sum (skips Order ID, Postal Code...)."""

    metrics = []
    for column in get_number_columns(df):
        words = column.lower().replace("_", " ").replace("-", " ").split()
        if not any(word in ID_WORDS for word in words):
            metrics.append(column)
    return metrics


def get_category_columns(df):
    """Text columns that are good for grouping (skips names, IDs, free text)."""

    categories = []
    for column in get_text_columns(df):
        unique = df[column].nunique()
        if 2 <= unique <= 50:
            categories.append(column)
    return categories


def numbers_summary(df):
    """count / mean / median / std / min / max / missing for every number column."""

    numbers = df[get_number_columns(df)]
    table = numbers.describe().T
    table["median"] = numbers.median()
    table["missing"] = numbers.isna().sum()
    return table[["count", "mean", "median", "std", "min", "max", "missing"]].round(2)


def kpi_table(df):
    """Total, average, min and max for every metric column."""

    rows = []
    for column in get_metric_columns(df):
        values = df[column].dropna()
        if len(values) > 0:
            rows.append({
                "Metric": column,
                "Total": values.sum(),
                "Average": values.mean(),
                "Minimum": values.min(),
                "Maximum": values.max(),
            })
    return pd.DataFrame(rows).round(2)


def monthly_trend(df, date_column, metric_column):
    """Total of a metric for each month."""

    data = df[[date_column, metric_column]].dropna()
    data["Month"] = data[date_column].dt.to_period("M").astype(str)
    return data.groupby("Month", as_index=False)[metric_column].sum()


def category_summary(df, category_column, metric_column):
    """Total / average / min / max of a metric for each category, best first."""

    table = (
        df.groupby(category_column)[metric_column]
        .agg(Total="sum", Average="mean", Minimum="min", Maximum="max")
        .reset_index()
        .sort_values("Total", ascending=False)
    )
    return table.round(2)


def date_summary(df, date_column):
    """Earliest date, latest date, number of days and number of valid dates."""

    dates = df[date_column].dropna()
    if dates.empty:
        return None

    return {
        "earliest": dates.min(),
        "latest": dates.max(),
        "days": (dates.max() - dates.min()).days,
        "records": len(dates),
    }


def profile_dataset(df):
    """Detailed information about every column (shown as JSON in the app)."""

    profile = {"rows": len(df), "columns": len(df.columns), "column_details": []}

    for column in df.columns:
        values = df[column]
        details = {
            "column": column,
            "dtype": str(values.dtype),
            "missing_values": int(values.isna().sum()),
            "missing_percentage": round(values.isna().mean() * 100, 2),
            "unique_values": int(values.nunique()),
            "sample_values": [str(v) for v in values.dropna().head(5)],
        }

        if column in get_number_columns(df):
            details["type"] = "numeric"
            details["statistics"] = {
                "mean": round(float(values.mean()), 2),
                "median": round(float(values.median()), 2),
                "minimum": round(float(values.min()), 2),
                "maximum": round(float(values.max()), 2),
                "standard_deviation": round(float(values.std()), 2),
            }
        elif column in get_date_columns(df):
            details["type"] = "date"
            details["earliest_date"] = str(values.min())
            details["latest_date"] = str(values.max())
        else:
            details["type"] = "categorical"
            top = values.value_counts().head(10)
            details["top_categories"] = {str(k): int(v) for k, v in top.items()}

        profile["column_details"].append(details)

    return profile


def short(value, max_length=40):
    """Shorten long cell values before sending them to the AI."""

    text = " ".join(str(value).split())
    return text if len(text) <= max_length else text[:max_length] + "..."


def describe_for_ai(df, max_characters=12000):
    """
    A short text description of the dataset for the AI model.
    The model never sees the rows, only this summary.
    """

    lines = [f"The dataset has {len(df)} rows and {len(df.columns)} columns.", ""]

    for column in df.columns:
        values = df[column]
        missing = values.isna().sum()
        line = f"- {column} | missing: {missing} | unique: {values.nunique()}"

        if column in get_number_columns(df):
            line += (
                f" | type: number | mean: {values.mean():.2f}"
                f" | median: {values.median():.2f}"
                f" | min: {values.min():.2f} | max: {values.max():.2f}"
            )
        elif column in get_date_columns(df):
            line += f" | type: date | from {values.min()} to {values.max()}"
        else:
            line += " | type: text"
            if column in get_category_columns(df):
                top = values.value_counts().head(5)
                line += " | most common: " + ", ".join(
                    f"{short(name)} ({count})" for name, count in top.items()
                )
            else:
                examples = values.dropna().head(3)
                line += " | examples: " + ", ".join(short(v) for v in examples)

        lines.append(line)

    text = "\n".join(lines)

    # The model has a limited memory; cut very long descriptions
    if len(text) > max_characters:
        text = text[:max_characters] + "\n[description cut: too many columns]"

    return text
