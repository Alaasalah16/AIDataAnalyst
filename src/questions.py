"""
Step 3: answer simple questions directly with pandas (no AI needed).

answer_question() returns (text, table) when it understands the question,
or None when it does not, so the question can be sent to the AI instead.

Words are matched as WHOLE words: "summary" does not count as "sum" and
"laptop" does not count as "top".
"""

import re

from src.analysis import get_number_columns, get_text_columns


def clean_text(text):
    """Lower case, and "_" / "-" become spaces: "Order_Date" -> "order date"."""
    return re.sub(r"[_\-\s]+", " ", str(text).lower()).strip()


def has_word(question, *words):
    """True if the question contains any of the words as a whole word."""
    return any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", question) for word in words)


def find_column(question, columns):
    """
    The column mentioned in the question.
    If two columns match ("Sales" and "Sales Amount"), the longer name wins.
    Returns None if no column is mentioned.
    """

    found = [c for c in columns if has_word(question, clean_text(c))]
    if not found:
        return None
    return max(found, key=lambda c: len(str(c)))


def find_number(question, default=5):
    """The first number in the question: "top 10 ..." -> 10."""
    match = re.search(r"\b(\d+)\b", question)
    return min(int(match.group(1)), 100) if match else default


def answer_question(df, question):

    q = clean_text(question)
    metric = find_column(q, get_number_columns(df))
    category = find_column(q, get_text_columns(df))
    use_average = has_word(q, "average", "mean", "avg")

    # --- About the dataset itself ---
    if has_word(q, "how many rows", "number of rows", "how many records"):
        return f"The dataset has {len(df):,} rows.", None

    if has_word(q, "how many columns", "number of columns"):
        return f"The dataset has {len(df.columns)} columns.", None

    if has_word(q, "what columns", "list columns", "column names", "which columns"):
        return "Available columns:", df.dtypes.astype(str).rename("Type").to_frame()

    if has_word(q, "unique values", "unique categories"):
        unique = df.nunique().rename("Unique values").to_frame()
        return "Unique values per column:", unique

    if has_word(q, "overview", "dataset summary", "summarize dataset"):
        text = (
            f"Rows: {len(df):,} · Columns: {len(df.columns)} · "
            f"Number columns: {len(get_number_columns(df))} · "
            f"Text columns: {len(get_text_columns(df))} · "
            f"Missing values: {df.isna().sum().sum():,} · "
            f"Duplicate rows: {df.duplicated().sum():,}"
        )
        return text, None

    if has_word(q, "missing", "null", "nulls"):
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return "There are no missing values.", None
        return "Missing values per column:", missing.rename("Missing").to_frame()

    if has_word(q, "duplicate", "duplicates"):
        return f"There are {df.duplicated().sum():,} duplicate rows.", None

    # --- Top / bottom: "top 5 Category by Sales", "lowest 3 Profit" ---
    is_top = has_word(q, "top", "highest", "best", "most")
    is_bottom = has_word(q, "bottom", "lowest", "worst", "least")

    if metric and (is_top or is_bottom):
        n = find_number(q)
        word = "Top" if is_top else "Bottom"

        if category:
            groups = df.groupby(category)[metric]
            totals = groups.mean() if use_average else groups.sum()
            totals = totals.sort_values(ascending=is_bottom).head(n).round(2)
            return f"{word} {n} {category} by {metric}:", totals.to_frame()

        rows = df.nlargest(n, metric) if is_top else df.nsmallest(n, metric)
        return f"{word} {n} rows by {metric}:", rows

    # --- Group: "Sales by Region", "average Profit per Category" ---
    if metric and category and has_word(q, "by", "per", "each", "for each"):
        groups = df.groupby(category)[metric]
        totals = groups.mean() if use_average else groups.sum()
        label = "Average" if use_average else "Total"
        table = totals.sort_values(ascending=False).round(2).to_frame()
        return f"{label} {metric} by {category}:", table

    # --- One number: "total Sales", "average Profit" ---
    if metric and not category:
        values = df[metric]
        calculations = {
            "total": (["total", "sum"], values.sum),
            "average": (["average", "mean", "avg"], values.mean),
            "median": (["median"], values.median),
            "maximum": (["maximum", "max"], values.max),
            "minimum": (["minimum", "min"], values.min),
        }
        for name, (words, calculate) in calculations.items():
            if has_word(q, *words):
                return f"The {name} of {metric} is {calculate():,.2f}.", None

    # --- Count per category: "distribution of Region" ---
    if category and not metric and has_word(q, "count", "distribution", "breakdown"):
        counts = df[category].value_counts().head(20).rename("Count").to_frame()
        return f"Rows per {category}:", counts

    # Not understood -> the AI will handle it
    return None
