"""
Data normalization module.

Transforms heterogeneous raw data files into a canonical schema compatible with
Phase 3 analytics. Handles column mapping, date extraction, and numeric sanitization.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.logger import debug_watcher

if TYPE_CHECKING:
    from collections.abc import Mapping

# Canonical schema definition
CANONICAL_COLUMNS = [
    "ASIN",
    "SKU",
    "Snapshot_Date",
    "Product_Title",
    "Ordered_Revenue",
    "Ordered_Units",
    "CCOGS",
    "Promotional_Rebates",
    "source_file",
]

CANONICAL_DTYPES = {
    "ASIN": "string",
    "SKU": "string",
    "Snapshot_Date": "datetime64[ns]",
    "Product_Title": "string",
    "Ordered_Revenue": "float64",
    "Ordered_Units": "int64",
    "CCOGS": "float64",
    "Promotional_Rebates": "float64",
    "source_file": "string",
}


def get_canonical_schema() -> dict[str, str]:
    """
    Returns the canonical schema definition as a dictionary of column names to types.

    Returns:
        Dictionary mapping column names to pandas dtype strings.
    """
    return CANONICAL_DTYPES.copy()


# Column mapping dictionary: source column patterns -> canonical column
_COLUMN_MAPPINGS: dict[str, str] = {
    # ASIN mappings
    "ASIN": "ASIN",
    "matched_asin#1.value": "ASIN",
    "asin": "ASIN",
    # SKU mappings
    "SKU": "SKU",
    "sku": "SKU",
    # Product Title mappings
    "Product Title": "Product_Title",
    "Product_Title": "Product_Title",
    "item_name.value": "Product_Title",
    "Title": "Product_Title",
    "Brand Name": "Product_Title",  # Fallback if no title
    # Revenue mappings
    "Product Sales": "Ordered_Revenue",
    "Ordered Product Sales": "Ordered_Revenue",
    "Product GMS": "Ordered_Revenue",
    "Product GMS($)": "Ordered_Revenue",
    "Retail Net OPS": "Ordered_Revenue",
    "All Net OPS": "Ordered_Revenue",
    "3P Net OPS": "Ordered_Revenue",
    "NET_OPS": "Ordered_Revenue",
    # Units mappings
    "Ordered Units": "Ordered_Units",
    "Net Ordered Units": "Ordered_Units",
    "Retail Net Units": "Ordered_Units",
    "All Net Units": "Ordered_Units",
    "3P Net Units": "Ordered_Units",
    "NET_UNITS": "Ordered_Units",
    # CCOGS mappings
    "CCOGS": "CCOGS",
    "CP": "CCOGS",
    "CP($)": "CCOGS",
    "NET_SPOT_CP": "CCOGS",
    # Promotional Rebates mappings
    "Promotional Rebates": "Promotional_Rebates",
    "Deal GMS": "Promotional_Rebates",
    "Deal GMS($)": "Promotional_Rebates",
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map source columns to canonical column names using flexible matching.

    Args:
        df: Source DataFrame with potentially non-standard column names.

    Returns:
        DataFrame with mapped column names (may include unmapped columns).
    """
    result = df.copy()
    mapped_cols = {}

    # Normalize source column names for matching (case-insensitive, strip whitespace)
    source_cols_lower = {col.lower().strip(): col for col in df.columns}

    # Try exact matches first
    for source_pattern, canonical_col in _COLUMN_MAPPINGS.items():
        source_lower = source_pattern.lower().strip()
        if source_lower in source_cols_lower:
            original_col = source_cols_lower[source_lower]
            mapped_cols[original_col] = canonical_col
            continue

        # Try partial matches (contains pattern)
        for orig_col_lower, orig_col in source_cols_lower.items():
            if source_pattern.lower() in orig_col_lower and orig_col not in mapped_cols:
                # Check if this column hasn't been mapped to a more specific target
                if canonical_col not in mapped_cols.values():
                    mapped_cols[orig_col] = canonical_col
                    break

    # Apply mappings
    result = result.rename(columns=mapped_cols)

    return result


def _sanitize_numeric(series: pd.Series) -> pd.Series:
    """
    Sanitize numeric series by removing currency symbols, commas, and handling negatives.

    Args:
        series: Series containing numeric values (may be strings with formatting).

    Returns:
        Series with numeric values (float, with NaN for invalid entries).
    """
    if series.dtype in ["int64", "float64"]:
        return series

    # Convert to string, handle NaN
    str_series = series.astype(str)

    # Replace common non-numeric values
    str_series = str_series.replace(["N/A", "NULL", "null", "None", "nan", ""], "NaN")

    # Remove currency symbols
    str_series = str_series.str.replace(r"[$€AEDUSD]", "", regex=True)

    # Remove commas (thousand separators)
    str_series = str_series.str.replace(",", "")

    # Handle parentheses as negative: (123.45) -> -123.45
    str_series = str_series.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # Remove percentage signs (for percentage fields that need to be converted)
    str_series = str_series.str.replace("%", "")

    # Strip whitespace
    str_series = str_series.str.strip()

    # Convert to numeric
    return pd.to_numeric(str_series, errors="coerce")


def _extract_date_from_filename(filepath: str | Path) -> datetime | None:
    """
    Extract date from filename using common patterns.

    Args:
        filepath: Path to the file.

    Returns:
        Parsed datetime or None if no date found.
    """
    path = Path(filepath)
    filename = path.stem + path.suffix  # Include extension for patterns like _20260106.csv

    # Pattern 1: YYYYMMDD (e.g., 20260106, _20260106_)
    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # Pattern 2: YYYY-MM-DD (e.g., 2026-01-06)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass

    # Pattern 3: YYYYMM (e.g., 202601) - use first day of month
    match = re.search(r"(\d{4})(\d{2})(?!\d{2})", filename)
    if match and len(match.group(0)) == 6:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), 1)
        except ValueError:
            pass

    return None


def _extract_snapshot_date(
    df: pd.DataFrame,
    source_file: str | Path,
) -> pd.Series:
    """
    Extract snapshot dates for each row from various sources.

    Priority:
    1. Date column in DataFrame (if exists)
    2. Date from filename
    3. Date from file metadata (modification time)

    Args:
        df: Source DataFrame.
        source_file: Path to source file.

    Returns:
        Series of datetime objects (one per row).
    """
    path = Path(source_file)

    # Try to find a date column in the DataFrame
    date_cols = [
        col
        for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in ["date", "snapshot", "period", "day", "week"]
        )
    ]

    if date_cols:
        # Use first date column found
        date_col = date_cols[0]
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if dates.notna().any():
                return dates
        except Exception:
            pass

    # Extract from filename
    file_date = _extract_date_from_filename(source_file)
    if file_date:
        return pd.Series([file_date] * len(df), index=df.index)

    # Fallback to file modification time
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return pd.Series([mtime] * len(df), index=df.index)
    except Exception:
        # Final fallback: current time
        return pd.Series([datetime.now()] * len(df), index=df.index)


def _detect_file_type(df: pd.DataFrame, source_file: str | Path) -> str:
    """
    Detect file type based on filename patterns and column structure.

    Args:
        df: Source DataFrame.
        source_file: Path to source file.

    Returns:
        File type: "daily", "historical", "weekly", or "unknown".
    """
    path = Path(source_file)
    filename_lower = path.name.lower()

    if "daily" in filename_lower or "DailyOrdersSummary" in filename_lower:
        return "daily"
    if "historical" in filename_lower or "T12M" in filename_lower or "T12M_" in path.name:
        return "historical"
    if "weekly" in filename_lower or "W49" in filename_lower or "Week" in filename_lower:
        return "weekly"
    if "ratings" in filename_lower:
        return "ratings"

    # Check column structure
    cols_lower = [col.lower() for col in df.columns]
    if any("week" in col for col in cols_lower):
        return "weekly"
    if any("t12m" in col for col in cols_lower):
        return "historical"

    return "unknown"


def _handle_daily_file(df: pd.DataFrame, source_file: str | Path) -> pd.DataFrame:
    """
    Transform daily file format (time-series with dates as columns) into canonical format.

    Daily files have dates as column headers and need to be unpivoted.
    Note: Daily files are typically aggregated and may not have ASIN-level data.

    Args:
        df: Source DataFrame (may have metadata rows at top).
        source_file: Path to source file.

    Returns:
        DataFrame in canonical format.
    """
    # Find the row that contains "Period" (this is the header row)
    header_row_idx = None
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).lower() if len(row) > 0 else ""
        if "period" in first_val:
            header_row_idx = idx
            break

    if header_row_idx is None:
        # Try to find row with "Merchant Type" or "Metric Type"
        for idx, row in df.iterrows():
            row_str = " ".join([str(v).lower() for v in row.values[:5]])
            if "merchant type" in row_str or "metric type" in row_str:
                header_row_idx = idx
                break

    if header_row_idx is not None:
        # Use the header row as column names
        df_clean = df.iloc[header_row_idx + 1 :].copy()
        df_clean.columns = df.iloc[header_row_idx].values
        df_clean = df_clean.reset_index(drop=True)
    else:
        # Fallback: use existing columns
        df_clean = df.copy()

    # Find date columns (columns that look like dates)
    date_cols = []
    id_cols = []

    for col in df_clean.columns:
        col_str = str(col).lower()
        # Check if column is a date (contains month names or day abbreviations)
        if any(
            keyword in col_str
            for keyword in ["dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"]
        ) or re.match(r"^(mon|tue|wed|thu|fri|sat|sun)", col_str):
            date_cols.append(col)
        elif col_str in ["period", "merchant type", "metric type"]:
            id_cols.append(col)

    if not date_cols:
        # No date columns found, return empty with canonical structure
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Filter to rows with relevant metric types
    metric_col = None
    for col in df_clean.columns:
        if "metric type" in str(col).lower():
            metric_col = col
            break

    if metric_col:
        df_clean = df_clean[df_clean[metric_col].isin(["NET_OPS", "NET_UNITS", "NET_GMS", "NET_SPOT_CP"])]

    # Unpivot: melt date columns into rows
    id_vars = [col for col in df_clean.columns if col not in date_cols]
    melted = df_clean.melt(
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="date_str",
        value_name="value",
    )

    # Parse dates from column names
    melted["Snapshot_Date"] = pd.to_datetime(melted["date_str"], errors="coerce")
    melted = melted[melted["Snapshot_Date"].notna()]  # Remove rows where date parsing failed

    if melted.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Group by date and metric type to aggregate
    result_rows = []
    for date in melted["Snapshot_Date"].unique():
        date_data = melted[melted["Snapshot_Date"] == date]

        row = {
            "Snapshot_Date": date,
            "ASIN": "",  # Daily files typically don't have ASIN-level data
            "SKU": "",
            "Product_Title": "",
            "Ordered_Revenue": 0.0,
            "Ordered_Units": 0,
            "CCOGS": 0.0,
            "Promotional_Rebates": 0.0,
            "source_file": str(source_file),
        }

        # Map metric types to canonical columns
        if metric_col:
            for metric_type in date_data[metric_col].unique():
                metric_data = date_data[date_data[metric_col] == metric_type]
                value = _sanitize_numeric(metric_data["value"]).sum()

                if metric_type in ["NET_OPS", "NET_GMS"]:
                    row["Ordered_Revenue"] += float(value) if pd.notna(value) else 0.0
                elif metric_type == "NET_UNITS":
                    row["Ordered_Units"] += int(value) if pd.notna(value) else 0
                elif metric_type == "NET_SPOT_CP":
                    row["CCOGS"] += float(value) if pd.notna(value) else 0.0

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    return result


def _handle_historical_file(df: pd.DataFrame, source_file: str | Path) -> pd.DataFrame:
    """
    Transform historical file format (ASIN-level aggregated) into canonical format.

    Args:
        df: Source DataFrame.
        source_file: Path to source file.

    Returns:
        DataFrame in canonical format.
    """
    result = df.copy()

    # Map columns
    result = _map_columns(result)

    # Extract snapshot date
    result["Snapshot_Date"] = _extract_snapshot_date(result, source_file)

    # Handle Product GMS with time period suffix (e.g., "Product GMS($)(T12M )")
    # Map to Ordered_Revenue if not already mapped
    if "Ordered_Revenue" not in result.columns:
        gms_cols = [col for col in result.columns if "gms" in col.lower() and "product" in col.lower()]
        if gms_cols:
            result["Ordered_Revenue"] = _sanitize_numeric(result[gms_cols[0]])

    # Handle Net Ordered Units with time period suffix
    if "Ordered_Units" not in result.columns:
        units_cols = [col for col in result.columns if "ordered units" in col.lower() or "net ordered units" in col.lower()]
        if units_cols:
            result["Ordered_Units"] = _sanitize_numeric(result[units_cols[0]]).fillna(0).astype("int64")

    # Handle CP (Customer Price/Cost) - map to CCOGS
    if "CCOGS" not in result.columns:
        cp_cols = [col for col in result.columns if "cp($)" in col.lower() or (col.lower().startswith("cp") and "t12m" in col.lower())]
        if cp_cols:
            result["CCOGS"] = _sanitize_numeric(result[cp_cols[0]])

    # Handle Deal GMS as Promotional_Rebates
    if "Promotional_Rebates" not in result.columns:
        deal_cols = [col for col in result.columns if "deal gms" in col.lower()]
        if deal_cols:
            result["Promotional_Rebates"] = _sanitize_numeric(result[deal_cols[0]])

    # Sanitize numeric columns
    for col in ["Ordered_Revenue", "Ordered_Units", "CCOGS", "Promotional_Rebates"]:
        if col in result.columns:
            if col == "Ordered_Units":
                result[col] = _sanitize_numeric(result[col]).fillna(0).astype("int64")
            else:
                result[col] = _sanitize_numeric(result[col]).fillna(0.0).astype("float64")
        else:
            # Set default values
            if col in ["Ordered_Revenue", "CCOGS", "Promotional_Rebates"]:
                result[col] = 0.0
            elif col == "Ordered_Units":
                result[col] = 0

    # Ensure ASIN exists (critical for ASIN-level analysis)
    if "ASIN" not in result.columns:
        # Try to find ASIN in other columns
        asin_cols = [col for col in result.columns if "asin" in col.lower()]
        if asin_cols:
            result["ASIN"] = result[asin_cols[0]].astype("string")
        else:
            result["ASIN"] = ""

    # Ensure SKU exists
    if "SKU" not in result.columns:
        result["SKU"] = result.get("ASIN", "")  # Use ASIN as fallback for SKU

    # Ensure Product_Title exists
    if "Product_Title" not in result.columns:
        title_cols = [col for col in result.columns if "title" in col.lower() or "name" in col.lower()]
        if title_cols:
            result["Product_Title"] = result[title_cols[0]].astype("string")
        else:
            result["Product_Title"] = ""

    # Add source_file
    result["source_file"] = str(source_file)

    return result


def _handle_weekly_file(df: pd.DataFrame, source_file: str | Path) -> pd.DataFrame:
    """
    Transform weekly file format (week-specific columns) into canonical format.

    Args:
        df: Source DataFrame.
        source_file: Path to source file.

    Returns:
        DataFrame in canonical format.
    """
    # Similar to historical but may need to unpivot week columns
    # For now, treat similar to historical
    return _handle_historical_file(df, source_file)


@debug_watcher
def normalize_data(
    df: pd.DataFrame,
    source_file: str | Path,
    file_type: str | None = None,
) -> pd.DataFrame:
    """
    Normalize a DataFrame to canonical schema.

    Args:
        df: Source DataFrame with potentially non-standard structure.
        source_file: Path to source file (for date extraction and lineage).
        file_type: Optional file type hint ("daily", "historical", "weekly", "unknown").

    Returns:
        DataFrame matching canonical schema with columns:
        ASIN, SKU, Snapshot_Date, Product_Title, Ordered_Revenue, Ordered_Units,
        CCOGS, Promotional_Rebates, source_file.
    """
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    # Detect file type if not provided
    if file_type is None:
        file_type = _detect_file_type(df, source_file)

    # Handle different file types
    if file_type == "daily":
        normalized = _handle_daily_file(df, source_file)
    elif file_type == "historical":
        normalized = _handle_historical_file(df, source_file)
    elif file_type == "weekly":
        normalized = _handle_weekly_file(df, source_file)
    else:
        # Generic normalization
        normalized = _handle_historical_file(df, source_file)

    # Ensure all canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in normalized.columns:
            if col in ["Ordered_Revenue", "CCOGS", "Promotional_Rebates"]:
                normalized[col] = 0.0
            elif col == "Ordered_Units":
                normalized[col] = 0
            elif col == "Snapshot_Date":
                normalized[col] = _extract_snapshot_date(normalized, source_file)
            elif col == "source_file":
                normalized[col] = str(source_file)
            else:
                normalized[col] = ""

    # Ensure correct dtypes
    for col, dtype in CANONICAL_DTYPES.items():
        if col in normalized.columns:
            if dtype == "datetime64[ns]":
                normalized[col] = pd.to_datetime(normalized[col], errors="coerce")
            elif dtype == "int64":
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0).astype("int64")
            elif dtype == "float64":
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0).astype("float64")
            elif dtype == "string":
                normalized[col] = normalized[col].astype("string").fillna("")

    # Select only canonical columns and reorder in canonical order
    # All columns should exist after the ensure step above
    result = normalized[CANONICAL_COLUMNS].copy() if not normalized.empty else pd.DataFrame(columns=CANONICAL_COLUMNS)

    return result
