"""
Data Transformer Module - Mapping and Cleaning

Transforms raw performance data into the canonical WBR schema:
- Column mapping using parsed headers
- Numeric value cleaning (parentheses, currency, percentages)
- Wide-to-long format conversion (unpivoting weekly data)
- Vendor join logic (ASIN -> Vendor Code -> Vendor Name)
- WoW change calculations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np

from src.config import COLUMN_MAPPING, CRITICAL_METRICS
from src.header_parser import HeaderParser, ParseResult
from src.value_parser import parse_numeric_series, parse_numeric_value
from src.temporal_calcs import add_temporal_metrics

if TYPE_CHECKING:
    from src.data_loader import HistoryHandler


def clean_numeric_value(value: Any) -> float | None:
    """
    Clean a single numeric value.

    Uses shared parser with currency/percent/bps handling.
    """
    return parse_numeric_value(value)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Clean a pandas Series of numeric values.

    Args:
        series: Series containing numeric values.

    Returns:
        Series with cleaned numeric values.
    """
    return parse_numeric_series(series)


def map_columns(
    df: pd.DataFrame,
    parse_results: list[ParseResult],
    week_to_process: int | None = None
) -> pd.DataFrame:
    """
    Map source columns to canonical WBR column names.
    
    Args:
        df: Source DataFrame.
        parse_results: List of ParseResult objects from header parsing.
        week_to_process: Specific week to extract. If None, uses most recent.
    
    Returns:
        DataFrame with mapped columns.
    """
    # Build mapping from raw header to parse result
    header_map = {r.raw_header: r for r in parse_results}
    
    # Find all unique weeks in the data
    weeks = set()
    for r in parse_results:
        if r.week_number is not None:
            weeks.add(r.week_number)
    
    # Determine which week to process
    if week_to_process is None and weeks:
        week_to_process = max(weeks)
    
    # Build column mapping for the target week
    result_data = {}
    
    # First, copy static columns
    for col in df.columns:
        if col in header_map:
            r = header_map[col]
            if r.matched_pattern == "static_column":
                result_data[r.mapped_wbr_column] = df[col]
    
    # Then, process time-specific columns for target week
    for col in df.columns:
        if col in header_map:
            r = header_map[col]
            
            # Skip static columns (already processed)
            if r.matched_pattern == "static_column":
                continue
            
            # Only process columns for target week
            if r.week_number is not None and r.week_number != week_to_process:
                continue
            
            # Map to WBR column
            if r.mapped_wbr_column and r.mapped_wbr_column not in result_data:
                series = clean_numeric_series(df[col])
                if r.is_percentage and series.notna().any():
                    max_abs = series.dropna().abs().max()
                    if max_abs > 1:
                        series = series / 100.0
                result_data[r.mapped_wbr_column] = series
    
    result = pd.DataFrame(result_data)
    
    # Add week number
    if week_to_process:
        result["Week_Number"] = week_to_process
    
    return result


def unpivot_weekly_data(
    df: pd.DataFrame,
    parse_results: list[ParseResult]
) -> pd.DataFrame:
    """
    Convert wide format (W49, W50...) to long format with one row per ASIN per week.
    
    Args:
        df: Source DataFrame in wide format.
        parse_results: List of ParseResult objects from header parsing.
    
    Returns:
        DataFrame in long format.
    """
    # Build mapping from raw header to parse result
    header_map = {r.raw_header: r for r in parse_results}
    
    # Find all unique weeks
    weeks = sorted(set(r.week_number for r in parse_results if r.week_number is not None))
    
    if not weeks:
        # No weekly columns found, return as-is with single row mapping
        return map_columns(df, parse_results)
    
    # Process each week and combine
    week_dfs = []
    for week in weeks:
        week_df = map_columns(df, parse_results, week_to_process=week)
        if not week_df.empty:
            week_dfs.append(week_df)
    
    if not week_dfs:
        return pd.DataFrame()
    
    return pd.concat(week_dfs, ignore_index=True)


def join_with_reference(
    df: pd.DataFrame,
    vendor_map: pd.DataFrame,
    asin_selection: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Join performance data with vendor reference data.
    
    Args:
        df: Performance DataFrame with ASIN column.
        vendor_map: DataFrame with vendor_code and vendor_name columns.
        asin_selection: Optional DataFrame to filter ASINs.
    
    Returns:
        DataFrame with Vendor_Code and Vendor_Name added.
    """
    result = df.copy()
    
    # Ensure ASIN column exists
    asin_col = None
    for col in ["ASIN", "asin"]:
        if col in result.columns:
            asin_col = col
            break
    
    if asin_col is None:
        return result
    
    # Normalize ASIN column name
    if asin_col != "ASIN":
        result = result.rename(columns={asin_col: "ASIN"})
    
    # Filter by ASIN selection if provided
    if asin_selection is not None and not asin_selection.empty:
        if "ASIN" in asin_selection.columns:
            valid_asins = set(asin_selection["ASIN"].astype(str))
            result = result[result["ASIN"].astype(str).isin(valid_asins)]
    
    # Join with vendor map
    # The vendor_map has vendor codes, but we need to map ASINs to vendor codes
    # This typically requires a separate ASIN->vendor mapping
    # For now, we'll try to infer from filename or add placeholder columns
    
    if "Vendor_Code" not in result.columns:
        result["Vendor_Code"] = ""
    if "Vendor_Name" not in result.columns:
        result["Vendor_Name"] = ""
    
    # If vendor_map has the mapping, apply it
    if not vendor_map.empty:
        # Check if vendor_map has ASIN column for direct mapping
        if "ASIN" in vendor_map.columns:
            vendor_cols = ["vendor_code", "vendor_name"] if "vendor_code" in vendor_map.columns else ["Vendor_Code", "Vendor_Name"]
            result = result.merge(
                vendor_map[["ASIN"] + [c for c in vendor_cols if c in vendor_map.columns]],
                on="ASIN",
                how="left"
            )
    
    return result


def calculate_wow_change(
    df: pd.DataFrame,
    history_handler: "HistoryHandler | None" = None,
    current_week: int | None = None,
    current_year: int | None = None
) -> pd.DataFrame:
    """
    Calculate week-over-week changes for key metrics.
    
    Args:
        df: DataFrame with current week's data.
        history_handler: HistoryHandler for retrieving previous week data.
        current_week: Current week number.
        current_year: Current year.
    
    Returns:
        DataFrame with WoW change columns added.
    """
    from datetime import datetime
    
    result = df.copy()
    
    if current_year is None:
        current_year = datetime.now().year
    
    if current_week is None:
        if "Week_Number" in result.columns:
            current_week = result["Week_Number"].iloc[0] if len(result) > 0 else None
    
    if current_week is None:
        return result
    
    # Metrics to calculate WoW for
    wow_metrics = [
        ("Ordered_Revenue", "Revenue_WoW"),
        ("Ordered_Units", "Units_WoW"),
        ("Glance_Views", "GlanceViews_WoW"),
        ("Net_PPM", "NetPPM_WoW"),
        ("Average_Selling_Price", "ASP_WoW"),
    ]
    
    # If no history handler, initialize WoW columns with NaN
    if history_handler is None:
        for current_col, wow_col in wow_metrics:
            if current_col in result.columns:
                result[wow_col] = np.nan
        return result
    
    # Get ASINs and metrics from current data
    if "ASIN" not in result.columns:
        return result
    
    asins = result["ASIN"].tolist()
    metric_names = [m[0] for m in wow_metrics if m[0] in result.columns]
    
    if not metric_names:
        return result
    
    # Get previous week data in batch
    prev_data = history_handler.get_previous_week_batch(
        asins, metric_names, current_week, current_year
    )
    
    # Create pivot of previous data for easy lookup
    if not prev_data.empty:
        prev_pivot = prev_data.pivot(
            index="asin",
            columns="metric_name",
            values="previous_value"
        ).reset_index()
        prev_pivot.columns.name = None
        prev_pivot = prev_pivot.rename(columns={"asin": "ASIN"})
        
        # Merge with current data
        result = result.merge(
            prev_pivot,
            on="ASIN",
            how="left",
            suffixes=("", "_prev")
        )
        
        # Calculate WoW percentages (decimal form)
        for current_col, wow_col in wow_metrics:
            prev_col = f"{current_col}_prev"
            if current_col in result.columns and prev_col in result.columns:
                # WoW = (Current - Previous) / Previous
                result[wow_col] = np.where(
                    (result[prev_col].notna()) & (result[prev_col] != 0),
                    (result[current_col] - result[prev_col]) / result[prev_col].abs(),
                    np.nan,
                )
                # Drop the _prev column
                result = result.drop(columns=[prev_col])
    else:
        # No previous data, set WoW to NaN
        for current_col, wow_col in wow_metrics:
            if current_col in result.columns:
                result[wow_col] = np.nan
    
    return result


def segment_by_week(wbr_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Split DataFrame by Week_Number, return dict of {week: df}.
    
    Args:
        wbr_df: DataFrame with Week_Number column.
    
    Returns:
        Dictionary mapping week numbers to DataFrames.
    """
    if "Week_Number" not in wbr_df.columns:
        # If no Week_Number column, return single entry with all data
        return {None: wbr_df}
    
    result = {}
    for week in wbr_df["Week_Number"].unique():
        if pd.notna(week):
            week_df = wbr_df[wbr_df["Week_Number"] == week].copy()
            if not week_df.empty:
                result[int(week)] = week_df
    
    return result


def transform_raw_to_wbr(
    raw_df: pd.DataFrame,
    parse_results: list[ParseResult],
    vendor_map: pd.DataFrame | None = None,
    asin_selection: pd.DataFrame | None = None,
    history_handler: "HistoryHandler | None" = None,
    target_week: int | None = None,
    current_year: int | None = None,
    unpivot: bool = True
) -> pd.DataFrame:
    """
    Full transformation pipeline from raw data to WBR schema.
    
    Args:
        raw_df: Raw performance data.
        parse_results: List of ParseResult objects from header parsing.
        vendor_map: Vendor reference data.
        asin_selection: ASIN filter list.
        history_handler: Handler for WoW calculations.
        target_week: Specific week to process (None for most recent).
        current_year: Optional year for period calculations.
        unpivot: Whether to unpivot weekly columns.
    
    Returns:
        Transformed DataFrame in WBR schema.
    """
    # Step 1: Map columns and optionally unpivot
    if unpivot:
        df = unpivot_weekly_data(raw_df, parse_results)
    else:
        df = map_columns(raw_df, parse_results, target_week)
    
    if df.empty:
        return df
    
    # Step 2: Join with vendor reference
    if vendor_map is not None:
        df = join_with_reference(df, vendor_map, asin_selection)
    
    # Step 3: Calculate WoW changes
    if history_handler is not None:
        current_week = target_week or (df["Week_Number"].iloc[0] if "Week_Number" in df.columns and len(df) > 0 else None)
        df = calculate_wow_change(df, history_handler, current_week, current_year)

    # Step 3b: Add temporal deltas and rolling windows where possible
    df = add_temporal_metrics(
        df,
        group_cols=["ASIN"],
        week_col="Week_Number",
        base_year=current_year,
        overwrite=False,
    )
    
    # Step 4: Ensure critical metrics exist
    for metric in CRITICAL_METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
    
    return df


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Data Transformer Module Test")
    print("=" * 60)
    
    # Test numeric cleaning
    print("\n1. Testing numeric cleaning...")
    test_values = [
        "123.45",
        "(15.23)",
        "$1,234.56",
        "45.6%",
        "N/A",
        "",
        None,
        "(1,234.56)",
    ]
    for v in test_values:
        result = clean_numeric_value(v)
        print(f"   '{v}' -> {result}")
    
    # Test with real data
    print("\n2. Testing with real data...")
    from src.data_loader import detect_latest_file, load_raw_data, load_reference_data
    from src.header_parser import HeaderParser
    from src.config import DROPZONE_PATH
    
    latest = detect_latest_file(DROPZONE_PATH)
    if latest:
        raw_df, metadata = load_raw_data(latest)
        print(f"   Loaded: {metadata['filename']}")
        
        # Parse headers
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        print(f"   Parsed {len(parse_results)} headers")
        
        # Transform
        ref_data = load_reference_data()
        transformed = transform_raw_to_wbr(
            raw_df,
            parse_results,
            vendor_map=ref_data["vendor_map"],
            asin_selection=ref_data["asin_selection"],
            unpivot=True
        )
        
        print(f"   Transformed: {len(transformed)} rows, {len(transformed.columns)} columns")
        print(f"   Columns: {list(transformed.columns)[:10]}...")
        
        if "Ordered_Revenue" in transformed.columns:
            print(f"   Revenue range: {transformed['Ordered_Revenue'].min():.2f} - {transformed['Ordered_Revenue'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("Data Transformer Module Test Complete")
    print("=" * 60)
