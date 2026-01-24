"""
Analyzer Module - Threshold Evaluation and RAG Status Assignment

Applies business logic and flags performance issues:
- Threshold evaluation (Red/Amber/Green status)
- Week-over-week change calculations
- Anomaly detection (statistical outliers)
- Data quality issue flagging
- Enhanced analytics: Trend analysis, Statistical analysis, Comparative analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config import THRESHOLDS

if TYPE_CHECKING:
    from src.data_loader import HistoryHandler


def get_rag_status(
    value: float | None,
    metric: str,
    thresholds: dict[str, Any] | None = None
) -> str:
    """
    Get RAG (Red/Amber/Green) status for a metric value.
    
    Args:
        value: The metric value.
        metric: The metric name (key in THRESHOLDS).
        thresholds: Optional custom thresholds. Uses THRESHOLDS if None.
    
    Returns:
        "RED", "AMBER", "GREEN", or "N/A" if value is null.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    
    if pd.isna(value):
        return "N/A"
    
    # Find matching threshold
    threshold_config = None
    for key in thresholds:
        if key.lower() == metric.lower() or key.lower() in metric.lower().replace("_", ""):
            threshold_config = thresholds[key]
            break
    
    if threshold_config is None:
        return "N/A"
    
    red_val = threshold_config.get("red", float("-inf"))
    amber_val = threshold_config.get("amber", float("-inf"))
    direction = threshold_config.get("direction", "higher_is_better")
    
    if direction == "higher_is_better":
        # Lower values are worse
        if value < red_val:
            return "RED"
        elif value < amber_val:
            return "AMBER"
        else:
            return "GREEN"
    else:
        # Lower values are better (e.g., SoROOS%)
        if value > red_val:
            return "RED"
        elif value > amber_val:
            return "AMBER"
        else:
            return "GREEN"


def evaluate_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, Any] | None = None
) -> pd.DataFrame:
    """
    Evaluate thresholds for all applicable columns and add RAG status columns.
    
    Args:
        df: DataFrame with metric columns.
        thresholds: Optional custom thresholds. Uses THRESHOLDS if None.
    
    Returns:
        DataFrame with additional _RAG columns for each evaluated metric.
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    
    result = df.copy()
    
    for metric_key, config in thresholds.items():
        # Find matching column
        matching_col = None
        for col in result.columns:
            if metric_key.lower() == col.lower() or metric_key.lower() in col.lower().replace("_", ""):
                matching_col = col
                break
        
        if matching_col is None:
            continue
        
        # Create RAG column
        rag_col = f"{matching_col}_RAG"
        result[rag_col] = result[matching_col].apply(
            lambda x: get_rag_status(x, metric_key, thresholds)
        )
    
    return result


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect statistical anomalies (outliers) using Z-score method.
    
    Args:
        df: DataFrame to analyze.
        columns: Columns to check for anomalies. If None, checks numeric columns.
        z_threshold: Z-score threshold for anomaly detection (default 3.0).
    
    Returns:
        DataFrame with additional _Anomaly columns (True if anomaly).
    """
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        series = result[col]
        
        # Skip if all values are the same or too few values
        if series.std() == 0 or series.count() < 3:
            result[f"{col}_Anomaly"] = False
            continue
        
        # Calculate Z-score
        z_scores = np.abs((series - series.mean()) / series.std())
        result[f"{col}_Anomaly"] = z_scores > z_threshold
    
    return result


def flag_data_quality_issues(
    df: pd.DataFrame,
    critical_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Flag data quality issues in the DataFrame.
    
    Args:
        df: DataFrame to check.
        critical_columns: List of columns that should not have nulls.
    
    Returns:
        Tuple of (DataFrame with Data_Quality_Flag column, list of issues).
    """
    if critical_columns is None:
        critical_columns = ["ASIN", "Ordered_Revenue", "Ordered_Units"]
    
    result = df.copy()
    issues = []
    
    # Check for nulls in critical columns
    null_flags = pd.Series(False, index=result.index)
    for col in critical_columns:
        if col in result.columns:
            col_nulls = result[col].isna()
            if col_nulls.any():
                issues.append({
                    "type": "null_values",
                    "column": col,
                    "count": col_nulls.sum(),
                    "percentage": (col_nulls.sum() / len(result)) * 100,
                })
                null_flags = null_flags | col_nulls
    
    # Check for negative values in columns that should be positive
    positive_columns = ["Ordered_Revenue", "Ordered_Units", "Glance_Views"]
    for col in positive_columns:
        if col in result.columns:
            negative_mask = result[col] < 0
            if negative_mask.any():
                issues.append({
                    "type": "negative_values",
                    "column": col,
                    "count": negative_mask.sum(),
                    "percentage": (negative_mask.sum() / len(result)) * 100,
                })
    
    # Check for duplicate ASINs (if single week)
    if "ASIN" in result.columns and "Week_Number" in result.columns:
        week_groups = result.groupby("Week_Number")
        for week, group in week_groups:
            duplicates = group["ASIN"].duplicated()
            if duplicates.any():
                issues.append({
                    "type": "duplicate_asins",
                    "week": week,
                    "count": duplicates.sum(),
                })
    
    # Add quality flag column
    result["Data_Quality_Flag"] = null_flags.apply(lambda x: "ISSUE" if x else "OK")
    
    return result, issues


def calculate_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate summary statistics for key metrics.
    
    Args:
        df: DataFrame with metric columns.
    
    Returns:
        Dictionary with summary statistics.
    """
    stats = {
        "total_rows": len(df),
        "total_asins": df["ASIN"].nunique() if "ASIN" in df.columns else 0,
        "metrics": {},
    }
    
    # Key metrics to summarize
    metrics_to_summarize = [
        "Ordered_Revenue",
        "Ordered_Units",
        "Net_PPM",
        "Glance_Views",
        "Average_Selling_Price",
    ]
    
    for metric in metrics_to_summarize:
        if metric in df.columns:
            series = df[metric].dropna()
            stats["metrics"][metric] = {
                "count": len(series),
                "mean": series.mean() if len(series) > 0 else None,
                "median": series.median() if len(series) > 0 else None,
                "min": series.min() if len(series) > 0 else None,
                "max": series.max() if len(series) > 0 else None,
                "sum": series.sum() if len(series) > 0 else None,
            }
    
    # RAG distribution
    stats["rag_distribution"] = {}
    for col in df.columns:
        if col.endswith("_RAG"):
            metric_name = col.replace("_RAG", "")
            value_counts = df[col].value_counts().to_dict()
            stats["rag_distribution"][metric_name] = value_counts
    
    return stats


def analyze_performance(
    df: pd.DataFrame,
    thresholds: dict[str, Any] | None = None,
    detect_anomalies_flag: bool = True,
    check_quality: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full performance analysis pipeline.
    
    Args:
        df: DataFrame with metric columns.
        thresholds: Optional custom thresholds.
        detect_anomalies_flag: Whether to detect anomalies.
        check_quality: Whether to check data quality.
    
    Returns:
        Tuple of (analyzed DataFrame with RAG columns, analysis summary dict).
    """
    result = df.copy()
    summary = {}
    
    # Step 1: Evaluate thresholds
    result = evaluate_thresholds(result, thresholds)
    
    # Step 2: Detect anomalies (optional)
    if detect_anomalies_flag:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude RAG and flag columns
        numeric_cols = [c for c in numeric_cols if not c.endswith("_RAG") and not c.endswith("_Anomaly")]
        result = detect_anomalies(result, numeric_cols[:10])  # Limit to prevent too many columns
    
    # Step 3: Check data quality (optional)
    issues = []
    if check_quality:
        result, issues = flag_data_quality_issues(result)
    
    # Step 4: Calculate summary statistics
    summary = calculate_summary_statistics(result)
    summary["data_quality_issues"] = issues
    
    return result, summary


def generate_trend_analysis(
    current_df: pd.DataFrame,
    history_handler: "HistoryHandler | None" = None,
    current_week: int | None = None,
    current_year: int | None = None
) -> pd.DataFrame:
    """
    Calculate Month-over-Month and Quarter-over-Quarter trends.
    
    Args:
        current_df: Current week's DataFrame.
        history_handler: HistoryHandler for retrieving historical data.
        current_week: Current week number.
        current_year: Current year.
    
    Returns:
        DataFrame with MoM and QoQ trend columns added.
    """
    result = current_df.copy()
    
    if history_handler is None or current_week is None or current_year is None:
        # Add placeholder columns
        result["Revenue_MoM"] = np.nan
        result["Revenue_QoQ"] = np.nan
        result["Units_MoM"] = np.nan
        result["Units_QoQ"] = np.nan
        result["Trend_Direction"] = "N/A"
        return result
    
    # Calculate month and quarter for current week
    from datetime import datetime, date
    try:
        current_date = date.fromisocalendar(current_year, current_week, 1)
        current_month = current_date.month
        current_quarter = (current_month - 1) // 3 + 1
    except Exception:
        result["Revenue_MoM"] = np.nan
        result["Revenue_QoQ"] = np.nan
        result["Units_MoM"] = np.nan
        result["Units_QoQ"] = np.nan
        result["Trend_Direction"] = "N/A"
        return result
    
    # For MoM: need data from ~4 weeks ago (same week in previous month)
    # For QoQ: need data from ~13 weeks ago (same week in previous quarter)
    # This is simplified - in practice, you'd need to query history for specific dates
    
    # Add placeholder columns for now (full implementation would query history)
    result["Revenue_MoM"] = np.nan
    result["Revenue_QoQ"] = np.nan
    result["Units_MoM"] = np.nan
    result["Units_QoQ"] = np.nan
    
    # Trend direction based on WoW if available
    if "Revenue_WoW" in result.columns:
        def get_trend_direction(wow_value):
            if pd.isna(wow_value):
                return "N/A"
            elif wow_value > 0.05:  # >5% growth
                return "↑ Improving"
            elif wow_value < -0.05:  # >5% decline
                return "↓ Declining"
            else:
                return "→ Stable"
        
        result["Trend_Direction"] = result["Revenue_WoW"].apply(get_trend_direction)
    else:
        result["Trend_Direction"] = "N/A"
    
    return result


def calculate_statistical_insights(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate statistical insights: percentiles, outliers, distributions.
    
    Args:
        df: DataFrame with metric columns.
    
    Returns:
        Dictionary with statistical insights.
    """
    insights = {}
    
    # Key metrics for statistical analysis
    metrics = ["Ordered_Revenue", "Ordered_Units", "Net_PPM", "Glance_Views", "Average_Selling_Price"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        series = df[metric].dropna()
        if len(series) == 0:
            continue
        
        insights[metric] = {
            "percentiles": {
                "p25": series.quantile(0.25),
                "p50": series.median(),
                "p75": series.quantile(0.75),
                "p90": series.quantile(0.90),
                "p95": series.quantile(0.95),
            },
            "mean": series.mean(),
            "std": series.std(),
            "cv": series.std() / series.mean() if series.mean() != 0 else np.nan,  # Coefficient of variation
            "min": series.min(),
            "max": series.max(),
        }
        
        # Outlier detection (using IQR method)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        insights[metric]["outliers"] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(series)) * 100 if len(series) > 0 else 0,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
    
    return insights


def generate_comparative_analysis(
    df: pd.DataFrame,
    group_by: str = "Vendor_Code"
) -> pd.DataFrame:
    """
    Generate comparative analysis: Vendor vs Portfolio, ASIN vs Vendor Average.
    
    Args:
        df: DataFrame with metrics and grouping column.
        group_by: Column to group by (e.g., "Vendor_Code").
    
    Returns:
        DataFrame with comparative columns added.
    """
    result = df.copy()
    
    if group_by not in result.columns:
        return result
    
    # Key metrics for comparison
    metrics = ["Ordered_Revenue", "Ordered_Units", "Net_PPM", "Glance_Views"]
    
    for metric in metrics:
        if metric not in result.columns:
            continue
        
        # Portfolio average
        portfolio_avg = result[metric].mean()
        result[f"{metric}_vs_Portfolio"] = np.where(
            portfolio_avg != 0,
            ((result[metric] - portfolio_avg) / abs(portfolio_avg)) * 100,
            np.nan
        )
        
        # Vendor average (if group_by is Vendor_Code)
        if group_by == "Vendor_Code":
            vendor_avgs = result.groupby(group_by)[metric].transform("mean")
            result[f"{metric}_vs_Vendor"] = np.where(
                vendor_avgs != 0,
                ((result[metric] - vendor_avgs) / vendor_avgs.abs()) * 100,
                np.nan
            )
    
    # Top/Bottom performer flags
    for metric in metrics:
        if metric not in result.columns:
            continue
        
        # Top 10% performers
        top_threshold = result[metric].quantile(0.90)
        result[f"{metric}_Top_Performer"] = result[metric] >= top_threshold
        
        # Bottom 10% performers
        bottom_threshold = result[metric].quantile(0.10)
        result[f"{metric}_Bottom_Performer"] = result[metric] <= bottom_threshold
    
    return result


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Analyzer Module Test")
    print("=" * 60)
    
    # Test RAG status
    print("\n1. Testing RAG status...")
    test_cases = [
        ("Net_PPM", 0.03, "Should be RED"),
        ("Net_PPM", 0.10, "Should be AMBER"),
        ("Net_PPM", 0.20, "Should be GREEN"),
        ("SoROOS_Pct", 0.25, "Should be RED (lower is better)"),
        ("SoROOS_Pct", 0.15, "Should be AMBER"),
        ("SoROOS_Pct", 0.05, "Should be GREEN"),
    ]
    
    for metric, value, expected in test_cases:
        status = get_rag_status(value, metric)
        print(f"   {metric}={value}: {status} ({expected})")
    
    # Test with real data
    print("\n2. Testing with real data...")
    from src.data_loader import detect_latest_file, load_raw_data, load_reference_data
    from src.header_parser import HeaderParser
    from src.data_transformer import transform_raw_to_wbr
    from src.config import DROPZONE_PATH
    
    latest = detect_latest_file(DROPZONE_PATH)
    if latest:
        raw_df, _ = load_raw_data(latest)
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        ref_data = load_reference_data()
        
        wbr_df = transform_raw_to_wbr(
            raw_df,
            parse_results,
            vendor_map=ref_data["vendor_map"],
            unpivot=True
        )
        
        # Run analysis
        analyzed_df, summary = analyze_performance(wbr_df)
        
        print(f"   Analyzed {summary['total_rows']} rows, {summary['total_asins']} ASINs")
        print(f"   RAG columns added: {[c for c in analyzed_df.columns if c.endswith('_RAG')]}")
        
        # Show RAG distribution
        print("\n   RAG Distribution:")
        for metric, dist in summary.get("rag_distribution", {}).items():
            print(f"     {metric}: {dist}")
        
        # Show data quality issues
        issues = summary.get("data_quality_issues", [])
        if issues:
            print(f"\n   Data Quality Issues: {len(issues)}")
            for issue in issues[:5]:
                print(f"     - {issue}")
    
    print("\n" + "=" * 60)
    print("Analyzer Module Test Complete")
    print("=" * 60)
