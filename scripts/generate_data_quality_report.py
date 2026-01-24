"""
Generate Data Quality Baseline Report

Executes comprehensive data quality analysis on raw data files including:
- Null value analysis
- Value range validation
- Outlier detection
- Data type consistency
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data_loader import load_raw_data, DROPZONE_PATH
from src.config import OUTPUT_PATH


def detect_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Detect outliers in a numeric series.
    
    Args:
        series: Numeric pandas Series
        method: 'iqr' (Interquartile Range) or 'zscore'
    
    Returns:
        Boolean Series indicating outliers
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > 3
    else:
        return pd.Series([False] * len(series), index=series.index)


def analyze_column(series: pd.Series, col_name: str) -> dict:
    """
    Analyze a single column for data quality metrics.
    
    Args:
        series: Column Series
        col_name: Column name
    
    Returns:
        Dictionary with quality metrics
    """
    result = {
        "column": col_name,
        "dtype": str(series.dtype),
        "total_values": len(series),
        "null_count": series.isna().sum(),
        "null_percentage": (series.isna().sum() / len(series)) * 100,
        "unique_count": series.nunique(),
        "duplicate_count": series.duplicated().sum(),
    }
    
    # Numeric analysis
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = series.dropna()
        if len(numeric_series) > 0:
            result.update({
                "min": float(numeric_series.min()),
                "max": float(numeric_series.max()),
                "mean": float(numeric_series.mean()),
                "median": float(numeric_series.median()),
                "std": float(numeric_series.std()),
                "outliers_iqr": int(detect_outliers(numeric_series, "iqr").sum()),
                "outliers_zscore": int(detect_outliers(numeric_series, "zscore").sum()),
            })
            
            # Check for negative values in metrics that shouldn't be negative
            if "Revenue" in col_name or "GMS" in col_name or "Units" in col_name:
                negative_count = (numeric_series < 0).sum()
                result["negative_values"] = int(negative_count)
                result["negative_percentage"] = (negative_count / len(numeric_series)) * 100
    
    # String analysis
    elif pd.api.types.is_string_dtype(series) or series.dtype == "object":
        non_null = series.dropna()
        if len(non_null) > 0:
            result.update({
                "min_length": int(non_null.astype(str).str.len().min()) if len(non_null) > 0 else 0,
                "max_length": int(non_null.astype(str).str.len().max()) if len(non_null) > 0 else 0,
                "empty_strings": int((non_null.astype(str).str.strip() == "").sum()),
            })
            
            # ASIN format validation
            if "ASIN" in col_name.upper():
                asin_pattern = r"^[A-Z0-9]{10}$"
                valid_asins = non_null.astype(str).str.match(asin_pattern, na=False).sum()
                result["valid_asin_format"] = int(valid_asins)
                result["invalid_asin_format"] = int(len(non_null) - valid_asins)
    
    return result


def analyze_file(filepath: Path) -> dict:
    """
    Analyze a single data file for quality metrics.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Dictionary with file-level and column-level metrics
    """
    print(f"Analyzing: {filepath.name}")
    
    try:
        # Load data
        df, metadata = load_raw_data(filepath)
        
        file_result = {
            "file": filepath.name,
            "rows": len(df),
            "columns": len(df.columns),
            "file_size_kb": metadata.get("file_size_kb", 0),
            "column_analysis": [],
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = analyze_column(df[col], col)
            file_result["column_analysis"].append(col_analysis)
        
        # File-level summary
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        file_result["overall_null_percentage"] = (null_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Identify critical columns
        critical_columns = [col for col in df.columns if "ASIN" in col.upper() or "Revenue" in col or "Units" in col]
        file_result["critical_columns"] = critical_columns
        
        # Check critical column quality
        critical_issues = []
        for col in critical_columns:
            col_data = df[col]
            null_pct = (col_data.isna().sum() / len(col_data)) * 100
            if null_pct > 10:
                critical_issues.append(f"{col}: {null_pct:.1f}% null")
        
        file_result["critical_issues"] = critical_issues
        
        print(f"  Rows: {file_result['rows']}, Columns: {file_result['columns']}")
        print(f"  Overall null rate: {file_result['overall_null_percentage']:.2f}%")
        if critical_issues:
            print(f"  WARNING: Critical issues: {len(critical_issues)}")
        
        return file_result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "file": filepath.name,
            "error": str(e),
        }


def main():
    """Generate comprehensive data quality report."""
    print("=" * 80)
    print("Data Quality Baseline Report")
    print("=" * 80)
    print()
    
    # Get all weekly performance files
    performance_files = list(DROPZONE_PATH.glob("*.csv"))
    
    if not performance_files:
        print("No performance files found in dropzone!")
        return
    
    print(f"Found {len(performance_files)} performance files")
    print()
    
    # Analyze each file
    all_results = []
    summary_stats = {
        "total_files": len(performance_files),
        "total_rows": 0,
        "total_columns": 0,
        "files_with_issues": 0,
        "average_null_rate": 0.0,
    }
    
    for filepath in sorted(performance_files):
        result = analyze_file(filepath)
        all_results.append(result)
        
        if "error" not in result:
            summary_stats["total_rows"] += result["rows"]
            summary_stats["total_columns"] += result["columns"]
            summary_stats["average_null_rate"] += result.get("overall_null_percentage", 0)
            if result.get("critical_issues"):
                summary_stats["files_with_issues"] += 1
        print()
    
    # Calculate averages
    valid_files = [r for r in all_results if "error" not in r]
    if valid_files:
        summary_stats["average_null_rate"] /= len(valid_files)
        summary_stats["average_rows_per_file"] = summary_stats["total_rows"] / len(valid_files)
        summary_stats["average_columns_per_file"] = summary_stats["total_columns"] / len(valid_files)
    
    # Generate summary report
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    print(f"Files analyzed: {summary_stats['total_files']}")
    print(f"Total rows (across all files): {summary_stats['total_rows']:,}")
    print(f"Total columns (across all files): {summary_stats['total_columns']:,}")
    print(f"Average rows per file: {summary_stats.get('average_rows_per_file', 0):.0f}")
    print(f"Average columns per file: {summary_stats.get('average_columns_per_file', 0):.0f}")
    print(f"Average null rate: {summary_stats['average_null_rate']:.2f}%")
    print(f"Files with critical issues: {summary_stats['files_with_issues']}")
    print()
    
    # Create detailed column analysis DataFrame
    column_details = []
    for file_result in all_results:
        if "error" not in file_result and "column_analysis" in file_result:
            for col_analysis in file_result["column_analysis"]:
                col_analysis["source_file"] = file_result["file"]
                column_details.append(col_analysis)
    
    if column_details:
        df_columns = pd.DataFrame(column_details)
        
        # Save detailed column analysis
        output_file_columns = OUTPUT_PATH / "data_quality_column_analysis.csv"
        df_columns.to_csv(output_file_columns, index=False)
        print(f"Column-level analysis saved to: {output_file_columns}")
        
        # Generate summary by column type
        print()
        print("=" * 80)
        print("COLUMN TYPE SUMMARY")
        print("=" * 80)
        print()
        
        # Numeric columns summary
        numeric_cols = df_columns[df_columns["dtype"].str.contains("int|float", case=False, na=False)]
        if len(numeric_cols) > 0:
            print(f"Numeric columns: {len(numeric_cols)}")
            print(f"  Average null rate: {numeric_cols['null_percentage'].mean():.2f}%")
            if "outliers_iqr" in numeric_cols.columns:
                total_outliers = numeric_cols["outliers_iqr"].sum()
                print(f"  Total outliers (IQR method): {total_outliers:,}")
        
        # String columns summary
        string_cols = df_columns[df_columns["dtype"] == "object"]
        if len(string_cols) > 0:
            print(f"String columns: {len(string_cols)}")
            print(f"  Average null rate: {string_cols['null_percentage'].mean():.2f}%")
        
        # ASIN columns validation
        asin_cols = df_columns[df_columns["column"].str.contains("ASIN", case=False, na=False)]
        if len(asin_cols) > 0 and "invalid_asin_format" in asin_cols.columns:
            invalid_asins = asin_cols["invalid_asin_format"].sum()
            if invalid_asins > 0:
                print(f"WARNING: ASIN format issues: {invalid_asins} invalid ASINs found")
            else:
                print(f"ASIN format validation: All ASINs valid")
    
    # Save file-level summary
    file_summary = []
    for result in all_results:
        if "error" not in result:
            file_summary.append({
                "file": result["file"],
                "rows": result["rows"],
                "columns": result["columns"],
                "file_size_kb": result.get("file_size_kb", 0),
                "overall_null_percentage": result.get("overall_null_percentage", 0),
                "critical_issues_count": len(result.get("critical_issues", [])),
                "critical_issues": "; ".join(result.get("critical_issues", [])),
            })
    
    if file_summary:
        df_files = pd.DataFrame(file_summary)
        output_file_files = OUTPUT_PATH / "data_quality_file_summary.csv"
        df_files.to_csv(output_file_files, index=False)
        print()
        print(f"File-level summary saved to: {output_file_files}")
    
    print()
    print("=" * 80)
    print("Data quality analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
