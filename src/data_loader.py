"""
Data Loader Module - File Ingestion with History Handler (Amendment B)

Handles:
- Detection and loading of latest raw data files
- Loading reference data (vendor_map, ASIN selections)
- Historical data archiving for WoW calculations (Amendment B)
- File validation and structure checking
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd

from src.config import (
    DROPZONE_PATH,
    HISTORICAL_PATH,
    SELECTION_PATH,
    HISTORY_ARCHIVE_PATH,
    DATA_PATH,
)
from src.file_loader import ALLOWED_SUFFIXES, load_file

if TYPE_CHECKING:
    from collections.abc import Sequence



class HistoryHandler:
    """
    Manages historical data archive for WoW calculations (Amendment B).
    
    Stores processed data in DuckDB and retrieves previous period values
    when raw files don't contain embedded WoW deltas.
    """
    
    def __init__(self, archive_path: str | Path | None = None):
        """
        Initialize the HistoryHandler.
        
        Args:
            archive_path: Path to DuckDB archive file.
                         If None, uses default HISTORY_ARCHIVE_PATH.
        """
        if archive_path is None:
            archive_path = HISTORY_ARCHIVE_PATH
        
        self.archive_path = Path(archive_path)
        
        # Ensure directory exists
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to DuckDB
        self.con = duckdb.connect(str(self.archive_path))
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create history table if it doesn't exist."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                asin VARCHAR,
                vendor_code VARCHAR,
                week_number INTEGER,
                year INTEGER,
                metric_name VARCHAR,
                metric_value DOUBLE,
                processed_timestamp TIMESTAMP,
                source_file VARCHAR
            )
        """)
        
        # Create indexes for faster lookups
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_asin_week 
            ON history (asin, week_number, year)
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_metric 
            ON history (metric_name)
        """)
    
    def get_previous_week(
        self,
        asin: str,
        metric_name: str,
        current_week: int,
        current_year: int | None = None
    ) -> float | None:
        """
        Retrieve previous week's value for WoW calculation.
        
        Args:
            asin: Amazon Standard Identification Number.
            metric_name: Name of the metric (e.g., "Ordered_Revenue").
            current_week: Current week number.
            current_year: Current year. If None, uses current year.
        
        Returns:
            Previous week's metric value, or None if not found.
        """
        if current_year is None:
            current_year = datetime.now().year
        
        # Calculate previous week (handle year boundary)
        if current_week == 1:
            prev_week = 52
            prev_year = current_year - 1
        else:
            prev_week = current_week - 1
            prev_year = current_year
        
        result = self.con.execute("""
            SELECT metric_value
            FROM history
            WHERE asin = ?
              AND metric_name = ?
              AND week_number = ?
              AND year = ?
            ORDER BY processed_timestamp DESC
            LIMIT 1
        """, [asin, metric_name, prev_week, prev_year]).fetchone()
        
        return result[0] if result else None
    
    def get_previous_week_batch(
        self,
        asins: list[str],
        metric_names: list[str],
        current_week: int,
        current_year: int | None = None
    ) -> pd.DataFrame:
        """
        Retrieve previous week's values for multiple ASINs and metrics.
        
        Args:
            asins: List of ASINs.
            metric_names: List of metric names.
            current_week: Current week number.
            current_year: Current year.
        
        Returns:
            DataFrame with ASIN, metric_name, and previous_value columns.
        """
        if current_year is None:
            current_year = datetime.now().year
        
        # Ensure types are Python native (not numpy)
        current_week = int(current_week)
        current_year = int(current_year)
        
        # Calculate previous week
        if current_week == 1:
            prev_week = 52
            prev_year = current_year - 1
        else:
            prev_week = current_week - 1
            prev_year = current_year
        
        # Convert asins and metrics to strings (handle potential numpy types)
        asins = [str(a) for a in asins]
        metric_names = [str(m) for m in metric_names]
        
        # Create parameter placeholders
        asin_placeholders = ",".join(["?" for _ in asins])
        metric_placeholders = ",".join(["?" for _ in metric_names])
        
        query = f"""
            SELECT asin, metric_name, metric_value as previous_value
            FROM history
            WHERE asin IN ({asin_placeholders})
              AND metric_name IN ({metric_placeholders})
              AND week_number = ?
              AND year = ?
        """
        
        params = list(asins) + list(metric_names) + [int(prev_week), int(prev_year)]
        
        return self.con.execute(query, params).df()
    
    def archive_current_data(
        self,
        df: pd.DataFrame,
        week_number: int,
        year: int,
        source_file: str = ""
    ) -> int:
        """
        Save current week's processed data for future WoW calculations.
        
        Args:
            df: DataFrame with ASIN and metric columns.
            week_number: Week number.
            year: Year.
            source_file: Source file name for audit trail.
        
        Returns:
            Number of rows archived.
        """
        timestamp = datetime.now()
        rows_to_insert = []
        
        # Get ASIN and vendor code columns
        asin_col = "ASIN" if "ASIN" in df.columns else "asin"
        vendor_col = None
        for col in ["Vendor_Code", "vendor_code", "VENDOR_CODE"]:
            if col in df.columns:
                vendor_col = col
                break
        
        if asin_col not in df.columns:
            return 0
        
        # Define metrics to archive
        metrics_to_archive = [
            "Ordered_Revenue", "Ordered_Units", "Net_PPM", "Glance_Views",
            "Average_Selling_Price", "Fill_Rate_Sourceable", "SoROOS_Pct",
            "Contribution_Margin", "PPM"
        ]
        
        # Find which metrics exist in the DataFrame
        existing_metrics = [m for m in metrics_to_archive if m in df.columns]
        
        for _, row in df.iterrows():
            asin = row[asin_col]
            vendor_code = row[vendor_col] if vendor_col else ""
            
            for metric_name in existing_metrics:
                value = row[metric_name]
                if pd.notna(value):
                    rows_to_insert.append({
                        "asin": str(asin),
                        "vendor_code": str(vendor_code) if vendor_code else "",
                        "week_number": week_number,
                        "year": year,
                        "metric_name": metric_name,
                        "metric_value": float(value),
                        "processed_timestamp": timestamp,
                        "source_file": source_file,
                    })
        
        if rows_to_insert:
            insert_df = pd.DataFrame(rows_to_insert)
            self.con.execute("""
                INSERT INTO history 
                SELECT * FROM insert_df
            """)
        
        return len(rows_to_insert)
    
    def check_wow_availability(
        self,
        df: pd.DataFrame,
        current_week: int,
        current_year: int | None = None
    ) -> dict[str, Any]:
        """
        Check if WoW can be calculated from embedded data or archive.
        
        Args:
            df: DataFrame to check.
            current_week: Current week number.
            current_year: Current year.
        
        Returns:
            Dictionary with:
            - embedded: bool - Whether WoW columns exist in data
            - historical_available: bool - Whether historical data exists
            - first_week_asins: list - ASINs with no historical data
        """
        if current_year is None:
            current_year = datetime.now().year
        
        # Check for embedded WoW columns
        wow_columns = [col for col in df.columns if "_WoW" in col or "WoW" in col]
        embedded = len(wow_columns) > 0
        
        # Check historical availability
        asin_col = "ASIN" if "ASIN" in df.columns else "asin"
        if asin_col not in df.columns:
            return {
                "embedded": embedded,
                "historical_available": False,
                "first_week_asins": [],
            }
        
        asins = [str(a) for a in df[asin_col].unique().tolist()]
        
        # Calculate previous week
        if current_week == 1:
            prev_week = 52
            prev_year = current_year - 1
        else:
            prev_week = current_week - 1
            prev_year = current_year
        
        # Query for ASINs with historical data
        # Use DataFrame registration to avoid type inference issues with parameterized queries
        if not asins:
            asins_with_history = set()
        else:
            # Register ASINs as a temporary table
            asin_df = pd.DataFrame({'asin': asins})
            self.con.register('temp_asins', asin_df)
            
            result = self.con.execute("""
                SELECT DISTINCT h.asin
                FROM history h
                INNER JOIN temp_asins t ON h.asin = t.asin
                WHERE h.week_number = ?
                  AND h.year = ?
            """, [int(prev_week), int(prev_year)]).fetchall()
            
            asins_with_history = {r[0] for r in result}
            
            # Clean up temporary table
            self.con.unregister('temp_asins')
        
        first_week_asins = [a for a in asins if a not in asins_with_history]
        
        return {
            "embedded": embedded,
            "historical_available": len(asins_with_history) > 0,
            "first_week_asins": first_week_asins,
            "asins_with_history": len(asins_with_history),
            "total_asins": len(asins),
        }
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        self.con.close()


def _load_file(
    path: Path,
    *,
    delimiter: str | None = None,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    json_lines: bool | None = None,
) -> pd.DataFrame | None:
    """
    Load a file into a DataFrame, trying multiple encodings if needed.
    
    Args:
        path: Path to the file.
    
    Returns:
        DataFrame or None if loading fails.
    """
    try:
        return load_file(
            path,
            delimiter=delimiter,
            sheet_name=sheet_name,
            encoding=encoding,
            json_lines=json_lines,
        )
    except Exception:
        return None


def detect_latest_file(
    directory: Path | str,
    pattern: str = "*",
    by: str = "modified",
    allowed_suffixes: tuple[str, ...] = ALLOWED_SUFFIXES,
) -> Path | None:
    """
    Find the most recent file in a directory matching a pattern.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern to match files.
        by: Sort by "modified" (mtime) or "name".
    
    Returns:
        Path to the most recent file, or None if no files found.
    """
    directory = Path(directory)
    if not directory.exists():
        return None
    
    files = [f for f in directory.glob(pattern) if f.suffix.lower() in allowed_suffixes]
    if not files:
        return None
    
    if by == "modified":
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    else:
        files.sort(key=lambda f: f.name, reverse=True)
    
    return files[0]


def load_raw_data(
    filepath: Path | str,
    validate: bool = True,
    *,
    delimiter: str | None = None,
    sheet_name: str | int | None = None,
    encoding: str | None = None,
    json_lines: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load raw performance data from a CSV or Excel file.
    
    Args:
        filepath: Path to the file.
        validate: Whether to validate file structure.
        delimiter: Optional CSV delimiter override.
        sheet_name: Optional Excel sheet name or index.
        encoding: Optional CSV encoding override.
        json_lines: Optional flag for JSON lines format.
    
    Returns:
        Tuple of (DataFrame, metadata_dict).
    
    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is not supported or validation fails.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    df = _load_file(
        filepath,
        delimiter=delimiter,
        sheet_name=sheet_name,
        encoding=encoding,
        json_lines=json_lines,
    )
    if df is None:
        raise ValueError(f"Failed to load file: {filepath}")
    
    metadata = {
        "filepath": str(filepath),
        "filename": filepath.name,
        "rows": len(df),
        "columns": len(df.columns),
        "modified": datetime.fromtimestamp(filepath.stat().st_mtime),
        "file_size_kb": filepath.stat().st_size / 1024,
    }
    
    # Validate required columns
    if validate:
        if "ASIN" not in df.columns:
            raise ValueError("Required column 'ASIN' not found in file")
    
    return df, metadata


def load_reference_data(
    selection_path: Path | str | None = None,
    *,
    vendor_map_sheet: str | int | None = None,
    json_lines: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load reference data from the selection directory.
    
    Args:
        selection_path: Path to selection directory.
                       If None, uses default SELECTION_PATH.
        vendor_map_sheet: Optional sheet name/index for vendor map Excel files.
        json_lines: Optional flag for JSON lines format.
    
    Returns:
        Dictionary with:
        - vendor_map: DataFrame with vendor code to name mappings
        - asin_selection: DataFrame with ASIN selection list
    """
    if selection_path is None:
        selection_path = SELECTION_PATH
    
    selection_path = Path(selection_path)
    result = {
        "vendor_map": pd.DataFrame(),
        "asin_selection": pd.DataFrame(columns=["ASIN"]),
    }
    
    # Load vendor map (supports csv/xlsx/xls/xlsm/json)
    vendor_map_candidates = [
        selection_path / f"vendor_map{suffix}" for suffix in ALLOWED_SUFFIXES
    ]
    for candidate in vendor_map_candidates:
        if candidate.exists():
            df = _load_file(candidate, sheet_name=vendor_map_sheet, json_lines=json_lines)
            if df is not None:
                result["vendor_map"] = df
                break
    
    # Warn if vendor_map is empty
    if result["vendor_map"].empty:
        import warnings
        warnings.warn(
            f"vendor_map is empty or not found. Vendor attribution will be missing. "
            f"Expected file: {vendor_map_csv} or {vendor_map_xlsx}. "
            f"Run 'python scripts/generate_vendor_map.py' to generate vendor_map.csv from filenames.",
            UserWarning,
            stacklevel=2
        )
    
    # Load ASIN selection from vendors/ subdirectory
    vendors_dir = selection_path / "vendors"
    if vendors_dir.exists():
        asin_dfs = []
        for file in vendors_dir.iterdir():
            if file.suffix.lower() in ALLOWED_SUFFIXES:
                df = _load_file(file, json_lines=json_lines)
                if df is not None:
                    # Handle multi-sheet Excel files (returns dict when sheet_name=None)
                    if isinstance(df, dict):
                        # Take the first sheet if multiple sheets exist
                        df = list(df.values())[0] if df else None
                        if df is None:
                            continue
                    
                    # Find ASIN column (case-insensitive)
                    asin_cols = [c for c in df.columns if c.upper() == "ASIN" or "matched_asin" in c.lower()]
                    if asin_cols:
                        asin_df = df[[asin_cols[0]]].rename(columns={asin_cols[0]: "ASIN"})
                        asin_df["source_file"] = file.name
                        asin_dfs.append(asin_df)
        
        if asin_dfs:
            result["asin_selection"] = pd.concat(asin_dfs, ignore_index=True).drop_duplicates(subset=["ASIN"])
    
    return result


def validate_asin_cross_reference(
    raw_df: pd.DataFrame,
    asin_selection: pd.DataFrame | None = None,
    vendor_map: pd.DataFrame | None = None
) -> dict[str, Any]:
    """
    Validate ASIN cross-reference between raw data and reference files.
    
    Reports statistics on ASIN overlap, orphaned ASINs, and reference coverage.
    
    Args:
        raw_df: Raw data DataFrame (must contain ASIN column).
        asin_selection: Reference ASIN selection DataFrame (optional).
        vendor_map: Vendor map DataFrame (optional, for vendor attribution).
    
    Returns:
        Dictionary with validation statistics:
        - raw_asins_count: Total unique ASINs in raw data
        - reference_asins_count: Total unique ASINs in reference
        - matched_asins_count: ASINs present in both
        - orphaned_asins_count: ASINs in raw data but not in reference
        - unmatched_reference_count: ASINs in reference but not in raw data
        - match_percentage: Percentage of raw ASINs matched
        - orphaned_asins: List of orphaned ASINs (first 20)
        - unmatched_reference_asins: List of unmatched reference ASINs (first 20)
    """
    result: dict[str, Any] = {
        "raw_asins_count": 0,
        "reference_asins_count": 0,
        "matched_asins_count": 0,
        "orphaned_asins_count": 0,
        "unmatched_reference_count": 0,
        "match_percentage": 0.0,
        "orphaned_asins": [],
        "unmatched_reference_asins": [],
    }
    
    # Check if ASIN column exists in raw data
    asin_col = "ASIN" if "ASIN" in raw_df.columns else "asin"
    if asin_col not in raw_df.columns:
        return result
    
    # Get unique ASINs from raw data
    raw_asins = set(raw_df[asin_col].dropna().astype(str).unique())
    result["raw_asins_count"] = len(raw_asins)
    
    # If no reference data provided, return early
    if asin_selection is None or asin_selection.empty:
        result["orphaned_asins"] = sorted(list(raw_asins))[:20]
        result["orphaned_asins_count"] = len(raw_asins)
        result["match_percentage"] = 0.0
        return result
    
    # Get unique ASINs from reference
    ref_asin_col = "ASIN" if "ASIN" in asin_selection.columns else "asin"
    if ref_asin_col not in asin_selection.columns:
        result["orphaned_asins"] = sorted(list(raw_asins))[:20]
        result["orphaned_asins_count"] = len(raw_asins)
        result["match_percentage"] = 0.0
        return result
    
    reference_asins = set(asin_selection[ref_asin_col].dropna().astype(str).unique())
    result["reference_asins_count"] = len(reference_asins)
    
    # Calculate overlap
    matched_asins = raw_asins & reference_asins
    orphaned_asins = raw_asins - reference_asins
    unmatched_reference = reference_asins - raw_asins
    
    result["matched_asins_count"] = len(matched_asins)
    result["orphaned_asins_count"] = len(orphaned_asins)
    result["unmatched_reference_count"] = len(unmatched_reference)
    
    # Calculate match percentage
    if len(raw_asins) > 0:
        result["match_percentage"] = (len(matched_asins) / len(raw_asins)) * 100.0
    
    # Store sample lists (limit to 20 for readability)
    result["orphaned_asins"] = sorted(list(orphaned_asins))[:20]
    result["unmatched_reference_asins"] = sorted(list(unmatched_reference))[:20]
    
    return result


def validate_file_structure(df: pd.DataFrame, required_columns: list[str] | None = None) -> tuple[bool, list[str]]:
    """
    Validate that a DataFrame has the required structure.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
                         If None, just checks for ASIN.
    
    Returns:
        Tuple of (is_valid, list_of_missing_columns).
    """
    if required_columns is None:
        required_columns = ["ASIN"]
    
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


def get_file_list(
    directory: Path | str,
    pattern: str = "*.csv",
    recursive: bool = False
) -> list[Path]:
    """
    Get a list of files matching a pattern in a directory.
    
    Args:
        directory: Directory to search.
        pattern: Glob pattern.
        recursive: Whether to search recursively.
    
    Returns:
        List of file paths.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def extract_week_from_filename(
    filename: str,
    filepath: Path | str | None = None
) -> tuple[int | None, int | None]:
    """
    Extract week number and year from filename and file metadata.
    
    If year is not in filename, infers it from file modification date and week number.
    This handles cases where weeks 50-52 are from the previous year (e.g., 2025)
    while weeks 1-2 are from the current year (e.g., 2026).
    
    Args:
        filename: Filename to parse.
        filepath: Optional path to the file. If provided, uses file modification date
                 to infer year when not found in filename.
    
    Returns:
        Tuple of (week_number, year) or (None, None) if not found.
    
    Examples:
        "KY2O0-W49-W52.csv" -> (52, 2025)  # Infers 2025 from file date if week 52
        "WJTP1-W48-2025.csv" -> (48, 2025)  # Year in filename
        "W1-ALL-brightweek_AE_2026023.csv" -> (1, 2026)  # Infers 2026 from file date
    """
    import re
    from datetime import datetime
    
    # Pattern: W followed by digits
    week_matches = re.findall(r'W(\d+)', filename, re.IGNORECASE)
    
    # Pattern: 4-digit year
    year_matches = re.findall(r'20\d{2}', filename)
    
    week = int(week_matches[-1]) if week_matches else None
    year = int(year_matches[-1]) if year_matches else None
    
    # If year not found in filename and filepath provided, infer from file date
    if year is None and filepath is not None and week is not None:
        filepath = Path(filepath)
        if filepath.exists():
            # Get file modification time
            file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            file_year = file_mtime.year
            
            # Get ISO week year from file date
            iso_year, iso_week, _ = file_mtime.isocalendar()
            
            # Logic: If week is 50-52, it's likely from the previous calendar year
            # If week is 1-2, it's likely from the current calendar year
            # Use the ISO week year as a guide, but adjust based on week number
            if week >= 50:
                # Weeks 50-52 are typically from the previous calendar year
                # Check if file date is in Dec of previous year or Jan of current year
                if file_mtime.month == 12 or (file_mtime.month == 1 and file_mtime.day <= 7):
                    year = file_year if file_mtime.month == 12 else file_year - 1
                else:
                    year = iso_year
            elif week <= 2:
                # Weeks 1-2 are typically from the current calendar year
                # Check if file date is in Jan of current year
                if file_mtime.month == 1 or (file_mtime.month == 12 and file_mtime.day >= 25):
                    year = file_year if file_mtime.month == 1 else file_year + 1
                else:
                    year = iso_year
            else:
                # For weeks 3-49, use ISO week year from file date
                year = iso_year
    
    return week, year


def validate_asin_overlap(
    raw_data: pd.DataFrame,
    reference_data: dict[str, pd.DataFrame] | None = None,
    selection_path: Path | str | None = None
) -> dict[str, Any]:
    """
    Cross-reference ASINs between raw data and reference files.
    
    Validates ASIN overlap and reports statistics on:
    - ASINs in raw data that are in reference (matched)
    - ASINs in raw data that are NOT in reference (orphaned)
    - ASINs in reference that are NOT in raw data (missing)
    - Duplicate ASINs in reference files
    
    Args:
        raw_data: DataFrame with raw data (must contain ASIN column).
        reference_data: Optional pre-loaded reference data dictionary.
                      If None, loads from selection_path.
        selection_path: Path to selection directory. Only used if reference_data is None.
    
    Returns:
        Dictionary with validation statistics:
        - total_raw_asins: Total unique ASINs in raw data
        - total_reference_asins: Total unique ASINs in reference
        - matched_asins: Count of ASINs present in both
        - orphaned_asins: Count of ASINs in raw data but not in reference
        - missing_asins: Count of ASINs in reference but not in raw data
        - orphaned_list: List of orphaned ASINs (first 100)
        - missing_list: List of missing ASINs (first 100)
        - duplicate_reference_asins: Count of duplicate ASINs in reference files
        - match_percentage: Percentage of raw ASINs that match reference
        - overlap_percentage: Percentage of reference ASINs found in raw data
    """
    # Get ASIN column from raw data
    asin_col = None
    for col in ["ASIN", "asin", "ASINs"]:
        if col in raw_data.columns:
            asin_col = col
            break
    
    if asin_col is None:
        return {
            "error": "ASIN column not found in raw data",
            "total_raw_asins": 0,
            "total_reference_asins": 0,
            "matched_asins": 0,
            "orphaned_asins": 0,
            "missing_asins": 0,
            "orphaned_list": [],
            "missing_list": [],
            "duplicate_reference_asins": 0,
            "match_percentage": 0.0,
            "overlap_percentage": 0.0,
        }
    
    # Get unique ASINs from raw data
    raw_asins = set(raw_data[asin_col].dropna().astype(str).unique())
    total_raw_asins = len(raw_asins)
    
    # Load reference data if not provided
    if reference_data is None:
        reference_data = load_reference_data(selection_path)
    
    # Get ASIN selection from reference
    asin_selection = reference_data.get("asin_selection", pd.DataFrame())
    
    if asin_selection.empty or "ASIN" not in asin_selection.columns:
        # No reference ASINs available
        return {
            "total_raw_asins": total_raw_asins,
            "total_reference_asins": 0,
            "matched_asins": 0,
            "orphaned_asins": total_raw_asins,
            "missing_asins": 0,
            "orphaned_list": list(raw_asins)[:100],
            "missing_list": [],
            "duplicate_reference_asins": 0,
            "match_percentage": 0.0,
            "overlap_percentage": 0.0,
            "warning": "No reference ASIN selection found",
        }
    
    # Get unique ASINs from reference
    reference_asins = set(asin_selection["ASIN"].dropna().astype(str).unique())
    total_reference_asins = len(reference_asins)
    
    # Check for duplicates in reference
    duplicate_count = len(asin_selection) - len(reference_asins)
    
    # Calculate overlap
    matched_asins = raw_asins & reference_asins
    orphaned_asins = raw_asins - reference_asins
    missing_asins = reference_asins - raw_asins
    
    # Calculate percentages
    match_percentage = (len(matched_asins) / total_raw_asins * 100) if total_raw_asins > 0 else 0.0
    overlap_percentage = (len(matched_asins) / total_reference_asins * 100) if total_reference_asins > 0 else 0.0
    
    return {
        "total_raw_asins": total_raw_asins,
        "total_reference_asins": total_reference_asins,
        "matched_asins": len(matched_asins),
        "orphaned_asins": len(orphaned_asins),
        "missing_asins": len(missing_asins),
        "orphaned_list": list(orphaned_asins)[:100],  # Limit to first 100
        "missing_list": list(missing_asins)[:100],   # Limit to first 100
        "duplicate_reference_asins": duplicate_count,
        "match_percentage": match_percentage,
        "overlap_percentage": overlap_percentage,
    }


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Data Loader Module Test")
    print("=" * 60)
    
    # Test file detection
    print("\n1. Testing file detection...")
    latest = detect_latest_file(DROPZONE_PATH)
    if latest:
        print(f"   Latest file: {latest.name}")
    else:
        print("   No files found")
    
    # Test loading reference data
    print("\n2. Loading reference data...")
    ref_data = load_reference_data()
    print(f"   Vendor map: {len(ref_data['vendor_map'])} rows")
    print(f"   ASIN selection: {len(ref_data['asin_selection'])} ASINs")
    
    # Test loading raw data
    print("\n3. Loading raw data...")
    if latest:
        try:
            df, metadata = load_raw_data(latest)
            print(f"   Loaded {metadata['rows']} rows, {metadata['columns']} columns")
            print(f"   File size: {metadata['file_size_kb']:.1f} KB")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test HistoryHandler
    print("\n4. Testing HistoryHandler...")
    handler = HistoryHandler()
    print(f"   Archive path: {handler.archive_path}")
    handler.close()
    print("   Handler initialized successfully")
    
    print("\n" + "=" * 60)
    print("Data Loader Module Test Complete")
    print("=" * 60)
