"""
Automated Validation & Debugging Protocol (The Reviewer).

Diagnostic tool that validates the entire data pipeline, including directory structure,
schema compliance, and pipeline integrity tests with auto-fix and mock data generation.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src import analytics, ingestion
from src.normalization import CANONICAL_COLUMNS, CANONICAL_DTYPES, normalize_data

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Directory paths matching existing codebase
DROPZONE_PATH = Path("01_dropzone")
SELECTION_PATH = Path("00_selection")

# Supported file extensions
ALLOWED_SUFFIXES = (".csv", ".xlsx", ".xls")


def check_directories() -> dict[str, bool | str | int]:
    """
    Verify dropzone and selection directories exist, create if missing, and generate mock data if empty.

    Returns:
        Dictionary with status information:
        - "success": bool - Overall success status
        - "dropzone_exists": bool - Whether dropzone exists
        - "selection_exists": bool - Whether selection exists
        - "dropzone_created": bool - Whether dropzone was created
        - "selection_created": bool - Whether selection was created
        - "dropzone_mock_generated": bool - Whether mock data was generated in dropzone
        - "selection_mock_generated": bool - Whether mock data was generated in selection
        - "message": str - Status message
    """
    result: dict[str, bool | str | int] = {
        "success": True,
        "dropzone_exists": False,
        "selection_exists": False,
        "dropzone_created": False,
        "selection_created": False,
        "dropzone_mock_generated": False,
        "selection_mock_generated": False,
        "message": "",
    }

    # Check and create dropzone
    if not DROPZONE_PATH.exists():
        try:
            DROPZONE_PATH.mkdir(parents=True, exist_ok=True)
            result["dropzone_created"] = True
            result["message"] += f"Created directory: {DROPZONE_PATH}\n"
        except Exception as e:
            result["success"] = False
            result["message"] += f"Failed to create {DROPZONE_PATH}: {e}\n"
            return result
    else:
        result["dropzone_exists"] = True

    # Check and create selection
    if not SELECTION_PATH.exists():
        try:
            SELECTION_PATH.mkdir(parents=True, exist_ok=True)
            result["selection_created"] = True
            result["message"] += f"Created directory: {SELECTION_PATH}\n"
        except Exception as e:
            result["success"] = False
            result["message"] += f"Failed to create {SELECTION_PATH}: {e}\n"
            return result
    else:
        result["selection_exists"] = True

    # Check if dropzone is empty and generate mock data
    dropzone_files = list(DROPZONE_PATH.rglob("*"))
    dropzone_files = [
        f for f in dropzone_files if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]

    if not dropzone_files:
        try:
            # Generate mock historical_sales.csv
            mock_sales_data = {
                "ASIN": [
                    "B08XYZ1234",
                    "B09ABC5678",
                    "B10DEF9012",
                    "B11GHI3456",
                    "B12JKL7890",
                ],
                "SKU": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
                "Snapshot_Date": [
                    datetime.now() - timedelta(days=i) for i in range(5, 0, -1)
                ],
                "Product_Title": [
                    "Sample Product A",
                    "Sample Product B",
                    "Sample Product C",
                    "Sample Product D",
                    "Sample Product E",
                ],
                "Ordered_Revenue": [1000.50, 2000.75, 1500.25, 3000.00, 2500.50],
                "Ordered_Units": [10, 20, 15, 30, 25],
                "CCOGS": [600.30, 1200.45, 900.15, 1800.00, 1500.30],
                "Promotional_Rebates": [50.00, 100.00, 75.00, 150.00, 125.00],
            }
            mock_sales_df = pd.DataFrame(mock_sales_data)
            mock_sales_path = DROPZONE_PATH / "historical_sales.csv"
            mock_sales_df.to_csv(mock_sales_path, index=False)
            result["dropzone_mock_generated"] = True
            result["message"] += f"Generated mock data: {mock_sales_path}\n"
        except Exception as e:
            result["success"] = False
            result["message"] += f"Failed to generate dropzone mock data: {e}\n"

    # Check if selection is empty and generate mock data
    selection_files = list(SELECTION_PATH.rglob("*"))
    selection_files = [
        f for f in selection_files if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]

    if not selection_files:
        try:
            # Generate mock asin_list.csv
            mock_asin_data = {
                "ASIN": ["B08XYZ1234", "B09ABC5678", "B10DEF9012", "B11GHI3456", "B12JKL7890"],
                "CCOGS": [600.30, 1200.45, 900.15, 1800.00, 1500.30],
            }
            mock_asin_df = pd.DataFrame(mock_asin_data)
            mock_asin_path = SELECTION_PATH / "asin_list.csv"
            mock_asin_df.to_csv(mock_asin_path, index=False)
            result["selection_mock_generated"] = True
            result["message"] += f"Generated mock data: {mock_asin_path}\n"
        except Exception as e:
            result["success"] = False
            result["message"] += f"Failed to generate selection mock data: {e}\n"

    if result["message"] == "":
        result["message"] = "All directories exist and contain data files."

    return result


def validate_schemas() -> dict[str, bool | str | list[str]]:
    """
    Validate that files in dropzone conform to the canonical schema after normalization.

    Returns:
        Dictionary with validation results:
        - "success": bool - Overall validation success
        - "files_checked": int - Number of files checked
        - "missing_columns": list[str] - List of missing required columns
        - "found_columns": list[str] - List of columns found in normalized data
        - "type_mismatches": list[str] - List of columns with type mismatches
        - "message": str - Detailed validation message
    """
    result: dict[str, bool | str | list[str]] = {
        "success": True,
        "files_checked": 0,
        "missing_columns": [],
        "found_columns": [],
        "type_mismatches": [],
        "message": "",
    }

    if not DROPZONE_PATH.exists():
        result["success"] = False
        result["message"] = f"Dropzone directory not found: {DROPZONE_PATH}"
        return result

    # Find first file in dropzone
    dropzone_files = list(DROPZONE_PATH.rglob("*"))
    dropzone_files = [
        f for f in dropzone_files if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
    ]

    if not dropzone_files:
        result["success"] = False
        result["message"] = f"No data files found in {DROPZONE_PATH}"
        return result

    # Try to load and normalize files until we find one that works
    normalized_df = None
    test_file = None
    
    for file_path in dropzone_files:
        try:
            # Use ingestion module's file loading logic for consistency
            from src.ingestion import _load_file
            
            df = _load_file(file_path)
            if df is None or df.empty:
                continue

            # Normalize the data
            normalized_df = normalize_data(df, source_file=file_path)

            if normalized_df.empty:
                continue
                
            # Successfully loaded and normalized a file
            test_file = file_path
            break
        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")
            continue

    if normalized_df is None or test_file is None:
        result["success"] = False
        result["message"] = f"Could not load and normalize any file from {DROPZONE_PATH}"
        return result

    try:
        result["files_checked"] = 1
        result["found_columns"] = list(normalized_df.columns)
        
        # Update message to include which file was checked
        result["message"] = f"Validated file: {test_file.name}\n"

        # Check for missing required columns
        missing = [col for col in CANONICAL_COLUMNS if col not in normalized_df.columns]
        result["missing_columns"] = missing

        if missing:
            result["success"] = False
            result["message"] += (
                f"CRITICAL: Missing required columns: {missing}. "
                f"Found columns: {result['found_columns']}. "
                f"Please update normalization.py mapping."
            )
            return result

        # Check data types (basic check - pandas dtypes may vary)
        type_mismatches = []
        for col in CANONICAL_COLUMNS:
            if col not in normalized_df.columns:
                continue

            expected_dtype = CANONICAL_DTYPES[col]
            actual_dtype = str(normalized_df[col].dtype)

            # Flexible type checking
            if expected_dtype == "datetime64[ns]":
                if not pd.api.types.is_datetime64_any_dtype(normalized_df[col]):
                    type_mismatches.append(f"{col}: expected datetime, got {actual_dtype}")
            elif expected_dtype == "int64":
                if not pd.api.types.is_integer_dtype(normalized_df[col]):
                    type_mismatches.append(f"{col}: expected integer, got {actual_dtype}")
            elif expected_dtype == "float64":
                if not pd.api.types.is_float_dtype(normalized_df[col]):
                    type_mismatches.append(f"{col}: expected float, got {actual_dtype}")
            elif expected_dtype == "string":
                if not pd.api.types.is_string_dtype(normalized_df[col]):
                    type_mismatches.append(f"{col}: expected string, got {actual_dtype}")

        result["type_mismatches"] = type_mismatches

        if type_mismatches:
            result["success"] = False
            result["message"] += f"Type mismatches found: {type_mismatches}"
        else:
            result["message"] += (
                f"Schema validation PASSED. "
                f"All required columns present with correct types."
            )

    except Exception as e:
        result["success"] = False
        if test_file:
            result["message"] = f"Error validating schema for {test_file}: {e}"
        else:
            result["message"] = f"Error validating schema: {e}"
        logger.exception("Schema validation error")

    return result


def test_pipeline_integrity() -> dict[str, bool | str | int]:
    """
    Test the full pipeline: ingestion, DuckDB registration, and analytics.

    Returns:
        Dictionary with test results:
        - "success": bool - Overall test success
        - "ingestion_success": bool - Whether ingestion succeeded
        - "duckdb_success": bool - Whether DuckDB registration succeeded
        - "analytics_success": bool - Whether analytics succeeded
        - "overlap_warning": bool - Whether overlap warning was issued
        - "dropzone_rows": int - Number of rows in dropzone data
        - "selection_rows": int - Number of rows in selection data
        - "overlap_count": int - Number of overlapping ASINs
        - "message": str - Detailed test message
    """
    result: dict[str, bool | str | int] = {
        "success": True,
        "ingestion_success": False,
        "duckdb_success": False,
        "analytics_success": False,
        "overlap_warning": False,
        "dropzone_rows": 0,
        "selection_rows": 0,
        "overlap_count": 0,
        "message": "",
    }

    # Test ingestion
    try:
        dropzone_df = ingestion.scan_dropzone(path=str(DROPZONE_PATH))
        if dropzone_df.empty:
            result["success"] = False
            result["message"] = "Ingestion test FAILED: scan_dropzone() returned empty DataFrame"
            return result

        result["ingestion_success"] = True
        result["dropzone_rows"] = len(dropzone_df)

        # Check canonical columns
        missing_cols = [col for col in CANONICAL_COLUMNS if col not in dropzone_df.columns]
        if missing_cols:
            result["success"] = False
            result["message"] = (
                f"Ingestion test FAILED: Missing columns in normalized data: {missing_cols}"
            )
            return result

        result["message"] += f"Ingestion: SUCCESS ({len(dropzone_df)} rows, {len(dropzone_df.columns)} columns)\n"

    except Exception as e:
        result["success"] = False
        result["message"] = f"Ingestion test FAILED: {e}\n"
        logger.exception("Ingestion test error")
        return result

    # Test DuckDB connectivity
    try:
        con = duckdb.connect(":memory:")
        con.register("raw_sales", dropzone_df)

        # Test query
        count_result = con.execute("SELECT COUNT(*) FROM raw_sales").fetchone()
        row_count = count_result[0] if count_result else 0

        if row_count == 0:
            result["success"] = False
            result["message"] += "DuckDB test FAILED: Table registered but contains 0 rows\n"
            return result

        result["duckdb_success"] = True
        result["message"] += f"DuckDB registration: SUCCESS ({row_count} rows)\n"

    except Exception as e:
        result["success"] = False
        result["message"] += f"DuckDB test FAILED: {e}\n"
        logger.exception("DuckDB test error")
        return result

    # Test selection overlap
    try:
        selection_data = ingestion.scan_selection(path=str(SELECTION_PATH))
        ref_selection_df = selection_data.get("ref_selection", pd.DataFrame(columns=["ASIN"]))

        if not ref_selection_df.empty and "ASIN" in ref_selection_df.columns:
            result["selection_rows"] = len(ref_selection_df)

            # Extract ASINs
            dropzone_asins = set(dropzone_df["ASIN"].dropna().astype(str))
            selection_asins = set(ref_selection_df["ASIN"].dropna().astype(str))

            # Remove empty strings
            dropzone_asins = {a for a in dropzone_asins if a.strip()}
            selection_asins = {a for a in selection_asins if a.strip()}

            overlap = dropzone_asins & selection_asins
            result["overlap_count"] = len(overlap)

            if len(overlap) == 0:
                result["overlap_warning"] = True
                result["message"] += (
                    "WARNING: No overlap between Selection and Dropzone. "
                    "Analytics will return empty results.\n"
                )
            else:
                result["message"] += (
                    f"Selection overlap: {len(overlap)} matching ASINs "
                    f"({len(dropzone_asins)} in dropzone, {len(selection_asins)} in selection)\n"
                )

            # Register ref_selection for analytics test
            if not ref_selection_df.empty:
                con.register("ref_selection", ref_selection_df)
        else:
            result["message"] += "Selection: No ref_selection data found (optional)\n"

    except Exception as e:
        result["message"] += f"Selection overlap check: WARNING - {e}\n"
        logger.warning("Selection overlap check error", exc_info=True)

    # Test analytics dry run
    try:
        metrics_df = analytics.calculate_metrics(
            con,
            selection_dir=str(SELECTION_PATH),
            filter_by_selection=False,
        )

        # Analytics should return a DataFrame (may be empty, but should not raise exceptions)
        if not isinstance(metrics_df, pd.DataFrame):
            result["success"] = False
            result["message"] += "Analytics test FAILED: calculate_metrics() did not return a DataFrame\n"
            return result

        result["analytics_success"] = True
        result["message"] += f"Analytics dry run: SUCCESS ({len(metrics_df)} result rows)\n"

    except Exception as e:
        result["success"] = False
        result["message"] += f"Analytics test FAILED: {e}\n"
        logger.exception("Analytics test error")
        return result

    con.close()

    return result


def print_health_report(
    dir_result: dict[str, bool | str | int],
    schema_result: dict[str, bool | str | list[str]],
    pipeline_result: dict[str, bool | str | int],
) -> None:
    """Print formatted system health report."""
    print("\n" + "=" * 60)
    print("SYSTEM HEALTH REPORT")
    print("=" * 60)

    # Directory Check
    dir_status = "[PASS]" if dir_result.get("success", False) else "[FAIL]"
    print(f"\n{dir_status} Directory Check: {'PASS' if dir_result.get('success') else 'FAIL'}")
    if dir_result.get("dropzone_exists") or dir_result.get("dropzone_created"):
        print(f"    - {DROPZONE_PATH}/: exists")
    else:
        print(f"    - {DROPZONE_PATH}/: NOT FOUND")
    if dir_result.get("selection_exists") or dir_result.get("selection_created"):
        print(f"    - {SELECTION_PATH}/: exists")
    else:
        print(f"    - {SELECTION_PATH}/: NOT FOUND")
    if dir_result.get("dropzone_mock_generated"):
        print(f"    - Generated mock data in {DROPZONE_PATH}/")
    if dir_result.get("selection_mock_generated"):
        print(f"    - Generated mock data in {SELECTION_PATH}/")
    if dir_result.get("message"):
        for line in str(dir_result.get("message", "")).strip().split("\n"):
            if line.strip():
                print(f"    {line}")

    # Schema Validation
    schema_status = "[PASS]" if schema_result.get("success", False) else "[FAIL]"
    print(f"\n{schema_status} Schema Validation: {'PASS' if schema_result.get('success') else 'FAIL'}")
    if schema_result.get("files_checked", 0) > 0:
        print(f"    - Files checked: {schema_result.get('files_checked')}")
    if schema_result.get("missing_columns"):
        missing = schema_result.get("missing_columns", [])
        found = schema_result.get("found_columns", [])
        print(f"    CRITICAL: Missing columns: {missing}")
        print(f"    Found columns: {found}")
    if schema_result.get("type_mismatches"):
        print(f"    Type mismatches: {schema_result.get('type_mismatches')}")
    if schema_result.get("message"):
        for line in str(schema_result.get("message", "")).strip().split("\n"):
            if line.strip():
                print(f"    {line}")

    # Pipeline Integrity
    pipeline_status = "[PASS]" if pipeline_result.get("success", False) else "[FAIL]"
    print(f"\n{pipeline_status} Pipeline Integrity: {'PASS' if pipeline_result.get('success') else 'FAIL'}")
    if pipeline_result.get("dropzone_rows", 0) > 0:
        print(f"    - Dropzone scan: {pipeline_result.get('dropzone_rows')} rows")
    if pipeline_result.get("selection_rows", 0) > 0:
        print(f"    - Selection scan: {pipeline_result.get('selection_rows')} rows")
    if pipeline_result.get("overlap_count", 0) > 0:
        print(f"    - Selection overlap: {pipeline_result.get('overlap_count')} matching ASINs")
    if pipeline_result.get("overlap_warning"):
        print("    - WARNING: No overlap between Selection and Dropzone")
    if pipeline_result.get("message"):
        for line in str(pipeline_result.get("message", "")).strip().split("\n"):
            if line.strip():
                print(f"    {line}")

    # Summary
    print("\n" + "-" * 60)
    all_passed = (
        dir_result.get("success", False)
        and schema_result.get("success", False)
        and pipeline_result.get("success", False)
    )
    if all_passed:
        print("OVERALL STATUS: [PASS] ALL CHECKS PASSED")
    else:
        print("OVERALL STATUS: [FAIL] SOME CHECKS FAILED")
    print("=" * 60 + "\n")


def main() -> int:
    """Main execution function. Returns exit code (0 for success, 1 for failure)."""
    print("Running diagnostic checks...")

    # Run tests sequentially
    dir_result = check_directories()
    schema_result = validate_schemas()
    pipeline_result = test_pipeline_integrity()

    # Print health report
    print_health_report(dir_result, schema_result, pipeline_result)

    # Determine exit code
    all_passed = (
        dir_result.get("success", False)
        and schema_result.get("success", False)
        and pipeline_result.get("success", False)
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
