"""
Data ingestion module.

Scans dropzone and selection directories, loads files, and applies normalization.
Provides unified DataFrames for DuckDB registration.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.file_loader import ALLOWED_SUFFIXES, load_file
from src.logger import debug_watcher, get_logger
from src.normalization import normalize_data

if TYPE_CHECKING:
    from collections.abc import Sequence

# Get logger instance
logger = get_logger(__name__)

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
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _is_hidden_file(path: Path) -> bool:
    """Check if file should be skipped (hidden/system files)."""
    name = path.name.lower()
    return name.startswith(".") or name in ["thumbs.db", "desktop.ini"]


@debug_watcher
def scan_dropzone(path: str | Path = "01_dropzone") -> pd.DataFrame:
    """
    Scan dropzone directory recursively and load all data files.

    Applies normalization to each file and concatenates into a single DataFrame.

    Args:
        path: Path to dropzone directory (default: "01_dropzone").

    Returns:
        Unified DataFrame with canonical schema, ready for DuckDB registration.
    """
    dropzone_path = Path(path)
    if not dropzone_path.exists():
        logger.warning(f"Dropzone directory not found: {dropzone_path}")
        return pd.DataFrame()

    if not dropzone_path.is_dir():
        logger.error(f"Path is not a directory: {dropzone_path}")
        return pd.DataFrame()

    all_dataframes = []
    files_processed = 0
    files_failed = 0

    # Recursively find all supported files
    for file_path in dropzone_path.rglob("*"):
        if not file_path.is_file():
            continue

        if _is_hidden_file(file_path):
            continue

        if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        try:
            # Load file
            df = _load_file(file_path)
            if df is None or df.empty:
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            # Normalize data
            normalized = normalize_data(df, source_file=file_path)

            if normalized.empty:
                logger.debug(f"Normalization produced empty DataFrame for: {file_path}")
                continue

            all_dataframes.append(normalized)
            files_processed += 1
            logger.info(f"Processed: {file_path} ({len(normalized)} rows)")

        except PermissionError as e:
            logger.warning(f"Permission denied for {file_path}: {e}")
            files_failed += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            files_failed += 1

    if not all_dataframes:
        logger.warning("No data files found or successfully processed in dropzone")
        return pd.DataFrame()

    # Concatenate all DataFrames
    try:
        combined = pd.concat(all_dataframes, ignore_index=True)
        logger.info(
            f"Dropzone scan complete: {files_processed} files processed, "
            f"{files_failed} failed, {len(combined)} total rows"
        )
        return combined
    except Exception as e:
        logger.error(f"Failed to concatenate DataFrames: {e}", exc_info=True)
        return pd.DataFrame()


def scan_selection(path: str | Path = "00_selection") -> dict[str, pd.DataFrame | list[pd.DataFrame]]:
    """
    Scan selection directory for reference data files.

    Loads vendor maps and vendor selection files as separate DataFrames.
    These are reference/auxiliary data and are NOT normalized to canonical schema.

    Args:
        path: Path to selection directory (default: "00_selection").

    Returns:
        Dictionary with keys:
        - "vendor_map": DataFrame with vendor code mappings
        - "vendor_selections": List of DataFrames from vendors/ subdirectory
        - "ref_selection": Combined DataFrame suitable for ref_selection table
    """
    selection_path = Path(path)
    if not selection_path.exists():
        logger.warning(f"Selection directory not found: {selection_path}")
        return {
            "vendor_map": pd.DataFrame(),
            "vendor_selections": [],
            "ref_selection": pd.DataFrame(),
        }

    result: dict[str, pd.DataFrame | list[pd.DataFrame]] = {
        "vendor_map": pd.DataFrame(),
        "vendor_selections": [],
        "ref_selection": pd.DataFrame(),
    }

    # Load vendor_map.csv or vendor_map.xlsx
    vendor_map_paths = [
        selection_path / "vendor_map.csv",
        selection_path / "vendor_map.xlsx",
    ]

    for vendor_map_path in vendor_map_paths:
        if vendor_map_path.exists():
            try:
                df = _load_file(vendor_map_path)
                if df is not None and not df.empty:
                    result["vendor_map"] = df
                    logger.info(f"Loaded vendor map: {vendor_map_path} ({len(df)} rows)")
                    break
            except Exception as e:
                logger.warning(f"Failed to load vendor map {vendor_map_path}: {e}")

    # Load vendor selection files from vendors/ subdirectory
    vendors_dir = selection_path / "vendors"
    if vendors_dir.exists() and vendors_dir.is_dir():
        vendor_dfs = []
        for file_path in vendors_dir.iterdir():
            if not file_path.is_file():
                continue

            if _is_hidden_file(file_path):
                continue

            if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue

            try:
                df = _load_file(file_path)
                if df is not None and not df.empty:
                    vendor_dfs.append(df)
                    logger.info(f"Loaded vendor selection: {file_path} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Failed to load vendor selection {file_path}: {e}")

        result["vendor_selections"] = vendor_dfs

    # Create ref_selection DataFrame (ASIN watch list)
    # Try to extract ASIN column from vendor selections or vendor map
    ref_selection_dfs = []

    # Check vendor_map for ASIN column
    if isinstance(result["vendor_map"], pd.DataFrame) and not result["vendor_map"].empty:
        if "ASIN" in result["vendor_map"].columns:
            ref_selection_dfs.append(result["vendor_map"][["ASIN"]].drop_duplicates())

    # Check vendor selections for ASIN column
    for vendor_df in result["vendor_selections"]:
        if isinstance(vendor_df, pd.DataFrame) and not vendor_df.empty:
            # Look for ASIN column (case-insensitive)
            asin_cols = [col for col in vendor_df.columns if col.upper() == "ASIN"]
            if asin_cols:
                ref_selection_dfs.append(vendor_df[[asin_cols[0]]].rename(columns={asin_cols[0]: "ASIN"}).drop_duplicates())

    # Combine all ASIN lists
    if ref_selection_dfs:
        try:
            result["ref_selection"] = pd.concat(ref_selection_dfs, ignore_index=True).drop_duplicates(subset=["ASIN"])
            logger.info(f"Created ref_selection with {len(result['ref_selection'])} unique ASINs")
        except Exception as e:
            logger.warning(f"Failed to combine ref_selection DataFrames: {e}")
            result["ref_selection"] = pd.DataFrame(columns=["ASIN"])

    return result
