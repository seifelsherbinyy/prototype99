"""
Vendor Detection Module - Multi-Layer Vendor Identification

Detects vendor scope from files using:
1. Filename analysis (extract vendor code from filename)
2. ASIN-to-vendor mapping (using vendor selection files)
3. Data analysis (count unique vendors in data)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import SELECTION_PATH
from src.file_loader import ALLOWED_SUFFIXES, load_file


def detect_vendor_from_filename(filename: str) -> str | None:
    """
    Extract vendor code from filename.
    
    Examples:
        "KY2O0-W49-W52.csv" -> "KY2O0"
        "W1-ALL-brightweek_AE_2026023.csv" -> None (portfolio file)
        "SWM6A-Nov2-23-2025.csv" -> "SWM6A"
    
    Args:
        filename: Filename to parse.
    
    Returns:
        Vendor code if found, None if portfolio/ALL file.
    """
    # Pattern: vendor code is typically 5 characters (letters/numbers) at start
    # followed by a dash (unless it's "ALL" or similar)
    pattern = re.compile(r"^([A-Z0-9]{5})-[^-]")
    match = pattern.match(filename)
    
    if match:
        vendor_code = match.group(1)
        # Skip generic codes like "ALL", "W1", "W2"
        if vendor_code not in ("ALL", "W1", "W2", "W3", "W4", "W5"):
            return vendor_code
    
    return None


def load_vendor_selections(selection_path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load vendor selection files from vendors/ subdirectory.
    
    Each vendor selection file contains ASINs for that vendor.
    Filename format: {VendorName}-selection-*.csv
    
    Args:
        selection_path: Path to selection directory. If None, uses default.
    
    Returns:
        Dictionary mapping vendor codes to DataFrames with ASINs.
    """
    if selection_path is None:
        selection_path = SELECTION_PATH
    
    selection_path = Path(selection_path)
    vendors_dir = selection_path / "vendors"
    
    vendor_selections = {}
    
    if not vendors_dir.exists():
        return vendor_selections
    
    # Known vendor code mappings from filename patterns
    vendor_code_map = {
        "AlMaya": ["722XP", "4Y29Z", "NJ6XI"],
        "LinkMax": ["SWM6A"],
        "Champions": ["T072Y"],
        "McLane": ["KY2O0"],
        "WJTowell": ["WJTP1", "WJTP6"],
    }
    
    for file in vendors_dir.iterdir():
        if file.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        
        filename_lower = file.name.lower()
        
        # Try to identify vendor from filename
        vendor_name = None
        vendor_codes = []
        
        for name, codes in vendor_code_map.items():
            if name.lower() in filename_lower:
                vendor_name = name
                vendor_codes = codes
                break
        
        # Load the file
        try:
            df = load_file(file)
            
            # Handle multi-sheet Excel files
            if isinstance(df, dict):
                df = list(df.values())[0] if df else None
                if df is None:
                    continue
            
            # Find ASIN column (case-insensitive)
            # Look for ASIN, matched_asin, or matched_asin#1.value patterns
            asin_cols = []
            for c in df.columns:
                c_lower = c.lower()
                if c.upper() == "ASIN" or "matched_asin" in c_lower:
                    asin_cols.append(c)
            
            if not asin_cols:
                continue
            
            asin_col = asin_cols[0]
            asin_df = df[[asin_col]].rename(columns={asin_col: "ASIN"})
            asin_df["ASIN"] = asin_df["ASIN"].astype(str)
            # Remove any NaN or empty ASINs
            asin_df = asin_df[asin_df["ASIN"].notna() & (asin_df["ASIN"] != "")]
            
            # Store for each vendor code
            if vendor_codes:
                for code in vendor_codes:
                    vendor_selections[code] = asin_df
            elif vendor_name:
                # If we found vendor name but no codes, try to infer from data
                # For now, store with vendor name as key
                vendor_selections[vendor_name] = asin_df
        
        except Exception:
            # Skip files that can't be loaded
            continue
    
    return vendor_selections


def map_asins_to_vendors(
    df: pd.DataFrame,
    vendor_selections: dict[str, pd.DataFrame],
    vendor_map: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add Vendor_Code column by matching ASINs to vendor selection files.
    
    Args:
        df: DataFrame with ASIN column.
        vendor_selections: Dictionary mapping vendor codes to ASIN DataFrames.
        vendor_map: Optional vendor map for vendor name lookup.
    
    Returns:
        DataFrame with Vendor_Code and Vendor_Name columns added.
    """
    result = df.copy()
    
    # Ensure ASIN column exists
    if "ASIN" not in result.columns:
        return result
    
    # Initialize vendor columns
    if "Vendor_Code" not in result.columns:
        result["Vendor_Code"] = ""
    if "Vendor_Name" not in result.columns:
        result["Vendor_Name"] = ""
    
    # Map ASINs to vendors
    result["ASIN"] = result["ASIN"].astype(str)
    
    for vendor_code, asin_df in vendor_selections.items():
        if asin_df.empty or "ASIN" not in asin_df.columns:
            continue
        
        vendor_asins = set(asin_df["ASIN"].astype(str))
        mask = result["ASIN"].isin(vendor_asins)
        
        # Only assign if not already assigned (first match wins)
        result.loc[mask & (result["Vendor_Code"] == ""), "Vendor_Code"] = vendor_code
    
    # Map vendor codes to names if vendor_map provided
    if vendor_map is not None and not vendor_map.empty:
        # Check if vendor_map has vendor_code and vendor_name columns
        vendor_code_col = None
        vendor_name_col = None
        
        for col in vendor_map.columns:
            col_lower = col.lower()
            if "vendor_code" in col_lower or "code" in col_lower:
                vendor_code_col = col
            if "vendor_name" in col_lower or "name" in col_lower:
                vendor_name_col = col
        
        if vendor_code_col and vendor_name_col:
            vendor_name_map = dict(zip(
                vendor_map[vendor_code_col].astype(str),
                vendor_map[vendor_name_col].astype(str)
            ))
            
            result["Vendor_Name"] = result["Vendor_Code"].map(vendor_name_map).fillna("")
    
    return result


def identify_file_scope(
    df: pd.DataFrame,
    filename: str | None = None,
    vendor_selections: dict[str, pd.DataFrame] | None = None
) -> dict[str, Any]:
    """
    Identify if file contains single vendor or portfolio data.
    
    Args:
        df: DataFrame with Vendor_Code column (after mapping).
        filename: Optional filename for filename-based detection.
        vendor_selections: Optional vendor selections for mapping.
    
    Returns:
        Dictionary with:
        - type: 'single' or 'portfolio'
        - vendors: List of vendor codes found
        - vendor_counts: Dict of {vendor_code: count}
        - detected_from: 'filename' or 'data' or 'both'
    """
    result = {
        "type": "unknown",
        "vendors": [],
        "vendor_counts": {},
        "detected_from": "unknown",
    }
    
    # Layer 1: Filename analysis
    filename_vendor = None
    if filename:
        filename_vendor = detect_vendor_from_filename(filename)
        if filename_vendor:
            result["detected_from"] = "filename"
    
    # Layer 2: Data analysis
    if "Vendor_Code" in df.columns:
        # Get unique vendors (excluding empty strings)
        vendors = df[df["Vendor_Code"] != ""]["Vendor_Code"].unique().tolist()
        vendors = [v for v in vendors if v and str(v).strip()]
        
        if vendors:
            result["vendors"] = sorted(vendors)
            result["vendor_counts"] = df[df["Vendor_Code"] != ""]["Vendor_Code"].value_counts().to_dict()
            
            if result["detected_from"] == "unknown":
                result["detected_from"] = "data"
            elif filename_vendor and filename_vendor in vendors:
                result["detected_from"] = "both"
    
    # Determine type
    if len(result["vendors"]) == 0:
        result["type"] = "unknown"
    elif len(result["vendors"]) == 1:
        result["type"] = "single"
        # If filename suggests single vendor, use that
        if filename_vendor and filename_vendor in result["vendors"]:
            result["vendors"] = [filename_vendor]
    else:
        result["type"] = "portfolio"
    
    return result
