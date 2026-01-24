"""
Generate vendor_map.csv from filenames and existing data.

Scans dropzone files and selection files to extract vendor codes,
then generates or updates vendor_map.csv with vendor code to name mappings.
"""

import sys
from pathlib import Path
import re
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import DROPZONE_PATH, SELECTION_PATH


# Known vendor code to name mappings (can be extended)
KNOWN_VENDOR_NAMES = {
    "WJTP1": "WJTowell",
    "WJTP6": "WJTowell",
    "722XP": "AlMaya",
    "4Y29Z": "AlMaya",
    "NJ6XI": "AlMaya",
    "SWM6A": "LinkMax",
    "T072Y": "Champions",
    "KY2O0": "McLane",
}


def extract_vendor_codes_from_filenames(directory: Path) -> set[str]:
    """
    Extract vendor codes from filenames in a directory.
    
    Vendor codes are typically at the start of filenames before the first dash.
    Examples:
        - "4Y29Z-W49-W52.csv" -> "4Y29Z"
        - "KY2O0-Nov2-23-2025.csv" -> "KY2O0"
        - "W1-ALL-brightweek_AE_2026023-DoNotShareExternally.csv" -> skip (ALL)
    
    Args:
        directory: Directory to scan for files.
    
    Returns:
        Set of vendor codes found.
    """
    vendor_codes = set()
    
    if not directory.exists():
        return vendor_codes
    
    # Pattern: vendor code is typically 5 characters (letters/numbers) at start
    # followed by a dash (unless it's "ALL" or similar)
    pattern = re.compile(r"^([A-Z0-9]{5})-[^-]")
    
    for file_path in directory.rglob("*.csv"):
        filename = file_path.name
        match = pattern.match(filename)
        if match:
            vendor_code = match.group(1)
            # Skip generic codes like "ALL"
            if vendor_code != "ALL":
                vendor_codes.add(vendor_code)
    
    return vendor_codes


def extract_vendor_codes_from_selection_files(selection_path: Path) -> set[str]:
    """
    Extract vendor codes from selection directory files.
    
    Checks vendor selection files and vendor_map files for vendor codes.
    
    Args:
        selection_path: Path to selection directory.
    
    Returns:
        Set of vendor codes found.
    """
    vendor_codes = set()
    
    if not selection_path.exists():
        return vendor_codes
    
    # Check existing vendor_map.csv if it exists
    vendor_map_csv = selection_path / "vendor_map.csv"
    if vendor_map_csv.exists():
        try:
            df = pd.read_csv(vendor_map_csv)
            if "vendor_code" in df.columns:
                vendor_codes.update(df["vendor_code"].dropna().astype(str).unique())
        except Exception:
            pass
    
    # Check vendor_map.xlsx if it exists
    vendor_map_xlsx = selection_path / "vendor_map.xlsx"
    if vendor_map_xlsx.exists():
        try:
            df = pd.read_excel(vendor_map_xlsx)
            if "vendor_code" in df.columns:
                vendor_codes.update(df["vendor_code"].dropna().astype(str).unique())
        except Exception:
            pass
    
    # Check vendor selection files for vendor codes in filenames
    vendors_dir = selection_path / "vendors"
    if vendors_dir.exists():
        pattern = re.compile(r"^([A-Z0-9]+)-")
        for file_path in vendors_dir.glob("*.csv"):
            match = pattern.match(file_path.stem)
            if match:
                vendor_code = match.group(1)
                vendor_codes.add(vendor_code)
    
    return vendor_codes


def generate_vendor_map(
    output_path: Path | None = None,
    preserve_existing: bool = True
) -> pd.DataFrame:
    """
    Generate vendor_map.csv from available sources.
    
    Args:
        output_path: Path to output vendor_map.csv. If None, uses SELECTION_PATH/vendor_map.csv.
        preserve_existing: If True, preserves existing vendor names from current vendor_map.
    
    Returns:
        DataFrame with vendor_code and vendor_name columns.
    """
    if output_path is None:
        output_path = SELECTION_PATH / "vendor_map.csv"
    
    # Collect vendor codes from all sources
    vendor_codes = set()
    
    # From dropzone files
    if DROPZONE_PATH.exists():
        dropzone_codes = extract_vendor_codes_from_filenames(DROPZONE_PATH)
        vendor_codes.update(dropzone_codes)
        print(f"Found {len(dropzone_codes)} vendor codes in dropzone files")
    
    # From selection files
    if SELECTION_PATH.exists():
        selection_codes = extract_vendor_codes_from_selection_files(SELECTION_PATH)
        vendor_codes.update(selection_codes)
        print(f"Found {len(selection_codes)} vendor codes in selection files")
    
    # Load existing vendor_map if it exists and we want to preserve it
    existing_map = {}
    if preserve_existing and output_path.exists():
        try:
            df_existing = pd.read_csv(output_path)
            if "vendor_code" in df_existing.columns and "vendor_name" in df_existing.columns:
                existing_map = dict(zip(
                    df_existing["vendor_code"].astype(str),
                    df_existing["vendor_name"].astype(str)
                ))
                print(f"Loaded {len(existing_map)} existing vendor mappings")
        except Exception as e:
            print(f"Warning: Could not load existing vendor_map: {e}")
    
    # Build vendor map
    vendor_map_data = []
    for vendor_code in sorted(vendor_codes):
        # Use existing name if available, otherwise use known name, otherwise use code as name
        vendor_name = existing_map.get(vendor_code) or KNOWN_VENDOR_NAMES.get(vendor_code) or vendor_code
        vendor_map_data.append({
            "vendor_code": vendor_code,
            "vendor_name": vendor_name,
        })
    
    df = pd.DataFrame(vendor_map_data)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nGenerated vendor_map.csv with {len(df)} vendors")
    print(f"Saved to: {output_path}")
    
    return df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate vendor_map.csv from filenames and existing data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for vendor_map.csv (default: 00_selection/vendor_map.csv)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing vendor names (default: preserve existing names)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    
    print("=" * 60)
    print("Vendor Map Generator")
    print("=" * 60)
    print()
    
    df = generate_vendor_map(
        output_path=output_path,
        preserve_existing=not args.overwrite
    )
    
    print()
    print("Vendor Map Preview:")
    print(df.to_string(index=False))
    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
