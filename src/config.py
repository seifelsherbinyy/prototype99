"""
Configuration Module - Centralized Configuration Hub

Contains all configurable parameters for the WBR Pipeline:
- Directory paths
- RAG thresholds for performance metrics
- Financial impact weights for priority scoring (Amendment C)
- Column mapping configuration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input directories
DROPZONE_PATH = PROJECT_ROOT / "01_dropzone" / "weekly" / "performance"
HISTORICAL_PATH = PROJECT_ROOT / "01_dropzone" / "historical"
SELECTION_PATH = PROJECT_ROOT / "00_selection"

# Output directories
OUTPUT_PATH = PROJECT_ROOT / "02_output"

# Data/persistence directories
DATA_PATH = PROJECT_ROOT / "data"
HISTORY_ARCHIVE_PATH = DATA_PATH / "history.duckdb"

# Configuration file directory
CONFIG_DIR = PROJECT_ROOT / "config"


# ============================================================================
# RAG THRESHOLDS (Red/Amber/Green)
# ============================================================================
# Values below "red" threshold are RED, below "amber" are AMBER, else GREEN
# For metrics where higher is better (Conversion, etc.):
#   value < red -> RED, value < amber -> AMBER, else GREEN
# For metrics where lower is better (decline metrics):
#   Use negative thresholds appropriately
# 
# Note: Thresholds are loaded from JSON files if available (see bottom of file)
# Default values are defined in _THRESHOLDS_DEFAULT below.

# ============================================================================
# FINANCIAL IMPACT WEIGHTS (Amendment C - Priority Scoring)
# ============================================================================
# Higher weight = higher priority when calculating impact scores
# Impact Score = |Percentage_Change| x Metric_Weight x Direction_Multiplier
# Direction_Multiplier: -1 for declines (prioritize problems), +0.5 for improvements
#
# Note: Impact weights are loaded from JSON files if available (see bottom of file)
# Default values are defined in _IMPACT_WEIGHTS_DEFAULT below.

# ============================================================================
# COLUMN MAPPING
# ============================================================================
# Maps parsed metric names to canonical WBR column names
#
# Note: Column mappings are loaded from JSON files if available (see bottom of file)
# Default values are defined in _COLUMN_MAPPING_DEFAULT below.


# ============================================================================
# CRITICAL METRICS (must be present in output)
# ============================================================================

CRITICAL_METRICS = [
    "Ordered_Revenue",
    "Ordered_Units",
    "Net_PPM",
    "Glance_Views",
    "Average_Selling_Price",
]


# ============================================================================
# OUTPUT FORMAT SETTINGS
# ============================================================================

OUTPUT_SETTINGS = {
    "workbook_name_pattern": "WBR_{vendor_code}_{week}_{timestamp}.xlsx",
    "timestamp_format": "%Y%m%d_%H%M%S",
    "currency_format": "#,##0.00",
    "percentage_format": "0.0%",
    "integer_format": "#,##0",
    "decimal_format": "#,##0.00",
}


# ============================================================================
# RAG COLORS (for Excel formatting)
# ============================================================================

RAG_COLORS = {
    "RED": "#FF6B6B",
    "AMBER": "#FFE66D", 
    "GREEN": "#4ECDC4",
    "WHITE": "#FFFFFF",
}


# ============================================================================
# CONFIG LOADING AND VALIDATION
# ============================================================================

def _load_thresholds_from_json(defaults: dict[str, Any]) -> dict[str, Any]:
    """Load thresholds from JSON file, merge with defaults."""
    thresholds_file = CONFIG_DIR / "thresholds.json"
    if thresholds_file.exists():
        try:
            with open(thresholds_file, "r") as f:
                data = json.load(f)
                # Extract thresholds from JSON structure
                if "thresholds" in data:
                    # Merge JSON thresholds with defaults (JSON takes precedence)
                    merged = defaults.copy()
                    merged.update(data["thresholds"])
                    return merged
        except Exception as e:
            # Log error but continue with defaults
            import warnings
            warnings.warn(f"Failed to load thresholds from JSON: {e}. Using defaults.")
    return defaults


def _load_impact_weights_from_json(defaults: dict[str, Any]) -> dict[str, Any]:
    """Load impact weights from JSON file, merge with defaults."""
    weights_file = CONFIG_DIR / "impact_weights.json"
    if weights_file.exists():
        try:
            with open(weights_file, "r") as f:
                data = json.load(f)
                # Extract weights from JSON structure
                if "weights" in data:
                    weights_dict = defaults.copy()
                    for metric, weight_data in data["weights"].items():
                        if isinstance(weight_data, dict) and "weight" in weight_data:
                            weights_dict[metric] = weight_data["weight"]
                        elif isinstance(weight_data, (int, float)):
                            weights_dict[metric] = weight_data
                    return weights_dict
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load impact weights from JSON: {e}. Using defaults.")
    return defaults


def _load_column_mapping_from_json(defaults: dict[str, str]) -> dict[str, str]:
    """Load column mapping from JSON file, merge with defaults."""
    mapping_file = CONFIG_DIR / "column_mapping.json"
    if mapping_file.exists():
        try:
            with open(mapping_file, "r") as f:
                data = json.load(f)
                # Extract mappings from nested JSON structure
                if "mappings" in data:
                    flat_mapping = defaults.copy()
                    for category, mappings in data["mappings"].items():
                        if isinstance(mappings, dict):
                            flat_mapping.update(mappings)
                    return flat_mapping
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load column mapping from JSON: {e}. Using defaults.")
    return defaults


# Load configuration from JSON files if available, otherwise use hardcoded defaults
# This happens at module import time
_THRESHOLDS_DEFAULT = {
    # Profitability thresholds (higher is better)
    "Net_PPM": {
        "red": 0.05,
        "amber": 0.15,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "PPM": {
        "red": 0.0,
        "amber": 0.10,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "Contribution_Margin": {
        "red": -0.10,
        "amber": 0.05,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "WoW_Change": {
        "red": -0.15,
        "amber": -0.05,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "Revenue_WoW": {
        "red": -0.20,
        "amber": -0.10,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "Units_WoW": {
        "red": -0.20,
        "amber": -0.10,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "GlanceViews_WoW": {
        "red": -0.25,
        "amber": -0.10,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "Fill_Rate_Sourceable": {
        "red": 0.70,
        "amber": 0.90,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "Vendor_Confirmation_Rate": {
        "red": 0.70,
        "amber": 0.90,
        "unit": "%",
        "direction": "higher_is_better",
    },
    "SoROOS_Pct": {
        "red": 0.20,
        "amber": 0.10,
        "unit": "%",
        "direction": "lower_is_better",
    },
}

_IMPACT_WEIGHTS_DEFAULT = {
    "Ordered_Revenue": 10,
    "Ordered_Units": 8,
    "Conversion_Rate": 7,
    "Average_Selling_Price": 6,
    "Glance_Views": 4,
    "Sessions": 3,
    "Net_PPM": 7,
    "Fill_Rate_Sourceable": 5,
    "SoROOS_Pct": 5,
}

_COLUMN_MAPPING_DEFAULT = {
    "Product GMS": "Ordered_Revenue",
    "Product Sales": "Ordered_Revenue",
    "List Price": "List_Price",
    "ASP": "Average_Selling_Price",
    "Total GMM": "Total_GMM",
    "Deal GMS": "Deal_GMS",
    "Ordered Product Sales": "Ordered_Revenue",
    "Ordered Sales": "Ordered_Revenue",
    "Ordered Revenue": "Ordered_Revenue",
    "Retail Net OPS": "Ordered_Revenue",
    "All Net OPS": "Ordered_Revenue",
    "3P Net OPS": "Ordered_Revenue",
    "Net Receipts": "Net_Receipts_Revenue",
    "Customer Returns": "Customer_Returns",
    "CP": "Cost_Price",
    "Net Ordered Units": "Ordered_Units",
    "Net Receipts (Units)": "Net_Receipts_Units",
    "SELLABLE_ON_HAND_UNITS": "Sellable_On_Hand_Units",
    "Net PPM": "Net_PPM",
    "PPM": "PPM",
    "CM": "Contribution_Margin",
    "GV": "Glance_Views",
    "Amzn GV": "Amazon_Glance_Views",
    "SoROOS(%)": "SoROOS_Pct",
    "Daily SoRoos%": "Daily_SoRoos_Pct",
    "Conversion Rate": "Conversion_Rate",
    "Conversion Rate(%)": "Conversion_Rate",
    "Unit Session Percentage": "Conversion_Rate",
    "Fill Rate - Sourceable": "Fill_Rate_Sourceable",
    "Vendor Confirmation Rate - Sourceable": "Vendor_Confirmation_Rate",
    "CCOGS As A % Of PCOGS": "CCOGS_As_Pct_Of_PCOGS",
    "External Box Price Competitiveness (% Of GV)": "External_Price_Competitiveness",
    "IDQ Score": "IDQ_Score",
}

# Load from JSON if available, otherwise use defaults
THRESHOLDS = _load_thresholds_from_json(_THRESHOLDS_DEFAULT)
IMPACT_WEIGHTS = _load_impact_weights_from_json(_IMPACT_WEIGHTS_DEFAULT)
COLUMN_MAPPING = _load_column_mapping_from_json(_COLUMN_MAPPING_DEFAULT)


def load_config() -> dict[str, Any]:
    """
    Load and return all configuration as a dictionary.
    
    Returns:
        Dictionary with all configuration values.
    """
    return {
        "project_root": PROJECT_ROOT,
        "dropzone_path": DROPZONE_PATH,
        "historical_path": HISTORICAL_PATH,
        "selection_path": SELECTION_PATH,
        "output_path": OUTPUT_PATH,
        "data_path": DATA_PATH,
        "history_archive_path": HISTORY_ARCHIVE_PATH,
        "config_dir": CONFIG_DIR,
        "thresholds": THRESHOLDS,
        "impact_weights": IMPACT_WEIGHTS,
        "column_mapping": COLUMN_MAPPING,
        "critical_metrics": CRITICAL_METRICS,
        "output_settings": OUTPUT_SETTINGS,
        "rag_colors": RAG_COLORS,
    }


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration settings.
    
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Check directories exist
    if not DROPZONE_PATH.exists():
        errors.append(f"Dropzone path does not exist: {DROPZONE_PATH}")
    
    if not SELECTION_PATH.exists():
        errors.append(f"Selection path does not exist: {SELECTION_PATH}")
    
    # Check vendor_map.csv exists
    vendor_map = SELECTION_PATH / "vendor_map.csv"
    if not vendor_map.exists():
        errors.append(f"Vendor map file not found: {vendor_map}")
    
    # Validate thresholds
    for metric, threshold in THRESHOLDS.items():
        if "red" not in threshold:
            errors.append(f"Missing 'red' threshold for {metric}")
        if "amber" not in threshold:
            errors.append(f"Missing 'amber' threshold for {metric}")
        if threshold.get("red", 0) > threshold.get("amber", 0):
            if threshold.get("direction") == "higher_is_better":
                errors.append(f"Invalid thresholds for {metric}: red > amber for higher_is_better metric")
    
    # Validate impact weights
    for metric, weight in IMPACT_WEIGHTS.items():
        if not isinstance(weight, (int, float)) or weight < 0:
            errors.append(f"Invalid impact weight for {metric}: {weight}")
    
    return len(errors) == 0, errors


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def save_config_to_json(filepath: Path | str | None = None) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        filepath: Output path. If None, saves to CONFIG_DIR/config.json.
    """
    if filepath is None:
        ensure_directories()
        filepath = CONFIG_DIR / "config.json"
    
    filepath = Path(filepath)
    
    # Convert Path objects to strings for JSON serialization
    config = {
        "thresholds": THRESHOLDS,
        "impact_weights": IMPACT_WEIGHTS,
        "column_mapping": COLUMN_MAPPING,
        "critical_metrics": CRITICAL_METRICS,
        "output_settings": OUTPUT_SETTINGS,
        "rag_colors": RAG_COLORS,
    }
    
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Validate configuration on direct execution
    print("=" * 60)
    print("Configuration Validation")
    print("=" * 60)
    
    is_valid, errors = validate_config()
    
    if is_valid:
        print("[OK] Configuration is valid")
    else:
        print("[ERROR] Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    
    print("\nDirectory paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Dropzone: {DROPZONE_PATH}")
    print(f"  Selection: {SELECTION_PATH}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  History: {HISTORY_ARCHIVE_PATH}")
    
    print(f"\nThreshold metrics: {len(THRESHOLDS)}")
    print(f"Impact weight metrics: {len(IMPACT_WEIGHTS)}")
    print(f"Column mappings: {len(COLUMN_MAPPING)}")
