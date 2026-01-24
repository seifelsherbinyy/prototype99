# Configuration Files

This directory contains JSON configuration files for the WBR Pipeline. The pipeline automatically loads these files if they exist, otherwise falls back to hardcoded defaults in `src/config.py`.

## Configuration Strategy

**Current Implementation:** Load from JSON files if available, otherwise use hardcoded defaults.

The pipeline uses the following priority:
1. **JSON Files** (if they exist in `config/` directory)
2. **Hardcoded Defaults** (defined in `src/config.py`)

This approach provides:
- ✅ Flexibility to modify configuration without code changes
- ✅ Version control for configuration changes
- ✅ Fallback to working defaults if JSON files are missing
- ✅ No breaking changes if JSON structure is invalid

## Configuration Files

### `thresholds.json`
Defines RAG (Red/Amber/Green) thresholds for performance metrics. Percentages use decimal form.

**Structure:**
```json
{
  "thresholds": {
    "Net_PPM": {
      "red": 0.05,
      "amber": 0.15,
      "unit": "%",
      "direction": "higher_is_better"
    }
  }
}
```

**Usage:** Loaded automatically by `src/config.py` at module import time.

### `impact_weights.json`
Defines financial impact weights for priority scoring (Amendment C).

**Structure:**
```json
{
  "weights": {
    "Ordered_Revenue": {
      "weight": 10,
      "rationale": "Direct financial impact - highest priority"
    }
  }
}
```

**Usage:** Loaded automatically by `src/config.py` at module import time.

### `column_mapping.json`
Maps parsed metric names to canonical WBR column names.

**Structure:**
```json
{
  "mappings": {
    "revenue_metrics": {
      "Product GMS": "Ordered_Revenue"
    },
    "unit_metrics": {
      "Net Ordered Units": "Ordered_Units"
    }
  }
}
```

**Note:** The nested structure is automatically flattened when loaded.

**Usage:** Loaded automatically by `src/config.py` at module import time.

## Modifying Configuration

1. **Edit JSON files** in this directory
2. **Restart the pipeline** (configuration is loaded at import time)
3. **Verify changes** by checking the Configuration Log sheet in output workbooks

## Validation

Configuration is validated on pipeline startup. Errors are reported if:
- Threshold values are invalid (e.g., red > amber for higher_is_better metrics)
- Impact weights are negative
- Required fields are missing

## Fallback Behavior

If JSON files are:
- **Missing:** Hardcoded defaults from `src/config.py` are used
- **Invalid:** Hardcoded defaults from `src/config.py` are used (with warning)
- **Corrupted:** Hardcoded defaults from `src/config.py` are used (with error logged)

This ensures the pipeline always has working configuration, even if JSON files are problematic.

## Testing Configuration

To verify configuration loading:

```python
from src.config import THRESHOLDS, IMPACT_WEIGHTS, COLUMN_MAPPING

print(f"Thresholds: {len(THRESHOLDS)}")
print(f"Impact Weights: {len(IMPACT_WEIGHTS)}")
print(f"Column Mappings: {len(COLUMN_MAPPING)}")
```

## Notes

- Configuration is loaded once at module import time
- Changes to JSON files require restarting the pipeline
- JSON files are optional - the pipeline works without them
- Hardcoded defaults serve as the source of truth for structure
