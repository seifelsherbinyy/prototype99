# Archive Directory

This directory contains files and modules that are not needed for the WBR Pipeline development but are preserved for reference or potential future use.

## Archive Structure

### `demo/`
- `showcase_src.py` - Demo script showcasing existing src modules (not needed for production WBR pipeline)

### `streamlit_app/`
- `app.py` - Streamlit application for profitability dashboard (different use case from WBR)
- `visuals.py` - Streamlit-specific visualization module (not needed for WBR Excel output)

### `tools/`
- `reviewer.py` - Diagnostic tool for pipeline validation (not needed for WBR pipeline)

### `logs/`
- `system_debug.log` - Previous execution log file (can be regenerated)

### `legacy_data/`
- `data/` - Legacy data directory structure (duplicate of `01_dropzone/` and `00_selection/`)

## Restoration

If any of these files are needed, they can be restored by moving them back to their original locations:
- `showcase_src.py` → root directory
- `app.py` → root directory
- `src/visuals.py` → `src/` directory
- `src/reviewer.py` → `src/` directory
- `data/` → root directory

## Archive Date
January 23, 2026
