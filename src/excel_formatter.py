"""
Excel Formatter Module - WBR Output Generator with Validation Tab (Amendment A)

Creates formatted Excel workbooks with:
- Weekly Business Review sheet (primary data with RAG formatting)
- Validation Tab (Amendment A - header parsing audit trail)
- Raw Data Archive (unmodified source copy)
- Executive Summary (Phase 2)
- Configuration Log (Phase 2)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xlsxwriter
from xlsxwriter.workbook import Workbook
from xlsxwriter.worksheet import Worksheet

from src.config import (
    OUTPUT_PATH,
    OUTPUT_SETTINGS,
    RAG_COLORS,
    THRESHOLDS,
)

if TYPE_CHECKING:
    from src.header_parser import ParseResult


class ExcelFormatter:
    """
    Creates formatted WBR Excel workbooks.
    
    Implements Amendment A (Validation Tab) for transparent header parsing audit trail.
    """
    
    def __init__(self, output_path: Path | str | None = None):
        """
        Initialize the ExcelFormatter.
        
        Args:
            output_path: Directory for output files. Defaults to OUTPUT_PATH.
        """
        self.output_path = Path(output_path) if output_path else OUTPUT_PATH
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.workbook: Workbook | None = None
        self.formats: dict[str, Any] = {}
    
    def _generate_filename(
        self,
        vendor_code: str = "ALL",
        week_number: int | str = "WXX"
    ) -> str:
        """Generate output filename from pattern."""
        timestamp = datetime.now().strftime(OUTPUT_SETTINGS["timestamp_format"])
        return OUTPUT_SETTINGS["workbook_name_pattern"].format(
            vendor_code=vendor_code,
            week=f"W{week_number}" if isinstance(week_number, int) else week_number,
            timestamp=timestamp
        )
    
    def _setup_formats(self) -> None:
        """Set up cell formats for the workbook."""
        if self.workbook is None:
            return
        
        # Header format
        self.formats["header"] = self.workbook.add_format({
            "bold": True,
            "bg_color": "#4472C4",
            "font_color": "#FFFFFF",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })
        
        # Currency format
        self.formats["currency"] = self.workbook.add_format({
            "num_format": OUTPUT_SETTINGS["currency_format"],
            "border": 1,
        })
        
        # Percentage format
        self.formats["percentage"] = self.workbook.add_format({
            "num_format": OUTPUT_SETTINGS["percentage_format"],
            "border": 1,
        })
        
        # Integer format
        self.formats["integer"] = self.workbook.add_format({
            "num_format": OUTPUT_SETTINGS["integer_format"],
            "border": 1,
        })
        
        # Default format
        self.formats["default"] = self.workbook.add_format({
            "border": 1,
        })
        
        # RAG formats
        self.formats["red"] = self.workbook.add_format({
            "bg_color": RAG_COLORS["RED"],
            "border": 1,
        })
        self.formats["amber"] = self.workbook.add_format({
            "bg_color": RAG_COLORS["AMBER"],
            "border": 1,
        })
        self.formats["green"] = self.workbook.add_format({
            "bg_color": RAG_COLORS["GREEN"],
            "border": 1,
        })
        
        # Validation Tab formats
        self.formats["validation_header"] = self.workbook.add_format({
            "bold": True,
            "bg_color": "#2F5496",
            "font_color": "#FFFFFF",
            "border": 1,
            "align": "center",
            "text_wrap": True,
        })
        self.formats["validation_ok"] = self.workbook.add_format({
            "bg_color": "#FFFFFF",
            "border": 1,
        })
        self.formats["validation_warning"] = self.workbook.add_format({
            "bg_color": RAG_COLORS["AMBER"],
            "border": 1,
        })
        self.formats["validation_error"] = self.workbook.add_format({
            "bg_color": RAG_COLORS["RED"],
            "border": 1,
        })
    
    def _get_column_format(self, column_name: str) -> Any:
        """Get appropriate format for a column based on name."""
        name_lower = column_name.lower()
        
        if any(kw in name_lower for kw in ["revenue", "price", "gms", "cost", "gmm", "cp", "returns"]):
            return self.formats["currency"]
        elif any(kw in name_lower for kw in ["ppm", "margin", "rate", "pct", "%", "wow", "soroos"]):
            return self.formats["percentage"]
        elif any(kw in name_lower for kw in ["units", "gv", "views", "score", "week"]):
            return self.formats["integer"]
        else:
            return self.formats["default"]
    
    def create_wbr_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str = "Weekly Business Review"
    ) -> Worksheet:
        """
        Create the main WBR data sheet.
        
        Args:
            df: Transformed WBR data.
            sheet_name: Name of the worksheet.
        
        Returns:
            The created worksheet.
        """
        if self.workbook is None:
            raise RuntimeError("Workbook not initialized")
        
        ws = self.workbook.add_worksheet(sheet_name)
        
        # Write headers
        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, self.formats["header"])
        
        # Write data
        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                value = row[col_name]
                cell_format = self._get_column_format(col_name)
                
                # Handle NaN/None values
                if pd.isna(value):
                    ws.write_blank(row_idx + 1, col_idx, None, cell_format)
                else:
                    ws.write(row_idx + 1, col_idx, value, cell_format)
        
        # Set column widths
        for col_idx, col_name in enumerate(df.columns):
            # Calculate width based on header and sample data
            max_width = len(str(col_name))
            for value in df[col_name].head(10):
                if pd.notna(value):
                    max_width = max(max_width, len(str(value)))
            ws.set_column(col_idx, col_idx, min(max_width + 2, 30))
        
        # Freeze header row
        ws.freeze_panes(1, 0)
        
        # Add conditional formatting for RAG columns
        self._apply_conditional_formatting(ws, df)
        
        return ws
    
    def _apply_conditional_formatting(
        self,
        ws: Worksheet,
        df: pd.DataFrame
    ) -> None:
        """Apply conditional formatting to RAG-applicable columns."""
        if self.workbook is None:
            return
        
        for col_idx, col_name in enumerate(df.columns):
            # Check if this column has thresholds defined
            threshold_key = None
            for key in THRESHOLDS:
                if key.lower() in col_name.lower().replace("_", ""):
                    threshold_key = key
                    break
            
            if threshold_key is None:
                continue
            
            threshold = THRESHOLDS[threshold_key]
            red_val = threshold["red"]
            amber_val = threshold["amber"]
            direction = threshold.get("direction", "higher_is_better")
            
            # Get column letter
            col_letter = xlsxwriter.utility.xl_col_to_name(col_idx)
            cell_range = f"{col_letter}2:{col_letter}{len(df) + 1}"
            
            if direction == "higher_is_better":
                # Red if below red threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": "<",
                    "value": red_val,
                    "format": self.formats["red"],
                })
                # Amber if below amber threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": "<",
                    "value": amber_val,
                    "format": self.formats["amber"],
                })
                # Green if above amber threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": ">=",
                    "value": amber_val,
                    "format": self.formats["green"],
                })
            else:
                # Lower is better (e.g., SoROOS%)
                # Red if above red threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": ">",
                    "value": red_val,
                    "format": self.formats["red"],
                })
                # Amber if above amber threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": ">",
                    "value": amber_val,
                    "format": self.formats["amber"],
                })
                # Green if below/equal amber threshold
                ws.conditional_format(cell_range, {
                    "type": "cell",
                    "criteria": "<=",
                    "value": amber_val,
                    "format": self.formats["green"],
                })
    
    def create_validation_tab(
        self,
        parse_results: list["ParseResult"],
        sheet_name: str = "Validation Tab"
    ) -> Worksheet:
        """
        Create the Validation Tab (Amendment A).
        
        Shows header parsing audit trail with confidence scoring.
        
        Args:
            parse_results: List of ParseResult objects from header parsing.
            sheet_name: Name of the worksheet.
        
        Returns:
            The created worksheet.
        """
        if self.workbook is None:
            raise RuntimeError("Workbook not initialized")
        
        ws = self.workbook.add_worksheet(sheet_name)
        
        # Define columns
        columns = [
            "Raw_Header_Found",
            "Regex_Pattern_Matched",
            "Parsed_Week",
            "Parsed_Metric",
            "Parsed_Currency_Unit",
            "Mapped_To_WBR_Column",
            "Confidence_Score",
            "Flag_Status",
        ]
        
        # Write headers
        for col_idx, col_name in enumerate(columns):
            ws.write(0, col_idx, col_name, self.formats["validation_header"])
        
        # Write data
        for row_idx, r in enumerate(parse_results):
            # Determine row format based on flag status
            if r.flag_status == "ERROR":
                row_format = self.formats["validation_error"]
            elif r.flag_status == "WARNING" or r.confidence == "LOW":
                row_format = self.formats["validation_warning"]
            else:
                row_format = self.formats["validation_ok"]
            
            ws.write(row_idx + 1, 0, r.raw_header, row_format)
            ws.write(row_idx + 1, 1, r.matched_pattern, row_format)
            ws.write(row_idx + 1, 2, r.time_period, row_format)
            ws.write(row_idx + 1, 3, r.metric_name, row_format)
            ws.write(row_idx + 1, 4, r.currency_unit, row_format)
            ws.write(row_idx + 1, 5, r.mapped_wbr_column, row_format)
            ws.write(row_idx + 1, 6, r.confidence, row_format)
            ws.write(row_idx + 1, 7, r.flag_status, row_format)
        
        # Set column widths
        widths = [40, 25, 20, 35, 15, 30, 15, 12]
        for col_idx, width in enumerate(widths):
            ws.set_column(col_idx, col_idx, width)
        
        # Freeze header row
        ws.freeze_panes(1, 0)
        
        return ws
    
    def create_raw_archive_sheet(
        self,
        raw_df: pd.DataFrame,
        sheet_name: str = "Raw Data Archive"
    ) -> Worksheet:
        """
        Create archive sheet with unmodified source data.
        
        Args:
            raw_df: Original raw DataFrame.
            sheet_name: Name of the worksheet.
        
        Returns:
            The created worksheet.
        """
        if self.workbook is None:
            raise RuntimeError("Workbook not initialized")
        
        ws = self.workbook.add_worksheet(sheet_name)
        
        # Write headers
        for col_idx, col_name in enumerate(raw_df.columns):
            ws.write(0, col_idx, col_name, self.formats["header"])
        
        # Write data (limit for performance if needed)
        max_rows = min(len(raw_df), 10000)  # Limit to 10k rows
        for row_idx in range(max_rows):
            for col_idx, col_name in enumerate(raw_df.columns):
                value = raw_df.iloc[row_idx][col_name]
                if pd.isna(value):
                    ws.write_blank(row_idx + 1, col_idx, None, self.formats["default"])
                else:
                    ws.write(row_idx + 1, col_idx, value, self.formats["default"])
        
        # Set column widths
        for col_idx, col_name in enumerate(raw_df.columns):
            ws.set_column(col_idx, col_idx, min(len(str(col_name)) + 2, 25))
        
        # Freeze header row
        ws.freeze_panes(1, 0)
        
        return ws
    
    def create_executive_summary(
        self,
        top_issues: list[dict[str, Any]],
        summary_stats: dict[str, Any],
        wbr_data: pd.DataFrame | None = None,
        week_number: int | str = "WXX",
        sheet_name: str = "Executive Summary"
    ) -> Worksheet:
        """
        Create the Executive Summary sheet (Phase 2).
        
        Args:
            top_issues: List of top issues from insight generator.
            summary_stats: Summary statistics dictionary.
            week_number: Week number for header.
            sheet_name: Name of the worksheet.
        
        Returns:
            The created worksheet.
        """
        if self.workbook is None:
            raise RuntimeError("Workbook not initialized")
        
        ws = self.workbook.add_worksheet(sheet_name)
        
        # Title format
        title_format = self.workbook.add_format({
            "bold": True,
            "font_size": 16,
            "font_color": "#2F5496",
        })
        subtitle_format = self.workbook.add_format({
            "bold": True,
            "font_size": 12,
            "bg_color": "#4472C4",
            "font_color": "#FFFFFF",
        })
        label_format = self.workbook.add_format({
            "bold": True,
        })
        
        # Title
        ws.write(0, 0, f"Weekly Business Review - Week {week_number}", title_format)
        ws.write(1, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Key Metrics Section
        row = 4
        ws.write(row, 0, "KEY METRICS OVERVIEW", subtitle_format)
        ws.merge_range(row, 0, row, 3, "KEY METRICS OVERVIEW", subtitle_format)
        
        row += 2
        # Write column headers
        ws.write(row, 0, "Metric", label_format)
        ws.write(row, 1, "Total/Average", label_format)
        ws.write(row, 2, "Min", label_format)
        ws.write(row, 3, "Max", label_format)
        row += 1
        
        metrics = summary_stats.get("metrics", {})
        if metrics:
            for metric_name, stats in metrics.items():
                ws.write(row, 0, metric_name, self.formats["default"])
                
                # Format value based on metric type
                if stats.get("sum") is not None and stats.get("sum", 0) > 100:
                    # For revenue/unit metrics, show total
                    ws.write(row, 1, f"{stats['sum']:,.0f}", self.formats["currency"] if "$" in metric_name or "Revenue" in metric_name else self.formats["integer"])
                elif stats.get("mean") is not None:
                    # For percentage/rate metrics, show average
                    ws.write(row, 1, f"{stats['mean']:.2f}", self.formats["percentage"] if "%" in metric_name or "PPM" in metric_name or "Rate" in metric_name else self.formats["default"])
                else:
                    ws.write(row, 1, "N/A", self.formats["default"])
                
                # Min/Max formatting
                if stats.get("min") is not None:
                    min_val = stats['min']
                    if isinstance(min_val, (int, float)):
                        if abs(min_val) > 1:
                            ws.write(row, 2, f"{min_val:,.0f}", self.formats["default"])
                        else:
                            ws.write(row, 2, f"{min_val:.2f}", self.formats["default"])
                    else:
                        ws.write(row, 2, str(min_val), self.formats["default"])
                else:
                    ws.write(row, 2, "N/A", self.formats["default"])
                
                if stats.get("max") is not None:
                    max_val = stats['max']
                    if isinstance(max_val, (int, float)):
                        if abs(max_val) > 1:
                            ws.write(row, 3, f"{max_val:,.0f}", self.formats["default"])
                        else:
                            ws.write(row, 3, f"{max_val:.2f}", self.formats["default"])
                    else:
                        ws.write(row, 3, str(max_val), self.formats["default"])
                else:
                    ws.write(row, 3, "N/A", self.formats["default"])
                
                row += 1
        else:
            ws.write(row, 0, "No metrics available", self.formats["default"])
            row += 1
        
        # RAG Distribution Section
        row += 2
        ws.write(row, 0, "RAG STATUS DISTRIBUTION", subtitle_format)
        ws.merge_range(row, 0, row, 4, "RAG STATUS DISTRIBUTION", subtitle_format)
        
        row += 2
        # Write column headers
        ws.write(row, 0, "Metric", label_format)
        ws.write(row, 1, "RED", label_format)
        ws.write(row, 2, "AMBER", label_format)
        ws.write(row, 3, "GREEN", label_format)
        ws.write(row, 4, "Total", label_format)
        row += 1
        
        rag_dist = summary_stats.get("rag_distribution", {})
        if rag_dist:
            for metric_name, dist in rag_dist.items():
                ws.write(row, 0, metric_name, self.formats["default"])
                
                # Get counts for each status
                red_count = dist.get("RED", 0)
                amber_count = dist.get("AMBER", 0)
                green_count = dist.get("GREEN", 0)
                total_count = red_count + amber_count + green_count
                
                ws.write(row, 1, red_count, self.formats["red"])
                ws.write(row, 2, amber_count, self.formats["amber"])
                ws.write(row, 3, green_count, self.formats["green"])
                ws.write(row, 4, total_count, self.formats["default"])
                row += 1
        else:
            ws.write(row, 0, "No RAG data available", self.formats["default"])
            row += 1
        
        # Vendor Segmentation Section (if portfolio)
        if wbr_data is not None and "Vendor_Code" in wbr_data.columns:
            vendors = wbr_data[wbr_data["Vendor_Code"] != ""]["Vendor_Code"].unique()
            if len(vendors) > 1:
                row += 2
                ws.write(row, 0, "VENDOR PERFORMANCE COMPARISON", subtitle_format)
                ws.merge_range(row, 0, row, 5, "VENDOR PERFORMANCE COMPARISON", subtitle_format)
                
                row += 2
                headers = ["Vendor", "ASINs", "Revenue", "Units", "Avg PPM", "Trend"]
                for col, header in enumerate(headers):
                    ws.write(row, col, header, label_format)
                row += 1
                
                for vendor in sorted(vendors):
                    vendor_df = wbr_data[wbr_data["Vendor_Code"] == vendor]
                    if vendor_df.empty:
                        continue
                    
                    ws.write(row, 0, vendor, self.formats["default"])
                    ws.write(row, 1, vendor_df["ASIN"].nunique(), self.formats["default"])
                    
                    if "Ordered_Revenue" in vendor_df.columns:
                        revenue = vendor_df["Ordered_Revenue"].sum()
                        ws.write(row, 2, f"{revenue:,.0f}", self.formats["currency"])
                    else:
                        ws.write(row, 2, "N/A", self.formats["default"])
                    
                    if "Ordered_Units" in vendor_df.columns:
                        units = vendor_df["Ordered_Units"].sum()
                        ws.write(row, 3, f"{units:,.0f}", self.formats["integer"])
                    else:
                        ws.write(row, 3, "N/A", self.formats["default"])
                    
                    if "Net_PPM" in vendor_df.columns:
                        avg_ppm = vendor_df["Net_PPM"].mean()
                        ws.write(row, 4, f"{avg_ppm:.2f}%", self.formats["percentage"])
                    else:
                        ws.write(row, 4, "N/A", self.formats["default"])
                    
                    if "Trend_Direction" in vendor_df.columns:
                        trend = vendor_df["Trend_Direction"].mode()
                        trend_str = trend.iloc[0] if len(trend) > 0 else "N/A"
                        ws.write(row, 5, trend_str, self.formats["default"])
                    else:
                        ws.write(row, 5, "N/A", self.formats["default"])
                    
                    row += 1
        
        # Statistical Insights Section
        if summary_stats.get("statistical_insights"):
            row += 2
            ws.write(row, 0, "STATISTICAL INSIGHTS", subtitle_format)
            ws.merge_range(row, 0, row, 6, "STATISTICAL INSIGHTS", subtitle_format)
            
            row += 2
            headers = ["Metric", "P25", "P50", "P75", "P90", "Mean", "Outliers"]
            for col, header in enumerate(headers):
                ws.write(row, col, header, label_format)
            row += 1
            
            stat_insights = summary_stats["statistical_insights"]
            for metric, stats in stat_insights.items():
                ws.write(row, 0, metric, self.formats["default"])
                
                percentiles = stats.get("percentiles", {})
                ws.write(row, 1, f"{percentiles.get('p25', 0):,.0f}", self.formats["default"])
                ws.write(row, 2, f"{percentiles.get('p50', 0):,.0f}", self.formats["default"])
                ws.write(row, 3, f"{percentiles.get('p75', 0):,.0f}", self.formats["default"])
                ws.write(row, 4, f"{percentiles.get('p90', 0):,.0f}", self.formats["default"])
                ws.write(row, 5, f"{stats.get('mean', 0):,.0f}", self.formats["default"])
                
                outliers = stats.get("outliers", {})
                outlier_pct = outliers.get("percentage", 0)
                ws.write(row, 6, f"{outlier_pct:.1f}%", self.formats["default"])
                
                row += 1
        
        # Top Issues Section
        row += 2
        ws.write(row, 0, "TOP 3 PRIORITY ISSUES", subtitle_format)
        ws.merge_range(row, 0, row, 5, "TOP 3 PRIORITY ISSUES", subtitle_format)
        
        row += 2
        if top_issues:
            # Headers
            headers = ["Rank", "ASIN", "Metric", "Value", "Status", "Commentary"]
            for col, header in enumerate(headers):
                ws.write(row, col, header, self.formats["header"])
            row += 1
            
            for i, issue in enumerate(top_issues[:3]):
                ws.write(row, 0, f"#{i+1}", self.formats["default"])
                ws.write(row, 1, issue.get("ASIN", ""), self.formats["default"])
                ws.write(row, 2, issue.get("metric", ""), self.formats["default"])
                ws.write(row, 3, f"{issue.get('current_value', 0):,.2f}", self.formats["default"])
                
                rag = issue.get("rag_status", "N/A")
                rag_fmt = self.formats.get(rag.lower(), self.formats["default"])
                ws.write(row, 4, rag, rag_fmt)
                
                ws.write(row, 5, issue.get("commentary", "")[:100], self.formats["default"])
                row += 1
        else:
            ws.write(row, 0, "No critical issues identified. Performance stable.", self.formats["green"])
        
        # Set column widths
        ws.set_column(0, 0, 25)
        ws.set_column(1, 1, 15)
        ws.set_column(2, 2, 20)
        ws.set_column(3, 3, 15)
        ws.set_column(4, 4, 10)
        ws.set_column(5, 5, 60)
        
        return ws
    
    def create_configuration_log(
        self,
        config: dict[str, Any],
        sheet_name: str = "Configuration Log"
    ) -> Worksheet:
        """
        Create Configuration Log sheet showing settings used.
        
        Args:
            config: Configuration dictionary.
            sheet_name: Name of the worksheet.
        
        Returns:
            The created worksheet.
        """
        if self.workbook is None:
            raise RuntimeError("Workbook not initialized")
        
        ws = self.workbook.add_worksheet(sheet_name)
        
        ws.write(0, 0, "Configuration Log", self.formats["header"])
        ws.write(0, 1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.formats["default"])
        
        row = 2
        ws.write(row, 0, "Setting", self.formats["header"])
        ws.write(row, 1, "Value", self.formats["header"])
        
        row += 1
        for key, value in config.items():
            if isinstance(value, dict):
                ws.write(row, 0, key, self.formats["default"])
                ws.write(row, 1, str(value)[:100], self.formats["default"])
            else:
                ws.write(row, 0, key, self.formats["default"])
                ws.write(row, 1, str(value), self.formats["default"])
            row += 1
        
        ws.set_column(0, 0, 30)
        ws.set_column(1, 1, 80)
        
        return ws
    
    def create_wbr_workbook(
        self,
        wbr_data: pd.DataFrame,
        parse_results: list["ParseResult"],
        raw_data: pd.DataFrame | None = None,
        top_issues: list[dict[str, Any]] | None = None,
        summary_stats: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        vendor_code: str = "ALL",
        week_number: int | str = "WXX",
        output_filename: str | None = None
    ) -> Path:
        """
        Create a complete WBR workbook with all sheets.
        
        Args:
            wbr_data: Transformed WBR data.
            parse_results: List of ParseResult objects for Validation Tab.
            raw_data: Optional raw data for archive sheet.
            top_issues: Optional top issues for Executive Summary.
            summary_stats: Optional summary statistics for Executive Summary.
            config: Optional configuration for Config Log.
            vendor_code: Vendor code for filename.
            week_number: Week number for filename.
            output_filename: Custom output filename. If None, auto-generated.
        
        Returns:
            Path to the created workbook.
        """
        # Generate filename
        if output_filename is None:
            output_filename = self._generate_filename(vendor_code, week_number)
        
        output_path = self.output_path / output_filename
        
        # Create workbook
        self.workbook = xlsxwriter.Workbook(str(output_path))
        self._setup_formats()
        
        try:
            # Create Executive Summary first (if data provided)
            if top_issues is not None and summary_stats is not None:
                self.create_executive_summary(top_issues, summary_stats, wbr_data, week_number, "Executive Summary")
            
            # Create main WBR sheet
            self.create_wbr_sheet(wbr_data, "Weekly Business Review")
            
            # Create Validation Tab (Amendment A)
            self.create_validation_tab(parse_results, "Validation Tab")
            
            # Create Raw Data Archive (if provided)
            if raw_data is not None:
                self.create_raw_archive_sheet(raw_data, "Raw Data Archive")
            
            # Create Configuration Log (if config provided)
            if config is not None:
                self.create_configuration_log(config, "Configuration Log")
            
        finally:
            self.workbook.close()
            self.workbook = None
        
        return output_path


def create_wbr_workbook(
    wbr_data: pd.DataFrame,
    parse_results: list["ParseResult"],
    raw_data: pd.DataFrame | None = None,
    top_issues: list[dict[str, Any]] | None = None,
    summary_stats: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    vendor_code: str = "ALL",
    week_number: int | str = "WXX",
    output_path: Path | str | None = None,
    output_filename: str | None = None
) -> Path:
    """
    Convenience function to create a WBR workbook.
    
    Args:
        wbr_data: Transformed WBR data.
        parse_results: List of ParseResult objects for Validation Tab.
        raw_data: Optional raw data for archive sheet.
        top_issues: Optional top issues for Executive Summary.
        summary_stats: Optional summary statistics for Executive Summary.
        config: Optional configuration for Config Log.
        vendor_code: Vendor code for filename.
        week_number: Week number for filename.
        output_path: Output directory.
        output_filename: Custom output filename.
    
    Returns:
        Path to the created workbook.
    """
    formatter = ExcelFormatter(output_path)
    return formatter.create_wbr_workbook(
        wbr_data,
        parse_results,
        raw_data,
        top_issues,
        summary_stats,
        config,
        vendor_code,
        week_number,
        output_filename
    )


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Excel Formatter Module Test")
    print("=" * 60)
    
    from src.data_loader import detect_latest_file, load_raw_data, load_reference_data
    from src.header_parser import HeaderParser
    from src.data_transformer import transform_raw_to_wbr
    from src.config import DROPZONE_PATH
    
    # Load and process data
    print("\n1. Loading data...")
    latest = detect_latest_file(DROPZONE_PATH)
    if latest:
        raw_df, metadata = load_raw_data(latest)
        print(f"   Loaded: {metadata['filename']}")
        
        # Parse headers
        print("\n2. Parsing headers...")
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        print(f"   Parsed {len(parse_results)} headers")
        
        # Transform data
        print("\n3. Transforming data...")
        ref_data = load_reference_data()
        wbr_df = transform_raw_to_wbr(
            raw_df,
            parse_results,
            vendor_map=ref_data["vendor_map"],
            asin_selection=ref_data["asin_selection"],
            unpivot=True
        )
        print(f"   Transformed: {len(wbr_df)} rows")
        
        # Get week number
        week = wbr_df["Week_Number"].iloc[0] if "Week_Number" in wbr_df.columns and len(wbr_df) > 0 else "WXX"
        
        # Create workbook
        print("\n4. Creating Excel workbook...")
        output_path = create_wbr_workbook(
            wbr_df,
            parse_results,
            raw_data=raw_df,
            vendor_code="TEST",
            week_number=week
        )
        print(f"   Created: {output_path}")
    else:
        print("   No data files found!")
    
    print("\n" + "=" * 60)
    print("Excel Formatter Module Test Complete")
    print("=" * 60)
