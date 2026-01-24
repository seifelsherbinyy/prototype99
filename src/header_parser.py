"""
Header Parser Module - Intelligent Header Decomposition (Phase 0)

Parses congested Amazon Vendor Central headers into structured metadata
with confidence scoring for the Validation Tab (Amendment A).

Example headers:
- "Product GMS($)(Week 49 )" -> metric="Product GMS", currency="$", week=49
- "Net PPM(Week 49 )(%)" -> metric="Net PPM", currency="%", week=49, is_percentage=True
- "SoROOS(%)(Week 49 )(%)" -> metric="SoROOS(%)", currency="%", week=49, is_percentage=True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

_MONTH_LOOKUP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


@dataclass
class ParseResult:
    """Structured result from parsing a header string."""
    
    raw_header: str
    metric_name: str = ""
    currency_unit: str = ""  # "$", "%", "units", ""
    time_period: str = ""  # "Week 49", "T12M", ""
    period_type: str = ""  # "week", "month", "quarter", "t12m", "daterange", ""
    period_label: str = ""  # "Week 49", "Jan 2025", "Q1 2025", ""
    week_number: int | None = None  # 49, None for T12M or static
    is_percentage: bool = False  # True if ends with (%)
    matched_pattern: str = ""  # Which regex pattern matched
    confidence: str = "ERROR"  # "HIGH", "MEDIUM", "LOW", "ERROR"
    mapped_wbr_column: str = ""  # Destination column name
    flag_status: str = "ERROR"  # "OK", "WARNING", "ERROR"


# Known metric names that map to WBR columns (for HIGH confidence)
KNOWN_METRICS = {
    # Revenue metrics
    "Product GMS": "Ordered_Revenue",
    "Product Sales": "Ordered_Revenue",
    "Ordered Product Sales": "Ordered_Revenue",
    "Ordered Sales": "Ordered_Revenue",
    "Ordered Revenue": "Ordered_Revenue",
    "Retail Net OPS": "Ordered_Revenue",
    "All Net OPS": "Ordered_Revenue",
    "3P Net OPS": "Ordered_Revenue",
    "List Price": "List_Price",
    "ASP": "Average_Selling_Price",
    "Total GMM": "Total_GMM",
    "Deal GMS": "Deal_GMS",
    "Net Receipts": "Net_Receipts_Revenue",
    "Customer Returns": "Customer_Returns",
    "CP": "Cost_Price",
    
    # Unit metrics
    "Net Ordered Units": "Ordered_Units",
    "Net Receipts (Units)": "Net_Receipts_Units",
    "SELLABLE_ON_HAND_UNITS": "Sellable_On_Hand_Units",
    
    # Performance metrics
    "Net PPM": "Net_PPM",
    "PPM": "PPM",
    "CM": "Contribution_Margin",
    "GV": "Glance_Views",
    "Amzn GV": "Amazon_Glance_Views",
    "Sessions": "Sessions",
    
    # Rate metrics
    "Conversion Rate": "Conversion_Rate",
    "Conversion Rate(%)": "Conversion_Rate",
    "Unit Session Percentage": "Conversion_Rate",
    "SoROOS(%)": "SoROOS_Pct",
    "Daily SoRoos%": "Daily_SoRoos_Pct",
    "Fill Rate - Sourceable": "Fill_Rate_Sourceable",
    "Vendor Confirmation Rate - Sourceable": "Vendor_Confirmation_Rate",
    "CCOGS As A % Of PCOGS": "CCOGS_As_Pct_Of_PCOGS",
    "External Box Price Competitiveness (% Of GV)": "External_Price_Competitiveness",
    
    # Score metrics
    "IDQ Score": "IDQ_Score",
}

# Static columns (no time period)
STATIC_COLUMNS = {
    "ASIN": "ASIN",
    "Brand Name": "Brand_Name",
    "SKU": "SKU",
    "Product Title": "Product_Title",
}


class HeaderParser:
    """
    Parses congested Amazon Vendor Central headers into structured metadata.
    
    Supports both weekly (Week N) and historical (T12M) time period formats.
    """
    
    def __init__(self, known_metrics: dict[str, str] | None = None):
        """
        Initialize the HeaderParser.
        
        Args:
            known_metrics: Optional custom metric-to-column mapping.
                          If None, uses default KNOWN_METRICS.
        """
        self.known_metrics = known_metrics or KNOWN_METRICS
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> list[tuple[str, re.Pattern, str]]:
        """
        Compile regex patterns in priority order (most specific first).
        
        Returns:
            List of (pattern_name, compiled_regex, description) tuples.
        """
        month_pattern = r"(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2})"
        quarter_pattern = r"(?:Q|Quarter\s*)(?P<quarter>[1-4])"

        patterns = [
            # Pattern 1: Metric($)(Week N ) - Currency with week
            # e.g., "Product GMS($)(Week 49 )", "List Price($)(Week 50 )"
            (
                "currency_week",
                re.compile(r"^(?P<metric>.+?)\(\$\)\(Week\s*(?P<week>\d+)\s*\)$"),
                "Metric($)(Week N)"
            ),
            
            # Pattern 2: Metric($)(T12M ) - Currency with T12M
            # e.g., "Product GMS($)(T12M )", "CP($)(T12M )"
            (
                "currency_t12m",
                re.compile(r"^(?P<metric>.+?)\(\$\)\(T12M\s*\)$"),
                "Metric($)(T12M)"
            ),
            
            # Pattern 3: Metric($)(MM/DD/YYYY-MM/DD/YYYY) - Currency with date range
            # e.g., "Product GMS($)(11/02/2025-11/09/2025)"
            (
                "currency_daterange",
                re.compile(r"^(?P<metric>.+?)\(\$\)\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)$"),
                "Metric($)(MM/DD/YYYY-MM/DD/YYYY)"
            ),

            # Pattern 3a: Metric($)(Month) - Currency with month
            (
                "currency_month",
                re.compile(rf"^(?P<metric>.+?)\(\$\)\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric($)(Month)"
            ),

            # Pattern 3b: Metric($)(Quarter) - Currency with quarter
            (
                "currency_quarter",
                re.compile(rf"^(?P<metric>.+?)\(\$\)\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric($)(Quarter)"
            ),
            
            # Pattern 4: Metric(%)(Week N )(%) - Percentage in metric name with trailing %
            # e.g., "SoROOS(%)(Week 49 )(%)"
            (
                "pct_metric_week_pct",
                re.compile(r"^(?P<metric>.+?\(%\))\(Week\s*(?P<week>\d+)\s*\)\(%\)$"),
                "Metric(%)(Week N)(%)"
            ),
            
            # Pattern 5: Metric(%)(T12M )(%) - Percentage in metric name with T12M
            # e.g., "SoROOS(%)(T12M )(%)"
            (
                "pct_metric_t12m_pct",
                re.compile(r"^(?P<metric>.+?\(%\))\(T12M\s*\)\(%\)$"),
                "Metric(%)(T12M)(%)"
            ),
            
            # Pattern 6: Metric(%)(MM/DD/YYYY-MM/DD/YYYY)(%) - Pct metric with date range
            # e.g., "SoROOS(%)(11/02/2025-11/09/2025)(%)"
            (
                "pct_metric_daterange_pct",
                re.compile(r"^(?P<metric>.+?\(%\))\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)\(%\)$"),
                "Metric(%)(MM/DD/YYYY-MM/DD/YYYY)(%)"
            ),
            
            # Pattern 7: Metric%(Week N )(%) - Percentage suffix in metric name
            # e.g., "Daily SoRoos%(Week 49 )(%)"
            (
                "pct_suffix_week_pct",
                re.compile(r"^(?P<metric>.+?%)\(Week\s*(?P<week>\d+)\s*\)\(%\)$"),
                "Metric%(Week N)(%)"
            ),
            
            # Pattern 8: Metric%(T12M )(%) - Percentage suffix with T12M
            # e.g., "Daily SoRoos%(T12M )(%)"
            (
                "pct_suffix_t12m_pct",
                re.compile(r"^(?P<metric>.+?%)\(T12M\s*\)\(%\)$"),
                "Metric%(T12M)(%)"
            ),
            
            # Pattern 9: Metric%(MM/DD/YYYY-MM/DD/YYYY)(%) - Pct suffix with date range
            # e.g., "Daily SoRoos%(11/02/2025-11/09/2025)(%)"
            (
                "pct_suffix_daterange_pct",
                re.compile(r"^(?P<metric>.+?%)\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)\(%\)$"),
                "Metric%(MM/DD/YYYY-MM/DD/YYYY)(%)"
            ),

            # Pattern 9a: Metric(%)(Month)(%) - Pct metric with month
            (
                "pct_metric_month_pct",
                re.compile(rf"^(?P<metric>.+?\(%\))\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric(%)(Month)(%)"
            ),

            # Pattern 9b: Metric(%)(Quarter)(%) - Pct metric with quarter
            (
                "pct_metric_quarter_pct",
                re.compile(rf"^(?P<metric>.+?\(%\))\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric(%)(Quarter)(%)"
            ),

            # Pattern 9c: Metric%(Month)(%) - Pct suffix with month
            (
                "pct_suffix_month_pct",
                re.compile(rf"^(?P<metric>.+?%)\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric%(Month)(%)"
            ),

            # Pattern 9d: Metric%(Quarter)(%) - Pct suffix with quarter
            (
                "pct_suffix_quarter_pct",
                re.compile(rf"^(?P<metric>.+?%)\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric%(Quarter)(%)"
            ),
            
            # Pattern 10: Metric (Units)(Week N ) - Units indicator with week
            # e.g., "Net Receipts (Units)(Week 49 )"
            (
                "units_week",
                re.compile(r"^(?P<metric>.+?)\s*\(Units\)\(Week\s*(?P<week>\d+)\s*\)$"),
                "Metric (Units)(Week N)"
            ),
            
            # Pattern 11: Metric (Units)(T12M ) - Units indicator with T12M
            # e.g., "Net Receipts (Units)(T12M )"
            (
                "units_t12m",
                re.compile(r"^(?P<metric>.+?)\s*\(Units\)\(T12M\s*\)$"),
                "Metric (Units)(T12M)"
            ),
            
            # Pattern 12: Metric (Units)(MM/DD/YYYY-MM/DD/YYYY) - Units with date range
            # e.g., "Net Receipts (Units)(11/02/2025-11/09/2025)"
            (
                "units_daterange",
                re.compile(r"^(?P<metric>.+?)\s*\(Units\)\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)$"),
                "Metric (Units)(MM/DD/YYYY-MM/DD/YYYY)"
            ),

            # Pattern 12a: Metric (Units)(Month) - Units with month
            (
                "units_month",
                re.compile(rf"^(?P<metric>.+?)\s*\(Units\)\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric (Units)(Month)"
            ),

            # Pattern 12b: Metric (Units)(Quarter) - Units with quarter
            (
                "units_quarter",
                re.compile(rf"^(?P<metric>.+?)\s*\(Units\)\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric (Units)(Quarter)"
            ),
            
            # Pattern 13: Metric(Week N )(%) - Simple percentage with week
            # e.g., "CM(Week 49 )(%)", "Net PPM(Week 49 )(%)"
            (
                "week_pct",
                re.compile(r"^(?P<metric>.+?)\(Week\s*(?P<week>\d+)\s*\)\(%\)$"),
                "Metric(Week N)(%)"
            ),
            
            # Pattern 14: Metric(T12M )(%) - Simple percentage with T12M
            # e.g., "Net PPM(T12M )(%)", "CM(T12M )(%)"
            (
                "t12m_pct",
                re.compile(r"^(?P<metric>.+?)\(T12M\s*\)\(%\)$"),
                "Metric(T12M)(%)"
            ),
            
            # Pattern 15: Metric(MM/DD/YYYY-MM/DD/YYYY)(%) - Simple percentage with date range
            # e.g., "CM(11/02/2025-11/09/2025)(%)", "Net PPM(11/02/2025-11/09/2025)(%)"
            (
                "daterange_pct",
                re.compile(r"^(?P<metric>.+?)\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)\(%\)$"),
                "Metric(MM/DD/YYYY-MM/DD/YYYY)(%)"
            ),

            # Pattern 15a: Metric(Month)(%) - Simple percentage with month
            (
                "month_pct",
                re.compile(rf"^(?P<metric>.+?)\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric(Month)(%)"
            ),

            # Pattern 15b: Metric(Quarter)(%) - Simple percentage with quarter
            (
                "quarter_pct",
                re.compile(rf"^(?P<metric>.+?)\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)\(%\)$"),
                "Metric(Quarter)(%)"
            ),
            
            # Pattern 15a: Metric(%)(Week N ) - Percentage in metric name with week (no trailing %)
            # e.g., "Conversion Rate(%)(Week 49 )"
            (
                "pct_metric_week",
                re.compile(r"^(?P<metric>.+?\(%\))\(Week\s*(?P<week>\d+)\s*\)$"),
                "Metric(%)(Week N)"
            ),
            
            # Pattern 16: Metric(Week N ) - No unit indicator, just week
            # e.g., "GV(Week 49 )", "ASP(Week 49 )", "SELLABLE_ON_HAND_UNITS(Week 49 )"
            (
                "week_only",
                re.compile(r"^(?P<metric>.+?)\(Week\s*(?P<week>\d+)\s*\)$"),
                "Metric(Week N)"
            ),
            
            # Pattern 17: Metric(T12M ) - No unit indicator, just T12M
            # e.g., "GV(T12M )", "ASP(T12M )"
            (
                "t12m_only",
                re.compile(r"^(?P<metric>.+?)\(T12M\s*\)$"),
                "Metric(T12M)"
            ),
            
            # Pattern 18: Metric(MM/DD/YYYY-MM/DD/YYYY) - No unit, just date range
            # e.g., "GV(11/02/2025-11/09/2025)", "ASP(11/02/2025-11/09/2025)"
            (
                "daterange_only",
                re.compile(r"^(?P<metric>.+?)\((?P<start_date>\d{1,2}/\d{1,2}/\d{4})-(?P<end_date>\d{1,2}/\d{1,2}/\d{4})\)$"),
                "Metric(MM/DD/YYYY-MM/DD/YYYY)"
            ),

            # Pattern 18a: Metric(Month) - No unit, just month
            (
                "month_only",
                re.compile(rf"^(?P<metric>.+?)\((?:Month\s*)?{month_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric(Month)"
            ),

            # Pattern 18b: Metric(Quarter) - No unit, just quarter
            (
                "quarter_only",
                re.compile(rf"^(?P<metric>.+?)\({quarter_pattern}\s*(?P<year>\d{{4}})?\s*\)$"),
                "Metric(Quarter)"
            ),
        ]
        
        return patterns
    
    def _normalize_metric_name(self, metric: str) -> str:
        """Normalize metric name by stripping extra whitespace."""
        return " ".join(metric.strip().split())

    def _normalize_month(self, token: str | None) -> tuple[int | None, str]:
        if not token:
            return None, ""
        token_clean = token.strip().lower()
        if token_clean.isdigit():
            month_num = int(token_clean)
        else:
            month_num = _MONTH_LOOKUP.get(token_clean)
        if not month_num:
            return None, token.strip()
        return month_num, _MONTH_LABELS.get(month_num, token.strip())
    
    def _calculate_confidence(
        self,
        metric_name: str,
        matched_pattern: str,
        is_static: bool = False
    ) -> tuple[str, str]:
        """
        Calculate confidence level and flag status.
        
        Args:
            metric_name: The parsed metric name.
            matched_pattern: Which regex pattern matched.
            is_static: Whether this is a static column (ASIN, Brand Name, etc.)
        
        Returns:
            Tuple of (confidence, flag_status).
        """
        if is_static:
            return "HIGH", "OK"
        
        # Check if metric is in known metrics
        if metric_name in self.known_metrics:
            return "HIGH", "OK"
        
        # Check for partial matches (metric name contains known metric)
        for known in self.known_metrics:
            if known in metric_name or metric_name in known:
                return "MEDIUM", "WARNING"
        
        # Pattern matched but metric unknown
        if matched_pattern:
            return "LOW", "WARNING"
        
        # No pattern matched
        return "ERROR", "ERROR"
    
    def _get_wbr_column(self, metric_name: str, is_static: bool = False) -> str:
        """
        Get the WBR destination column name.
        
        Args:
            metric_name: The parsed metric name.
            is_static: Whether this is a static column.
        
        Returns:
            WBR column name or empty string if not found.
        """
        if is_static:
            return STATIC_COLUMNS.get(metric_name, metric_name)
        
        # Direct match
        if metric_name in self.known_metrics:
            return self.known_metrics[metric_name]
        
        # Try without trailing spaces
        clean_metric = metric_name.strip()
        if clean_metric in self.known_metrics:
            return self.known_metrics[clean_metric]
        
        # Return normalized version as fallback
        return clean_metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "Pct")
    
    def parse(self, header: str) -> ParseResult:
        """
        Parse a single header string.
        
        Args:
            header: The raw header string to parse.
        
        Returns:
            ParseResult with extracted components and confidence.
        """
        result = ParseResult(raw_header=header)
        
        # Check for static columns first
        if header in STATIC_COLUMNS:
            result.metric_name = header
            result.confidence = "HIGH"
            result.flag_status = "OK"
            result.matched_pattern = "static_column"
            result.mapped_wbr_column = STATIC_COLUMNS[header]
            result.period_type = "static"
            return result
        
        # Try each pattern in priority order
        for pattern_name, pattern, description in self._patterns:
            match = pattern.match(header)
            if match:
                groups = match.groupdict()
                metric = self._normalize_metric_name(groups.get("metric", ""))
                week = groups.get("week")
                start_date = groups.get("start_date")
                end_date = groups.get("end_date")
                month_token = groups.get("month")
                quarter = groups.get("quarter")
                year = groups.get("year")
                
                result.metric_name = metric
                result.matched_pattern = pattern_name
                
                # Determine time period
                if week:
                    result.week_number = int(week)
                    result.time_period = f"Week {week}"
                    result.period_type = "week"
                    result.period_label = result.time_period
                elif start_date and end_date:
                    result.time_period = f"{start_date}-{end_date}"
                    result.period_type = "daterange"
                    result.period_label = result.time_period
                    # Try to extract week number from date range if possible
                    # For now, leave week_number as None for date ranges
                elif month_token:
                    _, month_label = self._normalize_month(month_token)
                    label = f"{month_label} {year}".strip() if year else month_label
                    result.time_period = label
                    result.period_type = "month"
                    result.period_label = label
                elif quarter:
                    label = f"Q{quarter} {year}".strip() if year else f"Q{quarter}"
                    result.time_period = label
                    result.period_type = "quarter"
                    result.period_label = label
                elif "t12m" in pattern_name:
                    result.time_period = "T12M"
                    result.period_type = "t12m"
                    result.period_label = "T12M"
                
                # Determine currency/unit
                if "currency" in pattern_name:
                    result.currency_unit = "$"
                elif "units" in pattern_name:
                    result.currency_unit = "units"
                elif "pct" in pattern_name:
                    result.currency_unit = "%"
                    result.is_percentage = True
                
                # Calculate confidence and get WBR column
                result.confidence, result.flag_status = self._calculate_confidence(
                    metric, pattern_name
                )
                result.mapped_wbr_column = self._get_wbr_column(metric)
                
                return result
        
        # No pattern matched - this is an error
        result.confidence = "ERROR"
        result.flag_status = "ERROR"
        result.matched_pattern = "NO_MATCH"
        result.metric_name = header  # Store original as metric name for debugging
        
        return result
    
    def parse_all(self, headers: Sequence[str]) -> list[ParseResult]:
        """
        Parse multiple headers.
        
        Args:
            headers: Sequence of header strings to parse.
        
        Returns:
            List of ParseResult objects.
        """
        return [self.parse(header) for header in headers]
    
    def generate_validation_report(self, results: list[ParseResult]) -> pd.DataFrame:
        """
        Generate a DataFrame suitable for the Validation Tab (Amendment A).
        
        Args:
            results: List of ParseResult objects from parsing.
        
        Returns:
            DataFrame with validation report columns.
        """
        data = []
        for r in results:
            data.append({
                "Raw_Header_Found": r.raw_header,
                "Regex_Pattern_Matched": r.matched_pattern,
                "Parsed_Week": r.time_period,
                "Parsed_Metric": r.metric_name,
                "Parsed_Currency_Unit": r.currency_unit,
                "Mapped_To_WBR_Column": r.mapped_wbr_column,
                "Confidence_Score": r.confidence,
                "Flag_Status": r.flag_status,
            })
        
        return pd.DataFrame(data)
    
    def get_parse_statistics(self, results: list[ParseResult]) -> dict:
        """
        Calculate parsing statistics for validation.
        
        Args:
            results: List of ParseResult objects.
        
        Returns:
            Dictionary with statistics.
        """
        total = len(results)
        if total == 0:
            return {
                "total_headers": 0,
                "successful_parses": 0,
                "success_rate": 0.0,
                "confidence_distribution": {},
                "flag_distribution": {},
                "failed_headers": [],
            }
        
        # Count by confidence
        confidence_counts = {}
        for r in results:
            confidence_counts[r.confidence] = confidence_counts.get(r.confidence, 0) + 1
        
        # Count by flag status
        flag_counts = {}
        for r in results:
            flag_counts[r.flag_status] = flag_counts.get(r.flag_status, 0) + 1
        
        # Successful = not ERROR
        successful = sum(1 for r in results if r.confidence != "ERROR")
        
        # Failed headers (for debugging)
        failed = [r.raw_header for r in results if r.confidence == "ERROR"]
        
        return {
            "total_headers": total,
            "successful_parses": successful,
            "success_rate": (successful / total) * 100,
            "confidence_distribution": confidence_counts,
            "flag_distribution": flag_counts,
            "failed_headers": failed,
        }


def validate_parser_against_file(filepath: str) -> dict:
    """
    Validate the parser against a single file.
    
    Args:
        filepath: Path to CSV/XLSX file.
    
    Returns:
        Dictionary with validation results.
    """
    from pathlib import Path
    from src.file_loader import load_file
    
    path = Path(filepath)
    df = load_file(path, nrows=0)  # Just get headers
    
    parser = HeaderParser()
    results = parser.parse_all(df.columns.tolist())
    stats = parser.get_parse_statistics(results)
    
    return {
        "file": str(path.name),
        "stats": stats,
        "results": results,
    }


if __name__ == "__main__":
    # Quick test with sample headers
    test_headers = [
        "ASIN",
        "Brand Name",
        "Product GMS($)(Week 49 )",
        "Net PPM(Week 49 )(%)",
        "GV(Week 49 )",
        "SoROOS(%)(Week 49 )(%)",
        "Fill Rate - Sourceable(Week 49 )(%)",
        "Net Ordered Units(Week 49 )",
        "Product GMS($)(T12M )",
        "Net PPM(T12M )(%)",
    ]
    
    parser = HeaderParser()
    results = parser.parse_all(test_headers)
    
    print("=" * 80)
    print("Header Parser Test Results")
    print("=" * 80)
    
    for r in results:
        print(f"\nHeader: {r.raw_header}")
        print(f"  Metric: {r.metric_name}")
        print(f"  Time Period: {r.time_period}")
        print(f"  Currency/Unit: {r.currency_unit}")
        print(f"  Pattern: {r.matched_pattern}")
        print(f"  Confidence: {r.confidence}")
        print(f"  WBR Column: {r.mapped_wbr_column}")
    
    stats = parser.get_parse_statistics(results)
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Total: {stats['total_headers']}")
    print(f"Successful: {stats['successful_parses']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Confidence Distribution: {stats['confidence_distribution']}")
