"""
Insight Generator Module - Priority-Scored Commentary Engine (Amendment C)

Generates financially-weighted insights and action items:
- Calculate impact scores based on financial weights
- Generate dynamic commentary with actual values (NOT static templates)
- Prioritize issues (top 3 only to avoid template fatigue)
- Generate actionable directives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config import IMPACT_WEIGHTS, THRESHOLDS

if TYPE_CHECKING:
    pass


@dataclass
class IssueScore:
    """Represents a scored performance issue."""
    
    metric: str
    current_value: float
    previous_value: float | None = None
    percentage_change: float | None = None
    impact_score: float = 0.0
    priority_rank: int | None = None
    rag_status: str = "N/A"
    commentary: str = ""
    next_action: str = ""


# Commentary templates with placeholders for dynamic values
COMMENTARY_TEMPLATES = {
    "revenue_drop_severe": (
        "Revenue dropped {pct_change:.1f}% WoW (from ${prev:,.0f} to ${curr:,.0f}), "
        "a significant decline requiring immediate attention."
    ),
    "revenue_drop_moderate": (
        "Revenue decreased {pct_change:.1f}% WoW to ${curr:,.0f}. "
        "Monitor closely and review promotional activity."
    ),
    "revenue_growth": (
        "Revenue grew {pct_change:.1f}% WoW to ${curr:,.0f}. "
        "Positive momentum - maintain current strategy."
    ),
    "units_drop_severe": (
        "Units sold dropped {pct_change:.1f}% WoW (from {prev:,.0f} to {curr:,.0f}), "
        "indicating demand or availability issues."
    ),
    "units_drop_moderate": (
        "Units decreased {pct_change:.1f}% WoW to {curr:,.0f}. "
        "Check inventory levels and pricing competitiveness."
    ),
    "conversion_drop_severe": (
        "Conversion dropped {pct_change:.1f}% WoW (from {prev:.1f}% to {curr:.1f}%), "
        "potentially impacting ~${estimated_impact:,.0f} in weekly revenue."
    ),
    "conversion_drop_moderate": (
        "Conversion decreased {pct_change:.1f}% WoW to {curr:.1f}%. "
        "Review listing quality and competitive pricing."
    ),
    "traffic_drop_severe": (
        "Traffic (Glance Views) dropped {pct_change:.1f}% WoW (from {prev:,.0f} to {curr:,.0f}), "
        "suggesting reduced visibility or search ranking."
    ),
    "traffic_drop_moderate": (
        "Traffic decreased {pct_change:.1f}% WoW to {curr:,.0f}. "
        "Check advertising spend and organic search performance."
    ),
    "traffic_spike": (
        "Traffic increased {pct_change:.1f}% WoW to {curr:,.0f}. "
        "Capitalize on increased visibility by ensuring strong conversion."
    ),
    "profitability_drop": (
        "Net PPM dropped to {curr:.1f}% ({rag_status}), "
        "indicating margin pressure. Review pricing and cost structure."
    ),
    "profitability_critical": (
        "Net PPM at critical level: {curr:.1f}% (RED). "
        "Urgent review of pricing strategy and cost optimization needed."
    ),
    "stock_risk": (
        "Stock out risk: SoROOS at {curr:.1f}% ({rag_status}). "
        "Review inventory levels and inbound shipments."
    ),
    "fill_rate_low": (
        "Fill rate at {curr:.1f}% ({rag_status}), below target. "
        "Check vendor confirmation and fulfillment processes."
    ),
    "stable_performance": (
        "Performance stable. No critical issues requiring immediate action."
    ),
}

# Action directives for each issue type
ACTION_DIRECTIVES = {
    "revenue_drop": [
        "1. Analyze price elasticity and competitive positioning",
        "2. Review promotional calendar and deals pipeline",
        "3. Check Buy Box ownership and pricing strategy",
    ],
    "units_drop": [
        "1. Verify inventory availability across fulfillment centers",
        "2. Review pricing competitiveness vs. competitors",
        "3. Check for listing suppression or content issues",
    ],
    "conversion_drop": [
        "1. Audit product listing content, images, and A+ content",
        "2. Analyze competitor pricing and review sentiment",
        "3. Review customer Q&A and address common concerns",
    ],
    "traffic_drop": [
        "1. Review PPC campaign performance and budget allocation",
        "2. Check organic search ranking for key terms",
        "3. Analyze recent algorithm or category changes",
    ],
    "profitability_issue": [
        "1. Review cost structure and supplier negotiations",
        "2. Analyze promotional ROI and reduce unprofitable deals",
        "3. Consider price optimization to improve margins",
    ],
    "stock_issue": [
        "1. Review inventory levels and reorder thresholds",
        "2. Check inbound shipment status and appointments",
        "3. Coordinate with vendor on fulfillment capacity",
    ],
    "stable": [
        "1. Document current successful tactics",
        "2. Monitor for emerging trends or opportunities",
        "3. Consider expansion or scaling strategies",
    ],
}


def calculate_impact_score(
    metric: str,
    percentage_change: float,
    impact_weights: dict[str, int] | None = None
) -> float:
    """
    Calculate financial impact score for a metric change.
    
    Formula: Impact Score = |Percentage_Change| x Metric_Weight x Direction_Multiplier
    Direction_Multiplier: -1 for declines (prioritize problems), +0.5 for improvements
    
    Args:
        metric: Metric name.
        percentage_change: WoW percentage change.
        impact_weights: Optional custom weights. Uses IMPACT_WEIGHTS if None.
    
    Returns:
        Impact score (higher = more important).
    """
    if impact_weights is None:
        impact_weights = IMPACT_WEIGHTS
    
    # Get weight for metric (default to 1 if not found)
    weight = impact_weights.get(metric, 1)
    
    pct_change = percentage_change
    if abs(pct_change) <= 1:
        pct_change *= 100

    # Direction multiplier: problems (declines) are prioritized
    if pct_change < 0:
        direction_multiplier = 1.0  # Full weight for declines
    else:
        direction_multiplier = 0.5  # Half weight for improvements
    
    return abs(pct_change) * weight * direction_multiplier


def generate_dynamic_commentary(
    issue: IssueScore,
    estimated_revenue_impact: float | None = None
) -> str:
    """
    Generate dynamic commentary with actual values.
    
    Args:
        issue: IssueScore object with metric details.
        estimated_revenue_impact: Optional estimated revenue impact.
    
    Returns:
        Dynamic commentary string with actual values.
    """
    metric = issue.metric
    curr = issue.current_value
    prev = issue.previous_value
    pct_change = issue.percentage_change or 0
    pct_change_display = abs(pct_change) * 100 if abs(pct_change) <= 1 else abs(pct_change)
    curr_display = curr
    prev_display = prev if prev is not None else curr
    if any(token in metric for token in ["PPM", "SoROOS", "Rate", "Conversion"]):
        curr_display = curr * 100 if curr is not None and abs(curr) <= 1 else curr
        if prev_display is not None and abs(prev_display) <= 1:
            prev_display = prev_display * 100
    rag_status = issue.rag_status
    
    # Revenue metrics
    if "Revenue" in metric:
        if pct_change < -0.15:
            template = COMMENTARY_TEMPLATES["revenue_drop_severe"]
        elif pct_change < -0.05:
            template = COMMENTARY_TEMPLATES["revenue_drop_moderate"]
        elif pct_change > 0.05:
            template = COMMENTARY_TEMPLATES["revenue_growth"]
        else:
            return f"Revenue at ${curr:,.0f}, relatively stable WoW."
    
    # Units metrics
    elif "Units" in metric:
        if pct_change < -0.15:
            template = COMMENTARY_TEMPLATES["units_drop_severe"]
        elif pct_change < -0.05:
            template = COMMENTARY_TEMPLATES["units_drop_moderate"]
        else:
            return f"Units at {curr:,.0f}, within normal range."
    
    # Conversion / traffic metrics
    elif "Glance" in metric or "Views" in metric:
        if pct_change < -0.20:
            template = COMMENTARY_TEMPLATES["traffic_drop_severe"]
        elif pct_change < -0.10:
            template = COMMENTARY_TEMPLATES["traffic_drop_moderate"]
        elif pct_change > 0.20:
            template = COMMENTARY_TEMPLATES["traffic_spike"]
        else:
            return f"Traffic at {curr:,.0f}, within normal range."
    
    # Profitability metrics
    elif "PPM" in metric:
        if rag_status == "RED":
            template = COMMENTARY_TEMPLATES["profitability_critical"]
        elif rag_status == "AMBER" or curr < 0.15:
            template = COMMENTARY_TEMPLATES["profitability_drop"]
        else:
            return f"Net PPM at {curr_display:.1f}% (GREEN), healthy profitability."
    
    # Stock metrics
    elif "SoROOS" in metric:
        if rag_status in ["RED", "AMBER"]:
            template = COMMENTARY_TEMPLATES["stock_risk"]
        else:
            return f"Stock availability good at {curr_display:.1f}% SoROOS."
    
    # Fill rate metrics
    elif "Fill" in metric:
        if rag_status in ["RED", "AMBER"]:
            template = COMMENTARY_TEMPLATES["fill_rate_low"]
        else:
            return f"Fill rate at {curr_display:.1f}%, meeting targets."
    
    else:
        # Generic commentary
        if pct_change < -0.10:
            return f"{metric} decreased {pct_change_display:.1f}% WoW to {curr:,.2f}. Monitor closely."
        elif pct_change > 0.10:
            return f"{metric} increased {pct_change_display:.1f}% WoW to {curr:,.2f}. Positive trend."
        else:
            return f"{metric} at {curr:,.2f}, stable WoW."
    
    # Format the template
    try:
        return template.format(
            curr=curr_display,
            prev=prev_display if prev_display is not None else curr_display,
            pct_change=pct_change_display,
            rag_status=rag_status,
            estimated_impact=estimated_revenue_impact or 0,
        )
    except (KeyError, ValueError):
        return f"{metric}: {curr:,.2f} ({pct_change:+.1f}% WoW)"


def get_directive_for_issue(issue: IssueScore) -> str:
    """
    Get actionable directive for an issue.
    
    Args:
        issue: IssueScore object.
    
    Returns:
        Directive string with action items.
    """
    metric = issue.metric
    
    if "Revenue" in metric:
        directives = ACTION_DIRECTIVES["revenue_drop"]
    elif "Units" in metric:
        directives = ACTION_DIRECTIVES["units_drop"]
    elif "Glance" in metric or "Views" in metric:
        directives = ACTION_DIRECTIVES["traffic_drop"]
    elif "PPM" in metric or "Margin" in metric:
        directives = ACTION_DIRECTIVES["profitability_issue"]
    elif "SoROOS" in metric or "Fill" in metric or "Stock" in metric:
        directives = ACTION_DIRECTIVES["stock_issue"]
    else:
        directives = ACTION_DIRECTIVES["stable"]
    
    return " ".join(directives)


def prioritize_issues(
    issues: list[IssueScore],
    top_n: int = 3
) -> list[IssueScore]:
    """
    Rank issues by impact score and return top N.
    
    Args:
        issues: List of IssueScore objects.
        top_n: Number of top issues to return.
    
    Returns:
        List of top N issues with priority ranks assigned.
    """
    # Sort by impact score (descending)
    sorted_issues = sorted(issues, key=lambda x: x.impact_score, reverse=True)
    
    # Assign priority ranks to top N
    top_issues = []
    for i, issue in enumerate(sorted_issues[:top_n]):
        issue.priority_rank = i + 1
        top_issues.append(issue)
    
    return top_issues


def analyze_row(
    row_data: dict[str, Any],
    wow_data: dict[str, float] | None = None
) -> list[IssueScore]:
    """
    Analyze a single row and generate issues.
    
    Args:
        row_data: Dictionary with metric values.
        wow_data: Optional dictionary with WoW percentage changes.
    
    Returns:
        List of IssueScore objects for issues found.
    """
    issues = []
    
    # Metrics to analyze
    metrics_to_check = [
        ("Ordered_Revenue", "Revenue_WoW"),
        ("Ordered_Units", "Units_WoW"),
        ("Glance_Views", "GlanceViews_WoW"),
        ("Net_PPM", "NetPPM_WoW"),
        ("Average_Selling_Price", "ASP_WoW"),
    ]
    
    for metric, wow_col in metrics_to_check:
        if metric not in row_data:
            continue
        
        current_value = row_data.get(metric)
        if pd.isna(current_value):
            continue
        
        # Get WoW change
        pct_change = None
        if wow_data and wow_col in wow_data:
            pct_change = wow_data.get(wow_col)
        elif wow_col in row_data:
            pct_change = row_data.get(wow_col)
        
        # Get RAG status if available
        rag_col = f"{metric}_RAG"
        rag_status = row_data.get(rag_col, "N/A")
        
        # Calculate impact score
        impact_score = 0
        if pct_change is not None and not pd.isna(pct_change):
            impact_score = calculate_impact_score(metric, pct_change)
        
        # Only create issue if significant
        if impact_score > 10 or rag_status in ["RED", "AMBER"]:
            issue = IssueScore(
                metric=metric,
                current_value=float(current_value),
                percentage_change=float(pct_change) if pct_change is not None and not pd.isna(pct_change) else None,
                impact_score=impact_score,
                rag_status=rag_status,
            )
            
            # Generate commentary
            issue.commentary = generate_dynamic_commentary(issue)
            issue.next_action = get_directive_for_issue(issue)
            
            issues.append(issue)
    
    return issues


def generate_insights_for_dataframe(
    df: pd.DataFrame,
    top_n: int = 3,
    include_enhanced: bool = True
) -> pd.DataFrame:
    """
    Generate insights for all rows in a DataFrame.
    
    Adds columns:
    - Priority_Rank (1, 2, 3 for top issues, blank for others)
    - Impact_Score
    - Automated_Commentary
    - Next_Actions
    - Enhanced_Insights (if include_enhanced=True)
    
    Args:
        df: Analyzed DataFrame with RAG columns.
        top_n: Number of top issues per row.
        include_enhanced: Whether to include enhanced insights (statistical, comparative).
    
    Returns:
        DataFrame with insight columns added.
    """
    result = df.copy()
    
    # Initialize columns
    result["Priority_Rank"] = ""
    result["Impact_Score"] = 0.0
    result["Automated_Commentary"] = ""
    result["Next_Actions"] = ""
    
    if include_enhanced:
        result["Enhanced_Insights"] = ""
    
    for idx, row in result.iterrows():
        row_data = row.to_dict()
        
        # Analyze row
        issues = analyze_row(row_data)
        
        if not issues:
            result.at[idx, "Automated_Commentary"] = COMMENTARY_TEMPLATES["stable_performance"]
            result.at[idx, "Next_Actions"] = " ".join(ACTION_DIRECTIVES["stable"])
            if include_enhanced:
                result.at[idx, "Enhanced_Insights"] = _generate_enhanced_insight_text(row_data, df)
            continue
        
        # Prioritize issues
        top_issues = prioritize_issues(issues, top_n)
        
        if top_issues:
            # Take the top issue for the row
            top_issue = top_issues[0]
            result.at[idx, "Priority_Rank"] = f"#{top_issue.priority_rank}"
            result.at[idx, "Impact_Score"] = top_issue.impact_score
            result.at[idx, "Automated_Commentary"] = top_issue.commentary
            result.at[idx, "Next_Actions"] = top_issue.next_action
            
            if include_enhanced:
                result.at[idx, "Enhanced_Insights"] = _generate_enhanced_insight_text(row_data, df)
    
    return result


def _generate_enhanced_insight_text(row_data: dict[str, Any], df: pd.DataFrame | None = None) -> str:
    """
    Generate enhanced insight text using statistical and comparative analysis.
    
    Args:
        row_data: Row data dictionary.
        df: Full DataFrame for comparisons.
    
    Returns:
        Enhanced insight text string.
    """
    insights = []
    
    # Trend insights
    if "Trend_Direction" in row_data:
        trend = row_data["Trend_Direction"]
        if trend != "N/A" and trend != "→ Stable":
            insights.append(f"Trend: {trend}")
    
    # Comparative insights
    if "Ordered_Revenue_vs_Vendor" in row_data and pd.notna(row_data["Ordered_Revenue_vs_Vendor"]):
        vs_vendor = row_data["Ordered_Revenue_vs_Vendor"]
        if abs(vs_vendor) > 10:  # >10% difference
            direction = "above" if vs_vendor > 0 else "below"
            insights.append(f"Revenue {abs(vs_vendor):.0f}% {direction} vendor average")
    
    if "Ordered_Revenue_vs_Portfolio" in row_data and pd.notna(row_data["Ordered_Revenue_vs_Portfolio"]):
        vs_portfolio = row_data["Ordered_Revenue_vs_Portfolio"]
        if abs(vs_portfolio) > 20:  # >20% difference
            direction = "above" if vs_portfolio > 0 else "below"
            insights.append(f"Revenue {abs(vs_portfolio):.0f}% {direction} portfolio average")
    
    # Top/Bottom performer flags
    if "Ordered_Revenue_Top_Performer" in row_data and row_data.get("Ordered_Revenue_Top_Performer"):
        insights.append("Top 10% revenue performer")
    
    if "Ordered_Revenue_Bottom_Performer" in row_data and row_data.get("Ordered_Revenue_Bottom_Performer"):
        insights.append("Bottom 10% revenue performer")
    
    return "; ".join(insights) if insights else "No additional insights"


def get_top_issues_summary(
    df: pd.DataFrame,
    top_n: int = 3
) -> list[dict[str, Any]]:
    """
    Get summary of top issues across all ASINs.
    
    Args:
        df: DataFrame with insights.
        top_n: Number of top issues to return.
    
    Returns:
        List of top issues with ASIN and details.
    """
    all_issues = []
    
    for idx, row in df.iterrows():
        row_data = row.to_dict()
        issues = analyze_row(row_data)
        
        for issue in issues:
            all_issues.append({
                "ASIN": row_data.get("ASIN", "Unknown"),
                "metric": issue.metric,
                "current_value": issue.current_value,
                "percentage_change": issue.percentage_change,
                "impact_score": issue.impact_score,
                "rag_status": issue.rag_status,
                "commentary": issue.commentary,
                "next_action": issue.next_action,
            })
    
    # Sort by impact score and return top N
    all_issues.sort(key=lambda x: x["impact_score"], reverse=True)
    return all_issues[:top_n]


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Insight Generator Module Test")
    print("=" * 60)
    
    # Test impact score calculation
    print("\n1. Testing impact score calculation...")
    test_cases = [
        ("Ordered_Revenue", -0.20, "Severe revenue decline"),
        ("Ordered_Revenue", 0.10, "Revenue improvement"),
        ("Glance_Views", -0.30, "Severe traffic drop"),
        ("Net_PPM", -0.05, "Profitability decline"),
    ]
    
    for metric, pct_change, desc in test_cases:
        score = calculate_impact_score(metric, pct_change)
        display = pct_change * 100 if abs(pct_change) <= 1 else pct_change
        print(f"   {desc}: {metric} {display:+.1f}% -> Score: {score:.1f}")
    
    # Test with real data
    print("\n2. Testing with real data...")
    from src.data_loader import detect_latest_file, load_raw_data, load_reference_data
    from src.header_parser import HeaderParser
    from src.data_transformer import transform_raw_to_wbr
    from src.analyzer import analyze_performance
    from src.config import DROPZONE_PATH
    
    latest = detect_latest_file(DROPZONE_PATH)
    if latest:
        raw_df, _ = load_raw_data(latest)
        parser = HeaderParser()
        parse_results = parser.parse_all(raw_df.columns.tolist())
        ref_data = load_reference_data()
        
        wbr_df = transform_raw_to_wbr(
            raw_df,
            parse_results,
            vendor_map=ref_data["vendor_map"],
            unpivot=True
        )
        
        # Analyze
        analyzed_df, summary = analyze_performance(wbr_df)
        
        # Generate insights
        insights_df = generate_insights_for_dataframe(analyzed_df)
        
        print(f"   Processed {len(insights_df)} rows")
        print(f"   Insight columns added: Priority_Rank, Impact_Score, Automated_Commentary, Next_Actions")
        
        # Show sample insights
        sample = insights_df[insights_df["Priority_Rank"] != ""].head(3)
        print("\n   Sample insights:")
        for _, row in sample.iterrows():
            print(f"     ASIN: {row['ASIN']}")
            print(f"     Priority: {row['Priority_Rank']}, Score: {row['Impact_Score']:.1f}")
            print(f"     Commentary: {row['Automated_Commentary'][:80]}...")
            print()
        
        # Get top issues summary
        print("   Top 3 issues overall:")
        top_issues = get_top_issues_summary(analyzed_df, top_n=3)
        for i, issue in enumerate(top_issues):
            print(f"     #{i+1}: {issue['ASIN']} - {issue['metric']}")
            print(f"         Score: {issue['impact_score']:.1f}, RAG: {issue['rag_status']}")
    
    print("\n" + "=" * 60)
    print("Insight Generator Module Test Complete")
    print("=" * 60)
