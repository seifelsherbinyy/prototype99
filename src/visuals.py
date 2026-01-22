"""
Visualization module for profitability dashboard.

Renders KPIs, trend charts, and filtered data grids from processed analytics data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_dashboard(df: pd.DataFrame, low_profitability_threshold: float = 20.0) -> None:
    """
    Render the main profitability dashboard.

    Displays:
    - KPI metrics (Total Revenue, Average Net PPM, Total Units)
    - Trend chart comparing Net_PPM vs 4-week moving average
    - Data grid filtered to show low profitability items

    Args:
        df: DataFrame from analytics.calculate_metrics() with columns:
            ASIN, Product_Title, Fiscal_Year, Fiscal_Week, Fiscal_Month,
            Ordered_Revenue, Ordered_Units, Net_Sales, CCOGS, Gross_Profit,
            Net_PPM, Net_PPM_4w_MA, Ordered_Units_4w_MA
        low_profitability_threshold: Threshold for Net_PPM below which items
            are considered "low profitability" (default: 20.0)
    """
    if df.empty:
        st.warning("No data available to display. Please run analysis first.")
        return

    # Validate required columns
    required_cols = [
        "Ordered_Revenue",
        "Net_PPM",
        "Ordered_Units",
        "Fiscal_Year",
        "Fiscal_Week",
        "ASIN",
        "Product_Title",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.dataframe(df.head())
        return

    # --- KPI Row ---
    st.subheader("Key Performance Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_revenue = df["Ordered_Revenue"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")

    with col2:
        avg_net_ppm = df["Net_PPM"].mean()
        st.metric("Average Net PPM", f"{avg_net_ppm:.2f}%")

    with col3:
        total_units = int(df["Ordered_Units"].sum())
        st.metric("Total Units", f"{total_units:,}")

    st.divider()

    # --- Trend Chart ---
    st.subheader("Net PPM Trend: Weekly vs 4-Week Moving Average")

    # Prepare data for chart
    chart_df = df.copy()

    # Create period label for X-axis
    chart_df["Period"] = (
        chart_df["Fiscal_Year"].astype(str) + "-W" + chart_df["Fiscal_Week"].astype(str).str.zfill(2)
    )

    # Aggregate by period (average across ASINs for overall trend)
    trend_data = (
        chart_df.groupby(["Fiscal_Year", "Fiscal_Week", "Period"])
        .agg(
            {
                "Net_PPM": "mean",
                "Net_PPM_4w_MA": "mean",
            }
        )
        .reset_index()
        .sort_values(["Fiscal_Year", "Fiscal_Week"])
    )

    if not trend_data.empty:
        # Use line chart
        chart_data = trend_data.set_index("Period")[["Net_PPM", "Net_PPM_4w_MA"]]
        st.line_chart(chart_data)

        # Show legend explanation
        st.caption("Blue line: Weekly Net PPM | Orange line: 4-Week Moving Average")
    else:
        st.info("Insufficient data for trend chart. Need multiple periods.")

    st.divider()

    # --- Low Profitability Data Grid ---
    st.subheader(f"Low Profitability Items (Net PPM < {low_profitability_threshold}%)")

    low_profit_df = df[df["Net_PPM"] < low_profitability_threshold].copy()

    if low_profit_df.empty:
        st.success(f"No items found below {low_profitability_threshold}% Net PPM threshold.")
    else:
        # Select and order columns for display
        display_cols = [
            "ASIN",
            "Product_Title",
            "Fiscal_Year",
            "Fiscal_Week",
            "Net_PPM",
            "Ordered_Revenue",
            "Ordered_Units",
            "Gross_Profit",
        ]

        # Only include columns that exist
        available_cols = [col for col in display_cols if col in low_profit_df.columns]
        display_df = low_profit_df[available_cols].copy()

        # Format numeric columns for better readability
        if "Net_PPM" in display_df.columns:
            display_df["Net_PPM"] = display_df["Net_PPM"].round(2)
        if "Ordered_Revenue" in display_df.columns:
            display_df["Ordered_Revenue"] = display_df["Ordered_Revenue"].round(2)
        if "Gross_Profit" in display_df.columns:
            display_df["Gross_Profit"] = display_df["Gross_Profit"].round(2)

        # Sort by Net_PPM (lowest first)
        display_df = display_df.sort_values("Net_PPM", ascending=True)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Showing {len(display_df)} items with Net PPM below {low_profitability_threshold}%")
