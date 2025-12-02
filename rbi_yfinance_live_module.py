# rbi_yfinance_live_module.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Auto-detect Plotly theme from main app
def get_plotly_template():
    try:
        return "plotly_white" if st.session_state.theme == "light" else "plotly_dark"
    except:
        return "plotly_white"


# -----------------------------
# HELPERS
# -----------------------------
def fetch(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Close": "Value"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# -----------------------------
# BOND YIELDS
# -----------------------------
def get_bond_yields():
    """Indian Govt Bond yields available on Yahoo Finance"""
    bonds = {
        "10Y G-Sec (India)": "^IN10Y",
        "5Y G-Sec (India)": "^IN5Y",
        "1Y T-Bill India": "^IN1Y",
        "Corporate Bond Index": "^CRPBND"  # not perfect but good proxy
    }
    return bonds


# -----------------------------
# INDIAN MARKET INDEXES
# -----------------------------
def get_indices():
    return {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^NSEFIN"
    }


# -----------------------------
# FX PAIRS
# -----------------------------
def get_fx_pairs():
    return {
        "USD/INR": "USDINR=X",
        "EUR/INR": "EURINR=X",
        "GBP/INR": "GBPINR=X",
        "JPY/INR": "JPYINR=X",
    }


# -----------------------------
# COMMODITIES
# -----------------------------
def get_commodities():
    return {
        "Gold (International)": "GC=F",
        "Crude Oil (Brent)": "BZ=F",
        "Crude Oil (WTI)": "CL=F",
    }


# -----------------------------
# MAIN LIVE MARKET TAB RENDER
# -----------------------------
def render_live_market_tab():

    st.title("📈 Live Market Data (India + Global)")
    st.caption("Bond Yields • NIFTY • FX • Gold • Oil (Auto via Yahoo Finance)")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # SECTION: BONDS
    # -------------------------
    st.subheader("🇮🇳 Government Bond Yields (Live)")
    bonds = get_bond_yields()

    b_choice = st.multiselect(
        "Select bond yields to display",
        list(bonds.keys()),
        default=["10Y G-Sec (India)"]
    )

    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=5)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=6)

    for name in b_choice:
        df = fetch(bonds[name], period=period, interval=interval)
        if df.empty:
            st.warning(f"No data available for {name}")
            continue

        st.markdown(f"### {name}")
        fig = px.line(
            df,
            x="Date",
            y="Value",
            title=name,
            template=get_plotly_template()
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # SECTION: INDIAN INDEX
    # -------------------------
    st.subheader("📊 Indian Equity Indexes (Live)")
    indices = get_indices()

    ind_choice = st.multiselect(
        "Select indexes",
        list(indices.keys()),
        default=["NIFTY 50"]
    )

    for name in ind_choice:
        df = fetch(indices[name], period=period, interval=interval)
        if df.empty:
            st.warning(f"No data for {name}")
            continue

        st.markdown(f"### {name}")
        fig = px.line(
            df, x="Date", y="Value", title=name, template=get_plotly_template()
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # SECTION: FX
    # -------------------------
    st.subheader("💱 Forex Rates (INR)")
    fx = get_fx_pairs()

    fx_choice = st.multiselect(
        "Select FX pairs",
        list(fx.keys()),
        default=["USD/INR"]
    )

    for name in fx_choice:
        df = fetch(fx[name], period=period, interval=interval)
        if df.empty:
            st.warning(f"No FX data for {name}")
            continue

        st.markdown(f"### {name}")
        fig = px.line(
            df,
            x="Date",
            y="Value",
            title=name,
            template=get_plotly_template(),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # -------------------------
    # SECTION: COMMODITIES
    # -------------------------
    st.subheader("🛢 Commodities (Gold, Oil)")

    com = get_commodities()
    com_choice = st.multiselect(
        "Select commodities",
        list(com.keys()),
        default=["Gold (International)", "Crude Oil (Brent)"]
    )

    for name in com_choice:
        df = fetch(com[name], period=period, interval=interval)
        if df.empty:
            st.warning(f"No commodity data for {name}")
            continue

        st.markdown(f"### {name}")
        fig = px.line(
            df,
            x="Date",
            y="Value",
            title=name,
            template=get_plotly_template(),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("Live market data updated successfully!")


# -----------------------------
# OPTIONAL SMALL SIDEBAR PANEL
# -----------------------------
def render_live_sidebar_panel():
    st.markdown("### 📈 Live Snapshot (NIFTY + USDINR)")

    df_nifty = fetch("^NSEI", period="5d", interval="1d")
    df_fx = fetch("USDINR=X", period="5d", interval="1d")

    if not df_nifty.empty:
        st.metric("NIFTY 50 (Last Close)", round(df_nifty["Value"].iloc[-1], 2))

    if not df_fx.empty:
        st.metric("USD/INR", round(df_fx["Value"].iloc[-1], 2))
