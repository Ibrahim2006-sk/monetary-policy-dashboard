# Monitory_of_RBI_fixed_full.py
"""
Full RBI Monetary Policy Dashboard — Fixed & Themed
Paste Segments 1, 2, 3 into a single file in order.
"""

import os
import re
import json
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# Forecasting
from statsmodels.tsa.arima.model import ARIMA

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# yfinance live module - ensure rbi_yfinance_live_module.py is in same folder
try:
    from rbi_yfinance_live_module import render_live_market_tab, render_live_inside_fx_tab, render_live_sidebar_panel
except Exception:
    # If module missing, define stubs so file still runs (user will be warned)
    def render_live_market_tab(*args, **kwargs):
        st.warning("rbi_yfinance_live_module not found. Place rbi_yfinance_live_module.py in the same folder.")

    def render_live_inside_fx_tab(*args, **kwargs):
        st.warning("rbi_yfinance_live_module not found. Place rbi_yfinance_live_module.py in the same folder.")

    def render_live_sidebar_panel(*args, **kwargs):
        st.warning("rbi_yfinance_live_module not found. Place rbi_yfinance_live_module.py in the same folder.")

# --------------------
# CONFIG
# --------------------
FRED_API_KEY = "f1d8b90ecd833ea9092dca3882935787"

st.set_page_config(
    page_title="Monetary Policy Dashboard (RBI Theme)",
    layout="wide"
)

# --------------------------
# THEME SWITCHER (LIGHT / DARK)
# --------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def switch_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# Sidebar toggle (keeps layout stable)
with st.sidebar:
    st.write("")  # spacer
    if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"):
        switch_theme()

# CSS for RBI look — dynamic based on theme
LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, .stApp {
    background-color: #ffffff !important;
    font-family: 'Inter', sans-serif;
    color: #003366 !important;
}
h1, h2, h3, h4 {
    font-family: 'Merriweather', serif !important;
    color: #003366 !important;
}
.header-container {
    width: 100%;
    padding: 18px 20px;
    background-color: #003366;
    border-bottom: 4px solid #0066CC;
}
.header-title {
    color: white !important;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    font-family: 'Merriweather', serif;
}
.rbi-card {
    padding: 16px;
    border: 2px solid #00336633;
    border-radius: 10px;
    background-color: #F2F7FF;
    box-shadow: 0px 2px 6px #00336622;
}
.rbi-card h3 { color: #003366 !important; margin-bottom: 6px; }
.rbi-card p { color: #003366; font-size: 18px; }
.stTabs [role="tablist"] button {
    font-size: 15px; font-weight: 600; background-color: #E8EEF7; color: #003366 !important;
    border-radius: 5px; margin-right: 4px; padding: 6px 14px;
}
.stTabs [aria-selected="true"] { background-color: #003366 !important; color: white !important; }
section[data-testid="stSidebar"] { background-color: #F1F4FA !important; border-right: 3px solid #00336633; }
section[data-testid="stSidebar"] h2, h3, h4, label { color: #003366 !important; }
.stButton > button { background-color: #003366 !important; color: white !important; border-radius: 6px; padding: 8px 18px; border: none; }
.stButton > button:hover { background-color: #0055AA !important; color: #ffffff !important; }
.rbi-divider { height: 3px; background-color: #00336633; margin: 25px 0px; border-radius: 2px; }
</style>
"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, .stApp {
    background-color: #0d1117 !important;
    font-family: 'Inter', sans-serif;
    color: #c9d1d9 !important;
}
h1, h2, h3, h4 {
    font-family: 'Merriweather', serif !important;
    color: #58a6ff !important;
}
.header-container {
    width: 100%;
    padding: 18px 20px;
    background-color: #161b22;
    border-bottom: 4px solid #30363d;
}
.header-title {
    color: #58a6ff !important;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    font-family: 'Merriweather', serif;
}
.rbi-card {
    padding: 16px;
    border: 1px solid #30363d;
    border-radius: 10px;
    background-color: #1f242b;
    box-shadow: 0px 2px 6px #00000066;
    color: #c9d1d9 !important;
}
.rbi-card h3 { color: #58a6ff !important; margin-bottom: 6px; }
.rbi-card p { color: #c9d1d9; font-size: 18px; }
.stTabs [role="tablist"] button {
    font-size: 15px; font-weight: 600; background-color: #0f1720; color: #c9d1d9 !important;
    border-radius: 5px; margin-right: 4px; padding: 6px 14px;
}
.stTabs [aria-selected="true"] { background-color: #0b1220 !important; color: #58a6ff !important; border: 1px solid #30363d; }
section[data-testid="stSidebar"] { background-color: #0f1720 !important; border-right: 3px solid #22272b; color: #c9d1d9 !important; }
section[data-testid="stSidebar"] h2, h3, h4, label { color: #c9d1d9 !important; }
.stButton > button { background-color: #0b1220 !important; color: #58a6ff !important; border-radius: 6px; padding: 8px 18px; border: 1px solid #30363d; }
.stButton > button:hover { background-color: #0e2336 !important; color: #ffffff !important; }
.rbi-divider { height: 3px; background-color: #30363d; margin: 25px 0px; border-radius: 2px; }
</style>
"""

# Apply selected CSS
if st.session_state.theme == "light":
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
else:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# Helper for Plotly template
def get_plotly_template():
    return "plotly_white" if st.session_state.theme == "light" else "plotly_dark"

# --------------------
# HEADER (RBI-style)
# --------------------
st.markdown(
    """
<div class='header-container'>
    <div class='header-title'>
        Monetary Policy Dashboard — Reserve Bank of India (Analytical View)
    </div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Interactive dashboard combining RBI-style UI with global macro data and uploads.")

# --------------------
# SIDEBAR CONTROLS
# --------------------
with st.sidebar:
    st.markdown("### Controls")
    start_year = st.number_input(
        "Start year",
        min_value=1960,
        max_value=datetime.now().year,
        value=2000,
    )
    end_year = st.number_input(
        "End year",
        min_value=1960,
        max_value=datetime.now().year,
        value=datetime.now().year,
    )
    use_fred = st.checkbox(
        "Use FRED for US data (CPI, Fed, Oil, Gold)", value=True
    )
    st.markdown("---")
    st.markdown("### Forecast & Index Weights")
    forecast_horizon = st.number_input(
        "Forecast horizon (years/steps)", min_value=1, max_value=60, value=5
    )
    w_infl = st.slider("Inflation weight", 0.0, 1.0, 0.4)
    w_rate = st.slider("Policy rate weight", 0.0, 1.0, 0.25)
    w_liq = st.slider("Liquidity weight", 0.0, 1.0, 0.2)
    w_vol = st.slider("Volatility weight", 0.0, 1.0, 0.15)

st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

# --------------------
# HELPERS
# --------------------
def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

@st.cache_data
def fetch_worldbank_indicator(country_code, indicator, per_page=2000):
    """Fetch World Bank indicator. Returns DataFrame with Date and Value columns."""
    try:
        url = (
            f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
            f"{indicator}?format=json&per_page={per_page}"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, list) or len(j) < 2:
            return pd.DataFrame()
        data = j[1]
        df = pd.DataFrame(
            [
                {"Date": item.get("date"), "Value": item.get("value")}
                for item in data
                if item.get("value") is not None
            ]
        )
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"], format="%Y", errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def fetch_fred_series(series_id, api_key=FRED_API_KEY, start=None, end=None):
    """Fetch a FRED series; returns DataFrame indexed by Date with column 'Value'."""
    try:
        if not api_key:
            return pd.DataFrame()
        url = (
            f"https://api.stlouisfed.org/fred/series/observations?"
            f"series_id={series_id}&api_key={api_key}&file_type=json"
        )
        if start:
            url += f"&observation_start={start}"
        if end:
            url += f"&observation_end={end}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations", [])
        df = pd.DataFrame(
            [
                {
                    "Date": pd.to_datetime(o["date"]),
                    "Value": (float(o["value"]) if o["value"] != "." else np.nan),
                }
                for o in obs
            ]
        )
        df = df.dropna().set_index("Date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def scrape_rbi_repo_rate():
    """Best-effort scraping of RBI pages to find latest repo rate."""
    candidates = [
        "https://www.rbi.org.in/Scripts/BS_ViewMasala.aspx?Id=2009",
        "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
        "https://www.rbi.org.in/Scripts/BS_View.aspx?Id=2009",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            m = re.search(
                r"Repo[ -]rate[^0-9\n\%\.]*([0-9]+\.?[0-9]*)", text, re.IGNORECASE
            )
            if m:
                val = float(m.group(1))
                df = pd.DataFrame({"RepoRate": [val]}, index=[pd.to_datetime("today")])
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data
def scrape_rbi_press_releases(max_items=10):
    """Scrape RBI press releases and filter for monetary policy."""
    url = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}
    out = []
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)
        for a in links:
            title = a.get_text(strip=True)
            href = a["href"]
            if any(
                kw in title.lower()
                for kw in ["monetary policy", "policy statement", "repo rate", "mpc"]
            ):
                full_url = href if href.startswith("http") else "https://www.rbi.org.in" + href
                out.append({"Title": title, "URL": full_url})
            if len(out) >= max_items:
                break
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def fetch_forex_inr(symbols=("USD", "EUR", "GBP", "JPY"), years=5):
    """Fetch INR FX using exchangerate.host timeseries (base=INR)."""
    try:
        end = datetime.now().date()
        start = datetime(end.year - years, 1, 1).date()
        url = (
            f"https://api.exchangerate.host/timeseries?start_date={start}"
            f"&end_date={end}&base=INR&symbols={','.join(symbols)}"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        rates = data.get("rates", {})
        rows = []
        for d, v in rates.items():
            row = {"Date": pd.to_datetime(d)}
            for s in symbols:
                if s in v:
                    row[s] = v[s]
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("Date").set_index("Date")
        return df
    except Exception:
        return pd.DataFrame()

def sample_econ_calendar():
    """Simple editable calendar template."""
    data = [
        {
            "Event": "RBI MPC Meeting",
            "Date": "2025-02-08",
            "Type": "Policy",
            "Importance": "High",
        },
        {
            "Event": "India CPI Release",
            "Date": "2025-01-15",
            "Type": "Inflation",
            "Importance": "High",
        },
        {
            "Event": "India GDP Data",
            "Date": "2025-03-01",
            "Type": "Growth",
            "Importance": "Medium",
        },
        {
            "Event": "US Fed FOMC Meeting",
            "Date": "2025-01-29",
            "Type": "Policy",
            "Importance": "High",
        },
    ]
    return pd.DataFrame(data)

def compute_mpi_msi_and_insights(
    wb_infl_in,
    m3_in,
    rbi_repo,
    liq_monthly,
    repo_monthly,
    w_infl,
    w_rate,
    w_liq,
    w_vol,
):
    """Compute MPI, MSI, and qualitative insights (no Streamlit inside)."""

    # Latest inflation (India, 3-year average)
    infl_latest = (
        safe_num(wb_infl_in["Value"].astype(float).dropna().iloc[-3:].mean())
        if not wb_infl_in.empty
        else np.nan
    )

    # Repo rate latest (uploaded or scraped)
    rate_latest = np.nan
    if not repo_monthly.empty:
        rate_latest = safe_num(repo_monthly.iloc[-1, 0])
    elif not rbi_repo.empty:
        rate_latest = safe_num(rbi_repo.iloc[-1, 0])

    # Liquidity latest (M3_IN from upload or WB)
    if not liq_monthly.empty and "M3_IN" in liq_monthly.columns:
        liq_latest = safe_num(liq_monthly["M3_IN"].dropna().iloc[-1])
    elif not m3_in.empty:
        liq_latest = safe_num(m3_in["Value"].dropna().iloc[-1])
    else:
        liq_latest = np.nan

    # Volatility latest (rolling std of inflation)
    if not wb_infl_in.empty:
        s = wb_infl_in["Value"].astype(float)
        vol_series = s.rolling(3, min_periods=1).std()
        vol_latest = safe_num(vol_series.dropna().iloc[-1])
    else:
        vol_latest = np.nan

    raw_vals = np.array(
        [infl_latest, rate_latest, liq_latest, vol_latest], dtype="float64"
    )
    missing = np.isnan(raw_vals).sum()
    vals = np.nan_to_num(raw_vals, nan=0.0)

    weights = np.array([w_infl, w_rate, w_liq, w_vol], dtype="float64")
    if weights.sum() == 0:
        weights = np.array([0.4, 0.25, 0.2, 0.15])
    weights = weights / weights.sum()

    mpi = float(np.dot(vals, weights))

    # Inflation trend (India)
    infl_trend = "N/A"
    if not wb_infl_in.empty:
        last_vals = wb_infl_in.dropna().tail(5)["Value"].astype(float)
        if len(last_vals) > 1:
            change = last_vals.iloc[-1] - last_vals.iloc[0]
            if change > 0.5:
                infl_trend = "Inflation in India has been rising in the recent period."
            elif change < -0.5:
                infl_trend = "Inflation in India has been easing compared to earlier years."
            else:
                infl_trend = "Inflation in India has been relatively stable recently."

    # Policy stance text
    rate_text = "N/A"
    if not np.isnan(rate_latest):
        if rate_latest >= 6.5:
            rate_text = "Policy stance appears relatively tight with a higher repo rate."
        elif rate_latest <= 4.5:
            rate_text = "Policy stance appears accommodative with a relatively low repo rate."
        else:
            rate_text = "Policy stance appears neutral to moderately tight."

    # Monetary Stress Index (0-100)
    norm_vals = []
    for v in raw_vals:
        if np.isnan(v):
            norm_vals.append(0.0)
        else:
            norm_vals.append(np.tanh(v / 10))  # soft normalization
    msi = float(np.mean(norm_vals) * 50 + 50)  # center around 50

    if msi >= 70:
        stress_text = "Overall monetary stress is elevated — conditions are tight or volatile."
    elif msi <= 40:
        stress_text = "Monetary stress is low — conditions are easy and supportive."
    else:
        stress_text = "Monetary conditions appear balanced with moderate stress."

    return mpi, msi, infl_trend, rate_text, stress_text, missing

# --------------------
# NEW: LIVE RBI DATA HELPERS (WSS Sections)
# --------------------
@st.cache_data
def fetch_rbi_policy_rates_live():
    """Fetch live RBI policy rates: Repo, Reverse Repo, MSF, Bank Rate, SDF."""
    url = "https://www.rbi.org.in/scripts/BS_ViewMonetaryPolicy.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        def find_rate(keyword):
            m = re.search(rf"{keyword}[^0-9]*([0-9]+\.?[0-9]*)", text, re.I)
            return float(m.group(1)) if m else np.nan

        df = pd.DataFrame({
            "Indicator": ["Repo Rate", "Reverse Repo", "MSF", "Bank Rate", "SDF"],
            "Value": [
                find_rate("Repo Rate"),
                find_rate("Reverse Repo"),
                find_rate("Marginal Standing Facility"),
                find_rate("Bank Rate"),
                find_rate("Standing Deposit Facility"),
            ]
        })
        df = df.dropna(subset=["Value"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def fetch_rbi_laf_daily():
    """Fetch DAILY RBI Liquidity (LAF, MSF, SDF, Call Money) from WSS Section 1."""
    url = "https://www.rbi.org.in/Scripts/WSS_Section1.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Table id for LAF daily data (may change if RBI updates page)
        table = soup.find("table", {"id": "gvLAF"})
        if table is None:
            return pd.DataFrame()

        rows = table.find_all("tr")
        data = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cols) >= 2 and cols[0]:
                data.append(cols)

        df = pd.DataFrame(data, columns=["Field", "Value"])
        df = df[df["Field"].str.contains("LAF|MSF|SDF|Call Money", case=False)]
        df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=["Value"])

        return df

    except Exception:
        return pd.DataFrame()

@st.cache_data
def fetch_rbi_m3_weekly():
    """Fetch WEEKLY Monetary Aggregates (M3, RM, Deposits) from WSS Section 8."""
    url = "https://www.rbi.org.in/Scripts/WSS_Section8.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        # Table id for monetary aggregates (may change if RBI updates page)
        table = soup.find("table", {"id": "gvMonetaryAgg"})
        if table is None:
            return pd.DataFrame()

        rows = table.find_all("tr")
        data = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cols) >= 2 and cols[0]:
                data.append(cols)

        df = pd.DataFrame(data, columns=["Indicator", "Value"])
        df = df[df["Indicator"].str.contains(
            "M3|Reserve Money|Currency in Circulation|Deposits",
            case=False
        )]
        df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=["Value"])

        return df

    except Exception:
        return pd.DataFrame()

# --------------------
# AUTO FETCH DATA
# --------------------
st.info("Auto-fetching key macro data (World Bank, FRED, RBI scraping).")

wb_infl_in = fetch_worldbank_indicator("IN", "FP.CPI.TOTL.ZG")
wb_infl_us = fetch_worldbank_indicator("US", "FP.CPI.TOTL.ZG")
m3_in = fetch_worldbank_indicator("IN", "FM.LBL.MQMY.GD.ZS")  # M3-like series

fred_cpi_us = fetch_fred_series("CPIAUCSL") if use_fred else pd.DataFrame()
fed_bs = fetch_fred_series("WALCL") if use_fred else pd.DataFrame()
oil_fred = fetch_fred_series("DCOILBRENTEU") if use_fred else pd.DataFrame()
gold_fred = fetch_fred_series("GOLDAMGBD228NLBM") if use_fred else pd.DataFrame()

rbi_repo = scrape_rbi_repo_rate()

# Default empty containers (for uploads)
liq_monthly = pd.DataFrame()
repo_monthly = pd.DataFrame()

# --------------------
# TABS (Option B: Live Market Data AFTER FX & Commodities)
# --------------------
overview_tab, rbi_live_tab, liq_repo_tab, global_tab, fx_com_tab, live_market_tab, forecast_tab, report_tab = st.tabs(
    [
        "Overview",
        "RBI Live Data",
        "Liquidity & Repo / Credit",
        "Global & Calendar",
        "FX & Commodities",
        "Live Market Data",
        "Forecasting & Indices",
        "News & Reports",
    ]
)

# --------------------
# OVERVIEW TAB
# --------------------
with overview_tab:
    st.subheader("Inflation Dashboard — India & United States (World Bank)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### India — Inflation (annual %)")
        if not wb_infl_in.empty:
            fig = px.line(
                wb_infl_in,
                x="Date",
                y="Value",
                title="India: Inflation (annual %)",
                template=get_plotly_template(),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No World Bank inflation data for India available.")

    with col2:
        st.markdown("#### United States — Inflation (annual %)")
        if not wb_infl_us.empty:
            fig = px.line(
                wb_infl_us,
                x="Date",
                y="Value",
                title="US: Inflation (annual %)",
                template=get_plotly_template(),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No World Bank inflation data for the USA available.")

    if not fred_cpi_us.empty:
        st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### US CPI (FRED - monthly)")
        st.plotly_chart(
            px.line(
                fred_cpi_us.reset_index(),
                x="Date",
                y="Value",
                title="US CPI (FRED, monthly)",
                template=get_plotly_template(),
            ),
            use_container_width=True,
        )

# --------------------
# RBI LIVE DATA TAB
# --------------------
with rbi_live_tab:
    st.subheader("🔵 RBI Live Dashboard — Policy Rates, Liquidity & Monetary Aggregates")

    # --- Live Policy Rates ---
    st.markdown("### LIVE RBI Policy Rates")
    rbi_rates_live = fetch_rbi_policy_rates_live()
    if not rbi_rates_live.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(rbi_rates_live, use_container_width=True)
        with c2:
            core_rates = rbi_rates_live[
                rbi_rates_live["Indicator"].isin(["Repo Rate", "MSF", "SDF", "Bank Rate"])
            ]
            if not core_rates.empty:
                fig_p = px.bar(
                    core_rates,
                    x="Indicator",
                    y="Value",
                    title="RBI Policy Corridor — Repo, MSF, SDF, Bank Rate",
                    template=get_plotly_template(),
                    text_auto=".2f",
                )
                fig_p.update_layout(yaxis_title="Rate (%)")
                st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.warning("Could not fetch LIVE RBI policy rates (Monetary Policy page).")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # --- Live Daily Liquidity (LAF / SDF / MSF / Call) ---
    st.markdown("### LIVE Daily Liquidity (LAF, MSF, SDF, Call Money)")
    laf_live = fetch_rbi_laf_daily()
    if not laf_live.empty:
        c3, c4 = st.columns([1, 2])
        with c3:
            st.dataframe(laf_live, use_container_width=True)
        with c4:
            fig_laf = px.bar(
                laf_live,
                x="Field",
                y="Value",
                title="RBI Daily Liquidity — Net LAF, SDF, MSF, Call Money",
                template=get_plotly_template(),
                text_auto=".0f",
            )
            fig_laf.update_layout(yaxis_title="₹ Crore (approx.)", xaxis_title="")
            st.plotly_chart(fig_laf, use_container_width=True)
    else:
        st.warning("Could not fetch LIVE daily liquidity from RBI (WSS Section 1).")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # --- Live Weekly Monetary Aggregates (M3, RM, Deposits) ---
    st.markdown("### LIVE Weekly Monetary Aggregates (M3, Reserve Money, Deposits)")
    m3_weekly_live = fetch_rbi_m3_weekly()
    if not m3_weekly_live.empty:
        c5, c6 = st.columns([1, 2])
        with c5:
            st.dataframe(m3_weekly_live, use_container_width=True)
        with c6:
            fig_m3 = px.bar(
                m3_weekly_live,
                x="Indicator",
                y="Value",
                title="Weekly Monetary Aggregates — M3, RM, Currency, Deposits",
                template=get_plotly_template(),
                text_auto=".0f",
            )
            fig_m3.update_layout(yaxis_title="₹ Crore (approx.)", xaxis_title="")
            st.plotly_chart(fig_m3, use_container_width=True)
    else:
        st.warning("Could not fetch LIVE weekly monetary aggregates from RBI (WSS Section 8).")

    st.caption(
        "Note: RBI WSS pages sometimes change structure. If a block fails, "
        "you can still rely on uploaded CSVs in the Liquidity & Repo tab."
    )

# --------------------
# LIQUIDITY & REPO / CREDIT TAB
# --------------------
with liq_repo_tab:
    st.subheader("Liquidity & Policy Rate — Data Uploads & Auto Series")

    # ---- Liquidity Upload / Auto ----
    st.markdown("#### Liquidity (M3 / broad money)")
    uploaded_liq = st.file_uploader(
        "Upload Liquidity CSV (Date, M3_IN, optional M3_US)",
        type="csv",
        key="liq_upload",
    )

    if uploaded_liq is None:
        if not m3_in.empty:
            st.write("India — Liquidity (World Bank broad money indicator)")
            st.plotly_chart(
                px.line(
                    m3_in,
                    x="Date",
                    y="Value",
                    title="India M3 (World Bank indicator)",
                    template=get_plotly_template(),
                ),
                use_container_width=True,
            )
        else:
            st.info(
                "No auto M3 data available — you can upload a liquidity CSV to override."
            )
    else:
        try:
            liq_df_raw = pd.read_csv(uploaded_liq)
            st.success("Liquidity CSV loaded! Please map your columns:")
            cols = liq_df_raw.columns.tolist()
            date_col = st.selectbox(
                "Select Date column for liquidity CSV", cols, key="liq_date_col"
            )
            m3_in_col = st.selectbox(
                "Select M3 India column", cols, index=min(1, len(cols) - 1), key="m3_in_col"
            )
            m3_us_col = st.selectbox(
                "Select M3 USA column (optional)", ["None"] + cols, key="m3_us_col"
            )

            liq_df = pd.DataFrame()
            liq_df["Date"] = pd.to_datetime(liq_df_raw[date_col], errors="coerce")
            if liq_df["Date"].isna().any():
                st.warning(
                    "Some dates could not be parsed — rows with invalid dates will be dropped."
                )
                liq_df = liq_df.dropna(subset=["Date"])
            liq_df["M3_IN"] = pd.to_numeric(liq_df_raw[m3_in_col], errors="coerce")
            if m3_us_col != "None":
                liq_df["M3_US"] = pd.to_numeric(liq_df_raw[m3_us_col], errors="coerce")
            liq_df = liq_df.set_index("Date").sort_index()

            diffs = liq_df.index.to_series().diff().dropna()
            median_days = diffs.dt.days.median() if not diffs.empty else 0
            if median_days > 45:
                liq_monthly = liq_df.resample("M").ffill()
                st.info("Converted liquidity series to monthly frequency via forward-fill.")
            else:
                liq_monthly = liq_df.resample("M").last()

            missing_count = liq_monthly.isna().sum().sum()
            if missing_count > 0:
                st.warning(
                    f"Uploaded liquidity has {missing_count} missing values after resampling."
                )
                if st.button("Interpolate missing liquidity values"):
                    liq_monthly = liq_monthly.interpolate()
                    st.success("Interpolated missing liquidity values.")

            smooth_k = st.slider(
                "Liquidity smoothing window (months, 1 = no smoothing)",
                1,
                12,
                1,
                key="liq_smooth",
            )
            plot_liq = liq_monthly.copy()
            if smooth_k > 1:
                plot_liq = plot_liq.rolling(window=smooth_k, min_periods=1).mean()
            st.dataframe(plot_liq.head())
            st.plotly_chart(
                px.line(
                    plot_liq.reset_index(),
                    x="Date",
                    y=plot_liq.columns,
                    title="Uploaded Liquidity Data (Monthly)",
                    template=get_plotly_template(),
                ).update_layout(hovermode="x unified"),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Failed to parse Liquidity CSV: {e}")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # ---- Repo Upload / Scrape ----
    st.markdown("#### Repo Rate (RBI policy rate)")

    uploaded_repo = st.file_uploader(
        "Upload Repo Rate CSV (Date, RepoRate)", type="csv", key="repo_upload"
    )

    if uploaded_repo is None:
        if not rbi_repo.empty:
            latest_repo = rbi_repo.iloc[-1, 0]
            colm1, colm2 = st.columns(2)
            with colm1:
                st.markdown("<div class='rbi-card'>", unsafe_allow_html=True)
                st.markdown("### Latest RBI Repo Rate", unsafe_allow_html=True)
                st.markdown(
                    f"<p><b>{latest_repo:.2f} %</b></p>", unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with colm2:
                st.write("")
            st.plotly_chart(
                px.line(
                    rbi_repo.reset_index().rename(columns={"index": "Date"}),
                    x="Date",
                    y="RepoRate",
                    title="RBI Repo Rate (scraped)",
                    template=get_plotly_template(),
                ).update_xaxes(title="Date"),
                use_container_width=True,
            )
        else:
            st.info(
                "RBI repo rate not found via scraping — upload a Repo CSV for full analysis."
            )
    else:
        try:
            repo_df_raw = pd.read_csv(uploaded_repo)
            st.success("Repo CSV loaded successfully!")
            cols = repo_df_raw.columns.tolist()

            date_col_r = None
            for c in cols:
                if "date" in c.lower() or "month" in c.lower():
                    date_col_r = c
                    break
            if date_col_r is None:
                date_col_r = st.selectbox(
                    "Select Date column for Repo CSV", cols, key="repo_date_col"
                )

            repo_df_raw[date_col_r] = pd.to_datetime(
                repo_df_raw[date_col_r], errors="coerce"
            )
            if repo_df_raw[date_col_r].isna().any():
                st.warning(
                    "Some repo dates could not be parsed — invalid rows will be dropped."
                )
                repo_df_raw = repo_df_raw.dropna(subset=[date_col_r])

            repo_col = None
            for c in cols:
                if "repo" in c.lower() or "rate" in c.lower():
                    repo_col = c
                    break
            if repo_col is None:
                repo_col = st.selectbox(
                    "Select Repo rate column",
                    [c for c in cols if c != date_col_r],
                    key="repo_rate_col",
                )

            repo_df = repo_df_raw[[date_col_r, repo_col]].rename(
                columns={date_col_r: "Date", repo_col: "RepoRate"}
            )
            repo_df["RepoRate"] = pd.to_numeric(
                repo_df["RepoRate"], errors="coerce"
            )
            repo_df = repo_df.set_index("Date").sort_index()
            repo_monthly = repo_df.resample("M").ffill()

            smooth_k_repo = st.slider(
                "Repo smoothing window (months, 1 = no smoothing)",
                1,
                12,
                1,
                key="repo_smooth",
            )
            plot_repo = repo_monthly.copy()
            if smooth_k_repo > 1:
                plot_repo = plot_repo.rolling(
                    window=smooth_k_repo, min_periods=1
                ).mean()
            st.dataframe(plot_repo.head())
            st.plotly_chart(
                px.line(
                    plot_repo.reset_index(),
                    x="Date",
                    y=plot_repo.columns,
                    title="Uploaded Repo Rate (Monthly)",
                    template=get_plotly_template(),
                ).update_layout(hovermode="x unified"),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Failed to read Repo CSV: {e}")

# --------------------
# GLOBAL & CALENDAR TAB
# --------------------
with global_tab:
    st.subheader("Global Monetary Dashboard — Multi-country Inflation (World Bank)")

    country_map = {
        "India": ("IN", "FP.CPI.TOTL.ZG"),
        "United States": ("US", "FP.CPI.TOTL.ZG"),
        "United Kingdom": ("GB", "FP.CPI.TOTL.ZG"),
        "Euro Area": ("XZ", "FP.CPI.TOTL.ZG"),  # best-effort WB code for Euro area
        "Japan": ("JP", "FP.CPI.TOTL.ZG"),
        "China": ("CN", "FP.CPI.TOTL.ZG"),
    }

    sel_countries = st.multiselect(
        "Select countries to compare (World Bank inflation)",
        options=list(country_map.keys()),
        default=["India", "United States"],
    )

    global_df = pd.DataFrame()
    for cname in sel_countries:
        ccode, ind = country_map[cname]
        df_c = fetch_worldbank_indicator(ccode, ind)
        if not df_c.empty:
            df_c = df_c[
                (df_c["Date"].dt.year >= start_year)
                & (df_c["Date"].dt.year <= end_year)
            ]
            df_c["Country"] = cname
            global_df = pd.concat([global_df, df_c], ignore_index=True)

    if not global_df.empty:
        fig_global = px.line(
            global_df,
            x="Date",
            y="Value",
            color="Country",
            title="Global Inflation Comparison (World Bank)",
            template=get_plotly_template(),
        )
        st.plotly_chart(fig_global, use_container_width=True)
    else:
        st.info(
            "No global data fetched — try a different selection of countries or years."
        )

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    st.subheader("Economic Calendar (Monetary Policy & Macro Events)")
    cal_df = sample_econ_calendar()
    cal_json = st.text_area(
        "Edit or add events (JSON list with Event, Date, Type, Importance):",
        value=json.dumps(cal_df.to_dict(orient="records"), indent=2),
        height=200,
    )

    try:
        cal_parsed = pd.DataFrame(json.loads(cal_json))
        cal_parsed["Date"] = pd.to_datetime(cal_parsed["Date"], errors="coerce")
        cal_parsed = cal_parsed.sort_values("Date")
        st.dataframe(cal_parsed)
        fig_cal = px.timeline(
            cal_parsed,
            x_start="Date",
            x_end="Date",
            y="Event",
            color="Importance",
            title="Timeline of Key Events",
            template=get_plotly_template(),
        )
        fig_cal.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_cal, use_container_width=True)
    except Exception as e:
        st.warning(f"Calendar JSON invalid: {e}")

# --------------------
# FX & COMMODITIES TAB
# --------------------
with fx_com_tab:
    st.subheader("Forex Dashboard — INR vs Major Currencies")

    fx_df = fetch_forex_inr()
    if not fx_df.empty:
        fx_choice = st.multiselect(
            "Select FX pairs (value = foreign currency per INR)",
            options=list(fx_df.columns),
            default=["USD", "EUR"],
        )
        if fx_choice:
            fig_fx = px.line(
                fx_df.reset_index(),
                x="Date",
                y=fx_choice,
                title="INR FX Rates (exchangerate.host)",
                template=get_plotly_template(),
            )
            st.plotly_chart(fig_fx, use_container_width=True)
        st.write("Higher line = INR stronger (1 INR buys more foreign currency).")
    else:
        st.info("Could not fetch FX data (API/network issue).")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    st.subheader("Commodity Dashboard — Oil & Gold (FRED)")
    if not oil_fred.empty or not gold_fred.empty:
        com_df = pd.DataFrame()
        if not oil_fred.empty:
            tmp = oil_fred.rename(columns={"Value": "BrentOil"})
            com_df = tmp if com_df.empty else com_df.join(tmp, how="outer")
        if not gold_fred.empty:
            tmp = gold_fred.rename(columns={"Value": "GoldPrice"})
            com_df = tmp if com_df.empty else com_df.join(tmp, how="outer")
        com_df = com_df.dropna(how="all")
        st.dataframe(com_df.tail())
        fig_com = px.line(
            com_df.reset_index(),
            x="Date",
            y=com_df.columns,
            title="Brent Oil & Gold Prices (FRED)",
            template=get_plotly_template(),
        )
        st.plotly_chart(fig_com, use_container_width=True)
    else:
        st.info("Commodity data not available (FRED disabled or API issue).")

# --------------------
# LIVE MARKET DATA TAB (from yfinance module) - Option B placement
# --------------------
with live_market_tab:
    # call our plugin function to render full live market data UI
    render_live_market_tab()

# --------------------
# FORECASTING & INDICES TAB
# --------------------
with forecast_tab:
    st.subheader("Forecasting & Monetary Indices")

    # --- Risk-o-meter & Volatility (annual World Bank series) ---
    st.markdown("#### Risk-o-meter: Inflation Volatility (World Bank annual)")
    combined_annual = pd.DataFrame()
    if not wb_infl_in.empty:
        combined_annual["IN_infl"] = wb_infl_in["Value"].values
    if not wb_infl_us.empty:
        combined_annual["US_infl"] = wb_infl_us["Value"].values

    if not combined_annual.empty:
        win = 3
        vol = combined_annual.rolling(window=win, min_periods=1).std()
        fig = go.Figure()
        if not wb_infl_in.empty:
            fig.add_trace(
                go.Scatter(
                    x=wb_infl_in["Date"],
                    y=vol["IN_infl"],
                    name="India volatility",
                )
            )
        if not wb_infl_us.empty:
            fig.add_trace(
                go.Scatter(
                    x=wb_infl_us["Date"],
                    y=vol["US_infl"],
                    name="US volatility",
                )
            )
        fig.update_layout(
            title="Rolling Volatility (Annual Inflation, 3-year window)",
            template=get_plotly_template(),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Not enough inflation series to compute volatility (World Bank data is annual)."
        )

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # --- Forecasting (ARIMA) for India inflation ---
    st.markdown("#### Forecasting (ARIMA) — India Inflation (annual)")

    if not wb_infl_in.empty:
        try:
            s = wb_infl_in.set_index("Date")["Value"].astype(float).dropna()
            model = ARIMA(s, order=(2, 1, 2)).fit()
            fc = model.forecast(steps=forecast_horizon)
            last_year = s.index.year[-1]
            years = [last_year + i for i in range(1, len(fc) + 1)]
            df_fc = pd.DataFrame({"Year": years, "Forecast": fc.values})
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=s.index.year, y=s.values, name="Historical")
            )
            fig.add_trace(
                go.Scatter(
                    x=df_fc["Year"],
                    y=df_fc["Forecast"],
                    name="Forecast",
                )
            )
            fig.update_layout(
                title="ARIMA Forecast — India Inflation (annual)",
                template=get_plotly_template(),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")
    else:
        st.info("No series available for ARIMA forecasting (India).")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # --- Monetary Policy Index (MPI) & Monetary Stress Index (MSI) ---
    st.markdown("#### Monetary Policy Index (MPI) & Monetary Stress Index (MSI)")

    mpi, msi, infl_trend, rate_text, stress_text, missing = compute_mpi_msi_and_insights(
        wb_infl_in,
        m3_in,
        rbi_repo,
        liq_monthly,
        repo_monthly,
        w_infl,
        w_rate,
        w_liq,
        w_vol,
    )

    col_mpi, col_msi = st.columns(2)
    with col_mpi:
        st.markdown("<div class='rbi-card'>", unsafe_allow_html=True)
        st.markdown("### Monetary Policy Index (MPI)", unsafe_allow_html=True)
        st.markdown(
            f"<p><b>{round(mpi, 4)}</b></p>", unsafe_allow_html=True
        )
        st.markdown(
            f"<p>Weights — Inflation: {w_infl:.2f}, Rate: {w_rate:.2f}, "
            f"Liquidity: {w_liq:.2f}, Volatility: {w_vol:.2f}</p>",
            unsafe_allow_html=True,
        )
        if missing >= 3:
            st.markdown(
                "<p><i>Note: Many inputs are missing, MPI may be unreliable. "
                "Upload Repo & Liquidity CSVs for better accuracy.</i></p>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_msi:
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=msi,
                title={"text": "Monetary Stress Index (MSI)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40], "color": "green"},
                        {"range": [40, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"},
                    ],
                },
            )
        )
        fig_g.update_layout(template=get_plotly_template(), height=260)
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # --- Insights (rule-based, no external AI API) ---
    st.subheader("Qualitative Insights (Rule-based)")

    st.markdown(
        f"""
**Inflation trend (India):** {infl_trend}  

**Policy rate stance:** {rate_text}  

**Monetary stress assessment:** {stress_text}  

You can directly use these insights as commentary in your project report or presentation.
"""
    )

# --------------------
# NEWS & REPORTS TAB
# --------------------
with report_tab:
    st.subheader("RBI Monetary Policy Press Releases (Auto-scraped)")

    rbi_news = scrape_rbi_press_releases()
    if not rbi_news.empty:
        st.dataframe(rbi_news)
    else:
        st.info(
            "Could not fetch RBI press releases (or no recent monetary policy items detected)."
        )

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # Re-compute MPI/MSI & insights for use in PDF (same function)
    mpi_pdf, msi_pdf, infl_trend_pdf, rate_text_pdf, stress_text_pdf, _ = (
        compute_mpi_msi_and_insights(
            wb_infl_in,
            m3_in,
            rbi_repo,
            liq_monthly,
            repo_monthly,
            w_infl,
            w_rate,
            w_liq,
            w_vol,
        )
    )

    st.subheader("Export: Detailed PDF Report (with Charts & Commentary)")

    @st.cache_data
    def save_figures_for_pdf(template_name):
        out = []
        os.makedirs("tmp_imgs", exist_ok=True)
        try:
            if not wb_infl_in.empty:
                f = "tmp_imgs/india_infl.png"
                fig = px.line(
                    wb_infl_in,
                    x="Date",
                    y="Value",
                    title="India Inflation",
                    template=template_name,
                )
                try:
                    fig.write_image(f)
                    out.append(f)
                except Exception:
                    pass
            if not wb_infl_us.empty:
                f = "tmp_imgs/usa_infl.png"
                fig = px.line(
                    wb_infl_us,
                    x="Date",
                    y="Value",
                    title="USA Inflation",
                    template=template_name,
                )
                try:
                    fig.write_image(f)
                    out.append(f)
                except Exception:
                    pass
            if not repo_monthly.empty:
                f = "tmp_imgs/repo.png"
                fig = px.line(
                    repo_monthly.reset_index(),
                    x="Date",
                    y=repo_monthly.columns[0],
                    title="Repo Rate (Monthly)",
                    template=template_name,
                )
                try:
                    fig.write_image(f)
                    out.append(f)
                except Exception:
                    pass
            if not fed_bs.empty:
                f = "tmp_imgs/fed_bs.png"
                fig = px.line(
                    fed_bs.reset_index(),
                    x="Date",
                    y="Value",
                    title="Fed Balance Sheet",
                    template=template_name,
                )
                try:
                    fig.write_image(f)
                    out.append(f)
                except Exception:
                    pass
        except Exception:
            pass
        return out

    def generate_pdf_with_images(images, texts, filename="Monetary_Report.pdf"):
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Monetary Policy Detailed Report", styles["Title"]))
        story.append(Spacer(1, 12))
        for t in texts:
            story.append(Paragraph(t, styles["BodyText"]))
            story.append(Spacer(1, 12))
        for img in images:
            if os.path.exists(img):
                story.append(Image(img, width=480, height=300))
                story.append(Spacer(1, 12))
        doc.build(story)
        return filename

    if st.button("Generate & Download PDF"):
        imgs = save_figures_for_pdf(get_plotly_template())
        texts = [
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"MPI (Monetary Policy Index): {round(mpi_pdf,4)}",
            f"MSI (Monetary Stress Index): {round(msi_pdf,2)}",
            f"Inflation trend (India): {infl_trend_pdf}",
            f"Policy stance: {rate_text_pdf}",
            f"Monetary stress assessment: {stress_text_pdf}",
        ]
        pdf = generate_pdf_with_images(imgs, texts)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf)

    st.caption(
        "End of RBI-style monetary policy dashboard. "
        "If any auto-fetch failed, upload CSVs or check network/API keys."
    )

# Optionally render compact live panel in sidebar
with st.sidebar:
    try:
        render_live_sidebar_panel()
    except Exception:
        pass

# End of file
