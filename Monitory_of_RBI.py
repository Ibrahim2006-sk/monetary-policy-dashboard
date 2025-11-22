# Monitory_of_RBI_full_app_fixed.py
"""
Monetary Policy Dashboard — RBI & Global (Auto + Uploads + RBI UI)

STYLE E — RBI Theme:
- RBI-like blue & white colors
- Official, institutional look
- Tabs for navigation
- KPI cards, clean layout

Features:
- World Bank inflation data (India, US, others)
- FRED macro series (US CPI, Fed balance sheet, Oil, Gold)
- Liquidity & Repo upload + merge + correlation
- Volatility & ARIMA forecast for India inflation
- Monetary Policy Index (MPI)
- Monetary Stress Index (MSI)
- Global inflation dashboard (multi-country)
- Editable Economic Calendar
- Yield curve visualizer (upload)
- MPC vote tracker (upload)
- Liquidity stress heatmap
- RBI press release scraper
- Forex INR dashboard (USD, EUR, GBP, JPY)
- Commodity dashboard (Oil & Gold)
- Bank credit dashboard (upload)
- Rule-based insights (no external AI API)
- Enhanced PDF export with charts
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

# ------------------------------
# RBI OFFICIAL UI THEME (STYLE E)
# ------------------------------
rbi_css = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, .stApp {
    background-color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Merriweather', serif !important;
    color: #003366 !important;
}

.stApp {
    padding-top: 0rem !important;
}

/* TOP HEADER BAR (RBI STYLE) */
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

/* KPI STAT CARDS */
.rbi-card {
    padding: 16px;
    border: 2px solid #00336633;
    border-radius: 10px;
    background-color: #F2F7FF;
    box-shadow: 0px 2px 6px #00336622;
}

.rbi-card h3 {
    color: #003366 !important;
    margin-bottom: 6px;
}

.rbi-card p {
    color: #003366;
    font-size: 18px;
}

/* TABS (OFFICIAL LOOK) */
.stTabs [role="tablist"] button {
    font-size: 15px;
    font-weight: 600;
    background-color: #E8EEF7;
    color: #003366 !important;
    border-radius: 5px;
    margin-right: 4px;
    padding: 6px 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #003366 !important;
    color: white !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #F1F4FA !important;
    border-right: 3px solid #00336633;
}

section[data-testid="stSidebar"] h2, h3, h4, label {
    color: #003366 !important;
}

/* BUTTONS */
.stButton > button {
    background-color: #003366 !important;
    color: white !important;
    border-radius: 6px;
    padding: 8px 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #0055AA !important;
    color: #ffffff !important;
}

/* DIVIDER LINE */
.rbi-divider {
    height: 3px;
    background-color: #00336633;
    margin: 25px 0px;
    border-radius: 2px;
}
</style>
"""

# --------------------
# CONFIG
# --------------------
FRED_API_KEY = "f1d8b90ecd833ea9092dca3882935787"
PLOTLY_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Monetary Policy Dashboard (RBI Theme)",
    layout="wide"
)

# Apply RBI CSS
st.markdown(rbi_css, unsafe_allow_html=True)

# RBI-style header
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
    """
    Fetch INR FX using exchangerate.host timeseries (base=INR).
    Returns Date index, columns like 'USD', 'EUR' (price of 1 INR in foreign currency).
    """
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
# TABS
# --------------------
overview_tab, liq_repo_tab, global_tab, fx_com_tab, forecast_tab, report_tab = st.tabs(
    [
        "Overview",
        "Liquidity & Repo / Credit",
        "Global & Calendar",
        "FX & Commodities",
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
                template=PLOTLY_TEMPLATE,
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
                template=PLOTLY_TEMPLATE,
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
                template=PLOTLY_TEMPLATE,
            ),
            use_container_width=True,
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
                    template=PLOTLY_TEMPLATE,
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
                    template=PLOTLY_TEMPLATE,
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
                    template=PLOTLY_TEMPLATE,
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
                    template=PLOTLY_TEMPLATE,
                ).update_layout(hovermode="x unified"),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Failed to read Repo CSV: {e}")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # ---- MERGED SERIES + CORRELATION + STRESS HEATMAP ----
    st.markdown("#### Combined Liquidity, Repo & CPI (monthly)")

    merged = pd.DataFrame()
    if not liq_monthly.empty:
        merged = liq_monthly.copy()
    if not repo_monthly.empty:
        if merged.empty:
            merged = repo_monthly.copy()
        else:
            merged = merged.join(repo_monthly, how="outer")
    if not fred_cpi_us.empty:
        if merged.empty:
            merged = fred_cpi_us.rename(columns={"Value": "US_CPI"}).copy()
        else:
            merged = merged.join(
                fred_cpi_us.rename(columns={"Value": "US_CPI"}), how="left"
            )
    if not wb_infl_in.empty:
        try:
            wb_in_in_monthly = (
                wb_infl_in.set_index("Date")
                .resample("M")
                .ffill()
                .rename(columns={"Value": "IN_Inflation"})
            )
            merged = merged.join(wb_in_in_monthly["IN_Inflation"], how="left")
        except Exception:
            pass

    if not merged.empty:
        st.dataframe(merged.tail())
        fig = go.Figure()
        for c in merged.columns:
            fig.add_trace(
                go.Scatter(x=merged.index, y=merged[c], name=c, mode="lines")
            )
        fig.update_layout(
            title="Combined Liquidity + Repo + CPI (monthly)",
            template=PLOTLY_TEMPLATE,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Correlation Matrix (merged series)")
        corr = merged.corr()
        try:
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception:
            st.dataframe(corr)

        st.markdown("#### Liquidity Stress Heatmap (normalized merged series)")
        try:
            z = (merged - merged.mean()) / merged.std(ddof=0)
            fig_heat = px.imshow(
                z.T,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels={"x": "Time", "y": "Variable", "color": "Z-score"},
                title="Liquidity Stress Heatmap",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute stress heatmap: {e}")
    else:
        st.info(
            "No merged monthly series yet — upload liquidity and/or repo CSVs or enable FRED."
        )

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # ---- YIELD CURVE VISUALIZER ----
    st.markdown("#### Yield Curve Visualizer (Upload Government Bond Yields)")
    yc_file = st.file_uploader(
        "Upload Yield Curve CSV (Date + maturities like 3M, 1Y, 2Y, 5Y, 10Y)",
        type="csv",
        key="yc_upload",
    )

    if yc_file is not None:
        try:
            yc_df_raw = pd.read_csv(yc_file)
            st.success("Yield curve CSV loaded.")
            cols = yc_df_raw.columns.tolist()
            date_col_y = st.selectbox("Select Date column", cols, key="yc_date_col")
            mat_cols = st.multiselect(
                "Select maturity columns (yields)",
                [c for c in cols if c != date_col_y],
                default=[c for c in cols if c != date_col_y],
            )
            yc_df_raw[date_col_y] = pd.to_datetime(
                yc_df_raw[date_col_y], errors="coerce"
            )
            yc_df = yc_df_raw.dropna(subset=[date_col_y])
            yc_df = yc_df.sort_values(date_col_y)
            last_date = yc_df[date_col_y].max()
            st.write(f"Latest yield curve date: {last_date.date()}")
            last_row = yc_df[yc_df[date_col_y] == last_date][mat_cols].iloc[0]
            plot_df = pd.DataFrame(
                {"Maturity": mat_cols, "Yield": last_row.values}
            )
            fig_yc = px.line(
                plot_df,
                x="Maturity",
                y="Yield",
                markers=True,
                title="Latest Yield Curve",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_yc, use_container_width=True)
        except Exception as e:
            st.warning(f"Error parsing yield curve file: {e}")
    else:
        st.info("Upload a yield curve CSV to visualize term structure of interest rates.")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # ---- BANK CREDIT DASHBOARD ----
    st.markdown("#### Bank Credit Dashboard (Upload Credit Data)")
    credit_file = st.file_uploader(
        "Upload Bank Credit CSV (Date, TotalCredit or sectoral columns)",
        type="csv",
        key="credit_upload",
    )
    if credit_file is not None:
        try:
            cr_df = pd.read_csv(credit_file)
            cols = cr_df.columns.tolist()
            date_col_cr = st.selectbox(
                "Select Date column", cols, key="credit_date_col"
            )
            cr_df[date_col_cr] = pd.to_datetime(
                cr_df[date_col_cr], errors="coerce"
            )
            cr_df = cr_df.dropna(subset=[date_col_cr]).sort_values(date_col_cr)
            val_cols = [c for c in cols if c != date_col_cr]
            st.dataframe(cr_df.tail())
            fig_cr = px.line(
                cr_df,
                x=date_col_cr,
                y=val_cols,
                title="Bank Credit Over Time",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_cr, use_container_width=True)

            if "TotalCredit" in cr_df.columns:
                cr_df["CreditGrowthYoY"] = (
                    cr_df["TotalCredit"].pct_change(periods=12) * 100
                )
                fig_gry = px.line(
                    cr_df,
                    x=date_col_cr,
                    y="CreditGrowthYoY",
                    title="YoY Bank Credit Growth (%)",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_gry, use_container_width=True)
        except Exception as e:
            st.warning(f"Error parsing credit data: {e}")
    else:
        st.info("Upload bank credit data to analyze total and sectoral credit trends.")

    st.markdown("<div class='rbi-divider'></div>", unsafe_allow_html=True)

    # ---- MPC VOTE TRACKER ----
    st.markdown("#### MPC Vote Tracker (Upload)")
    mpc_file = st.file_uploader(
        "Upload MPC vote CSV (MeetingDate, HikeVotes, HoldVotes, CutVotes)",
        type="csv",
        key="mpc_upload",
    )
    if mpc_file is not None:
        try:
            mpc_df = pd.read_csv(mpc_file)
            date_col_mpc = None
            for col in mpc_df.columns:
                if "date" in col.lower():
                    mpc_df[col] = pd.to_datetime(mpc_df[col], errors="coerce")
                    date_col_mpc = col
                    break
            if date_col_mpc is None:
                st.warning(
                    "No date column detected — please include a column like 'MeetingDate'."
                )
            else:
                mpc_df = mpc_df.dropna(subset=[date_col_mpc])
                mpc_df = mpc_df.sort_values(date_col_mpc)
                mpc_melt = mpc_df.melt(
                    id_vars=[date_col_mpc],
                    value_vars=[c for c in mpc_df.columns if "Votes" in c],
                    var_name="VoteType",
                    value_name="Count",
                )
                fig_mpc = px.bar(
                    mpc_melt,
                    x=date_col_mpc,
                    y="Count",
                    color="VoteType",
                    barmode="stack",
                    title="MPC Voting Pattern Over Time",
                    template=PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig_mpc, use_container_width=True)
        except Exception as e:
            st.warning(f"Error parsing MPC file: {e}")
    else:
        st.info("Upload MPC vote data to visualize hawkish vs dovish voting patterns.")

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
            template=PLOTLY_TEMPLATE,
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
            template=PLOTLY_TEMPLATE,
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
                template=PLOTLY_TEMPLATE,
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
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_com, use_container_width=True)
    else:
        st.info("Commodity data not available (FRED disabled or API issue).")

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
            template=PLOTLY_TEMPLATE,
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
                template=PLOTLY_TEMPLATE,
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
        fig_g.update_layout(template=PLOTLY_TEMPLATE, height=260)
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
        imgs = save_figures_for_pdf(PLOTLY_TEMPLATE)
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
