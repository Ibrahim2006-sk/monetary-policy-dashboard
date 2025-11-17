# Monitory_of_RBI_full_app_fixed.py
"""
Cleaned, error-free Streamlit app - Monetary Policy Dashboard (Auto + Uploads)
"""

import os
import re
import requests
from datetime import datetime

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

# --------------------
# CONFIG
# --------------------
# Your FRED API key (you provided earlier)
FRED_API_KEY = "f1d8b90ecd833ea9092dca3882935787"
PLOTLY_TEMPLATE = "plotly_dark"

st.set_page_config(page_title="Monetary Policy Dashboard (Auto & Upload)", layout="wide")
st.title("Monetary Policy Dashboard — RBI & Fed (Auto-data + Uploads)")

# --------------------
# HELPERS
# --------------------
@st.cache_data
def fetch_worldbank_indicator(country_code, indicator, per_page=2000):
    """Fetch World Bank indicator. Returns DataFrame with Date and Value columns."""
    try:
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page={per_page}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        if not isinstance(j, list) or len(j) < 2:
            return pd.DataFrame()
        data = j[1]
        df = pd.DataFrame([{"Date": item.get("date"), "Value": item.get("value")} for item in data if item.get("value") is not None])
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
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        if start:
            url += f"&observation_start={start}"
        if end:
            url += f"&observation_end={end}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations", [])
        df = pd.DataFrame([{"Date": pd.to_datetime(o["date"]), "Value": (float(o["value"]) if o["value"] != "." else np.nan)} for o in obs])
        df = df.dropna().set_index("Date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def scrape_rbi_repo_rate():
    """Best-effort scraping of RBI pages to find latest repo rate (returns small DF or empty DF)."""
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
            m = re.search(r"Repo[ -]rate[^0-9\n\%\.]*([0-9]+\.?[0-9]*)", text, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                df = pd.DataFrame({"RepoRate": [val]}, index=[pd.to_datetime("today")])
                return df
        except Exception:
            continue
    return pd.DataFrame()

def money_fmt(x, currency="INR"):
    try:
        v = float(x)
        if currency == "INR":
            return "₹" + f"{v:,.2f}"
        if currency == "USD":
            return "$" + f"{v:,.2f}"
        return f"{v:,.2f}"
    except Exception:
        return x

# --------------------
# SIDEBAR CONTROLS
# --------------------
with st.sidebar:
    st.markdown("## Controls")
    start_year = st.number_input("Start year", min_value=1960, max_value=datetime.now().year, value=2000)
    end_year = st.number_input("End year", min_value=1960, max_value=datetime.now().year, value=datetime.now().year)
    use_fred = st.checkbox("Use FRED for US monthly CPI & Fed data (needs API key)", value=True)
    st.markdown("---")
    st.markdown("Forecasting & MPI")
    forecast_horizon = st.number_input("Forecast horizon (years/steps)", min_value=1, max_value=60, value=5)
    w_infl = st.slider("Inflation weight", 0.0, 1.0, 0.4)
    w_rate = st.slider("Policy rate weight", 0.0, 1.0, 0.25)
    w_liq = st.slider("Liquidity weight", 0.0, 1.0, 0.2)
    w_vol = st.slider("Volatility weight", 0.0, 1.0, 0.15)

# --------------------
# AUTO FETCH DATA
# --------------------
st.info("Fetching auto data — this may take a few seconds on first run...")

wb_infl_in = fetch_worldbank_indicator("IN", "FP.CPI.TOTL.ZG")
wb_infl_us = fetch_worldbank_indicator("US", "FP.CPI.TOTL.ZG")
m3_in = fetch_worldbank_indicator("IN", "FM.LBL.MQMY.GD.ZS")  # best-effort M3-like series
fred_cpi_us = fetch_fred_series("CPIAUCSL") if use_fred else pd.DataFrame()
fed_bs = fetch_fred_series("WALCL") if use_fred else pd.DataFrame()
rbi_repo = scrape_rbi_repo_rate()

# --------------------
# PLOT: Inflation (World Bank)
# --------------------
st.subheader("Inflation (World Bank — annual %)")
col1, col2 = st.columns(2)
with col1:
    st.write("India — Inflation (annual %)")
    if not wb_infl_in.empty:
        fig = px.line(wb_infl_in, x="Date", y="Value", title="India: Inflation (annual %)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No World Bank data for India available.")
with col2:
    st.write("United States — Inflation (annual %)")
    if not wb_infl_us.empty:
        fig = px.line(wb_infl_us, x="Date", y="Value", title="US: Inflation (annual %)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No World Bank data for USA available.")

if not fred_cpi_us.empty:
    st.subheader("US CPI (FRED - monthly)")
    st.plotly_chart(px.line(fred_cpi_us.reset_index(), x="Date", y="Value", title="US CPI (FRED)", template=PLOTLY_TEMPLATE), use_container_width=True)

# --------------------
# Liquidity & Repo (auto + upload + mapping + validation + merge)
# --------------------
st.subheader("Liquidity & Policy Rate")

# Upload liquidity CSV (with mapping UI)
uploaded_liq = st.file_uploader("Upload Liquidity CSV (Date, M3_IN, optional M3_US)", type="csv", key="liq_upload")
liq_monthly = pd.DataFrame()
if uploaded_liq is None:
    if not m3_in.empty:
        st.write("India — Liquidity (best-effort, World Bank series)")
        st.plotly_chart(px.line(m3_in, x="Date", y="Value", title="India M3 (World Bank indicator)", template=PLOTLY_TEMPLATE), use_container_width=True)
    else:
        st.info("No auto M3 data available — you can upload liquidity CSV.")
else:
    # parse and validate uploaded liquidity CSV
    try:
        liq_df_raw = pd.read_csv(uploaded_liq)
        st.success("Liquidity CSV loaded! Please map your columns:")
        cols = liq_df_raw.columns.tolist()
        date_col = st.selectbox("Select Date column for liquidity CSV", cols, key="liq_date_col")
        m3_in_col = st.selectbox("Select M3 India column", cols, index=min(1, len(cols) - 1), key="m3_in_col")
        m3_us_col = st.selectbox("Select M3 USA column (optional)", ["None"] + cols, key="m3_us_col")
        # build cleaned df
        liq_df = pd.DataFrame()
        liq_df["Date"] = pd.to_datetime(liq_df_raw[date_col], errors="coerce")
        if liq_df["Date"].isna().any():
            st.warning("Some dates could not be parsed — rows with invalid dates will be dropped.")
            liq_df = liq_df.dropna(subset=["Date"])
        liq_df["M3_IN"] = pd.to_numeric(liq_df_raw[m3_in_col], errors="coerce")
        if m3_us_col != "None":
            liq_df["M3_US"] = pd.to_numeric(liq_df_raw[m3_us_col], errors="coerce")
        liq_df = liq_df.set_index("Date").sort_index()
        # convert to monthly frequency
        diffs = liq_df.index.to_series().diff().dropna()
        median_days = diffs.dt.days.median() if not diffs.empty else 0
        if median_days > 45:
            liq_monthly = liq_df.resample("M").ffill()
            st.info("Converted liquidity series to monthly frequency via forward-fill.")
        else:
            liq_monthly = liq_df.resample("M").last()
        # missing values and interpolation option
        missing_count = liq_monthly.isna().sum().sum()
        if missing_count > 0:
            st.warning(f"Uploaded liquidity has {missing_count} missing values after resampling.")
            if st.button("Interpolate missing liquidity values"):
                liq_monthly = liq_monthly.interpolate()
                st.success("Interpolated missing liquidity values.")
        # smoothing option
        smooth_k = st.slider("Liquidity smoothing window (months, 1 = no smoothing)", 1, 12, 1, key="liq_smooth")
        plot_liq = liq_monthly.copy()
        if smooth_k > 1:
            plot_liq = plot_liq.rolling(window=smooth_k, min_periods=1).mean()
        st.dataframe(plot_liq.head())
        st.plotly_chart(px.line(plot_liq.reset_index(), x="Date", y=plot_liq.columns, title="Uploaded Liquidity Data (Monthly)", template=PLOTLY_TEMPLATE).update_layout(hovermode="x unified"), use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to parse Liquidity CSV: {e}")

# Upload repo CSV (mapping + validation)
uploaded_repo = st.file_uploader("Upload Repo Rate CSV (Date, RepoRate)", type="csv", key="repo_upload")
repo_monthly = pd.DataFrame()
if uploaded_repo is None:
    if not rbi_repo.empty:
        latest_repo = rbi_repo.iloc[-1, 0]
        st.metric("Latest RBI Repo Rate", f"{latest_repo} %")
        st.plotly_chart(px.line(rbi_repo.reset_index().rename(columns={"index": "Date"}), x="Date", y="RepoRate", title="RBI Repo Rate (scraped)", template=PLOTLY_TEMPLATE).update_xaxes(title="Date"), use_container_width=True)
    else:
        st.info("RBI repo rate not found via scraping — please upload repo CSV if needed.")
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
            date_col_r = st.selectbox("Select Date column for Repo CSV", cols, key="repo_date_col")
        repo_df_raw[date_col_r] = pd.to_datetime(repo_df_raw[date_col_r], errors="coerce")
        if repo_df_raw[date_col_r].isna().any():
            st.warning("Some repo dates could not be parsed — invalid rows will be dropped.")
            repo_df_raw = repo_df_raw.dropna(subset=[date_col_r])
        # attempt to find repo rate column
        repo_col = None
        for c in cols:
            if "repo" in c.lower() or "rate" in c.lower():
                repo_col = c
                break
        if repo_col is None:
            repo_col = st.selectbox("Select Repo rate column", [c for c in cols if c != date_col_r], key="repo_rate_col")
        repo_df = repo_df_raw[[date_col_r, repo_col]].rename(columns={date_col_r: "Date", repo_col: "RepoRate"})
        repo_df["RepoRate"] = pd.to_numeric(repo_df["RepoRate"], errors="coerce")
        repo_df = repo_df.set_index("Date").sort_index()
        repo_monthly = repo_df.resample("M").ffill()
        smooth_k_repo = st.slider("Repo smoothing window (months, 1 = no smoothing)", 1, 12, 1, key="repo_smooth")
        plot_repo = repo_monthly.copy()
        if smooth_k_repo > 1:
            plot_repo = plot_repo.rolling(window=smooth_k_repo, min_periods=1).mean()
        st.dataframe(plot_repo.head())
        st.plotly_chart(px.line(plot_repo.reset_index(), x="Date", y=plot_repo.columns, title="Uploaded Repo Rate (Monthly)", template=PLOTLY_TEMPLATE).update_layout(hovermode="x unified"), use_container_width=True)
    except Exception as e:
        st.warning(f"Failed to read Repo CSV: {e}")

# --------------------
# AUTO-MERGE uploaded series with CPI/FRED & combined charts + correlation
# --------------------
# Build merged frame if any of uploaded sources present
merged = pd.DataFrame()
if "liq_monthly" in locals() and not liq_monthly.empty:
    merged = liq_monthly.copy()
if "repo_monthly" in locals() and not repo_monthly.empty:
    if merged.empty:
        merged = repo_monthly.copy()
    else:
        merged = merged.join(repo_monthly, how="outer")
# join fred_cpi_us (monthly) or wb_infl_in (annual -> forward-fill to monthly)
if not fred_cpi_us.empty:
    if merged.empty:
        merged = fred_cpi_us.rename(columns={"Value": "US_CPI"}).copy()
    else:
        merged = merged.join(fred_cpi_us.rename(columns={"Value": "US_CPI"}), how="left")
if not wb_infl_in.empty:
    # convert annual series to monthly by reindexing and forward-fill
    try:
        wb_in_in_monthly = wb_infl_in.set_index("Date").resample("M").ffill().rename(columns={"Value": "IN_Inflation"})
        merged = merged.join(wb_in_in_monthly["IN_Inflation"], how="left")
    except Exception:
        pass

if not merged.empty:
    st.subheader("Merged Liquidity / Repo / CPI (monthly)")
    st.dataframe(merged.tail())
    # combined chart
    fig = go.Figure()
    for c in merged.columns:
        fig.add_trace(go.Scatter(x=merged.index, y=merged[c], name=c, mode="lines"))
    fig.update_layout(title="Combined Liquidity + Repo + CPI (monthly)", template=PLOTLY_TEMPLATE, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    # correlation matrix
    st.subheader("Correlation Matrix (merged series)")
    corr = merged.corr()
    try:
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception:
        st.dataframe(corr)

# --------------------
# Risk-o-meter & Volatility (annual World Bank series)
# --------------------
st.subheader("Risk-o-meter: inflation volatility (annual WB series)")
combined_annual = pd.DataFrame()
if not wb_infl_in.empty:
    combined_annual["IN_infl"] = wb_infl_in["Value"].values
if not wb_infl_us.empty:
    combined_annual["US_infl"] = wb_infl_us["Value"].values

if not combined_annual.empty:
    win = 3
    vol = combined_annual.rolling(window=win, min_periods=1).std()
    fig = go.Figure()
    # align x-axis properly with respective dates
    if not wb_infl_in.empty:
        fig.add_trace(go.Scatter(x=wb_infl_in["Date"], y=vol["IN_infl"], name="IN volatility"))
    if not wb_infl_us.empty:
        fig.add_trace(go.Scatter(x=wb_infl_us["Date"], y=vol["US_infl"], name="US volatility"))
    fig.update_layout(title="Rolling volatility (annual inflation)", template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough inflation series to compute volatility (World Bank data is annual).")

# --------------------
# Forecasting (ARIMA) for India inflation (annual WB)
# --------------------
st.subheader("Forecasting (ARIMA) - India Inflation (annual WB)")
if not wb_infl_in.empty:
    try:
        s = wb_infl_in.set_index("Date")["Value"].astype(float).dropna()
        model = ARIMA(s, order=(2, 1, 2)).fit()
        fc = model.forecast(steps=forecast_horizon)
        last_year = s.index.year[-1]
        years = [last_year + i for i in range(1, len(fc) + 1)]
        df_fc = pd.DataFrame({"Year": years, "Forecast": fc.values})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index.year, y=s.values, name="Historical"))
        fig.add_trace(go.Scatter(x=df_fc["Year"], y=df_fc["Forecast"], name="Forecast"))
        fig.update_layout(title="ARIMA Forecast - India Inflation (annual)", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")
else:
    st.info("No series available for ARIMA forecasting (India).")

# --------------------
# Monetary Policy Index (MPI) - FIXED (robust to None/NaN)
# --------------------
st.subheader("Monetary Policy Index (MPI)")
try:
    # helper to safely convert to float or return np.nan
    def safe_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    infl_latest = safe_num(wb_infl_in["Value"].astype(float).dropna().iloc[-3:].mean()) if not wb_infl_in.empty else np.nan

    rate_latest = np.nan
    if "repo_monthly" in locals() and not repo_monthly.empty:
        try:
            rate_latest = safe_num(repo_monthly.iloc[-1, 0])
        except Exception:
            rate_latest = np.nan
    elif not rbi_repo.empty:
        try:
            rate_latest = safe_num(rbi_repo.iloc[-1, 0])
        except Exception:
            rate_latest = np.nan

    if "liq_monthly" in locals() and not liq_monthly.empty:
        try:
            liq_latest = safe_num(liq_monthly["M3_IN"].dropna().iloc[-1])
        except Exception:
            liq_latest = np.nan
    elif not m3_in.empty:
        try:
            liq_latest = safe_num(m3_in["Value"].dropna().iloc[-1])
        except Exception:
            liq_latest = np.nan
    else:
        liq_latest = np.nan

    vol_latest = safe_num(vol.mean().mean()) if "vol" in locals() and not vol.empty else np.nan

    # make numeric array and replace nan with 0 for combination, but track missing count
    raw_vals = np.array([infl_latest, rate_latest, liq_latest, vol_latest], dtype="float64")
    missing = np.isnan(raw_vals).sum()
    vals = np.nan_to_num(raw_vals, nan=0.0)

    weights = np.array([w_infl, w_rate, w_liq, w_vol], dtype="float64")
    if weights.sum() == 0:
        weights = np.array([0.4, 0.25, 0.2, 0.15])
    weights = weights / weights.sum()

    mpi = float(np.dot(vals, weights))

    if missing >= 3:
        st.warning("Too many MPI inputs are missing — MPI may be unreliable. Upload missing series (repo or M3) for a better MPI.")
    st.metric("Monetary Policy Index (MPI)", round(mpi, 4))
    st.write(f"Weights used — inflation: {weights[0]:.2f}, rate: {weights[1]:.2f}, liquidity: {weights[2]:.2f}, volatility: {weights[3]:.2f}")
except Exception as e:
    st.warning(f"Could not compute MPI: {e}")

# --------------------
# PDF Export (save charts to images then embed)
# --------------------
st.subheader("Export: Detailed PDF Report (with charts)")

@st.cache_data
def save_figures_for_pdf():
    out = []
    os.makedirs("tmp_imgs", exist_ok=True)
    try:
        if not wb_infl_in.empty:
            f = "tmp_imgs/india_infl.png"
            fig = px.line(wb_infl_in, x="Date", y="Value", title="India Inflation", template=PLOTLY_TEMPLATE)
            try:
                fig.write_image(f)
                out.append(f)
            except Exception:
                pass
        if not wb_infl_us.empty:
            f = "tmp_imgs/usa_infl.png"
            fig = px.line(wb_infl_us, x="Date", y="Value", title="USA Inflation", template=PLOTLY_TEMPLATE)
            try:
                fig.write_image(f)
                out.append(f)
            except Exception:
                pass
        if "repo_monthly" in locals() and not repo_monthly.empty:
            f = "tmp_imgs/repo.png"
            fig = px.line(repo_monthly.reset_index(), x="Date", y=repo_monthly.columns[0], title="Repo Rate", template=PLOTLY_TEMPLATE)
            try:
                fig.write_image(f)
                out.append(f)
            except Exception:
                pass
        if not fed_bs.empty:
            f = "tmp_imgs/fed.png"
            fig = px.line(fed_bs.reset_index(), x="Date", y="Value", title="Fed Balance Sheet", template=PLOTLY_TEMPLATE)
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
    imgs = save_figures_for_pdf()
    texts = [f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"MPI: {round(mpi,4) if 'mpi' in locals() else 'N/A'}"]
    pdf = generate_pdf_with_images(imgs, texts)
    with open(pdf, "rb") as f:
        st.download_button("Download PDF Report", f, file_name=pdf)

st.caption("End of auto-data dashboard. If any auto-fetch failed, consider uploading CSVs or check network/API keys.")
