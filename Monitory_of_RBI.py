"""
rbi_yfinance_live_module.py

Plug-and-play module to add LIVE market data (yfinance) to your Streamlit RBI dashboard.
Includes three integration entrypoints (Option 1, 2, 3):

- render_live_market_tab():    --> call from a new tab in your tabs row (Option 1)
- render_live_inside_fx_tab(): --> call inside your existing FX & Commodities tab (Option 2)
- render_live_sidebar_panel(): --> render a compact sidebar panel or full page via sidebar (Option 3)

The module is self-contained and uses st.cache_data to rate-limit yfinance calls.
It fetches:
- Bond yields (10y, 5y, 1y) with fallback candidates
- Corporate bond proxies (ETF proxies)
- Indian indices (NIFTY, BANKNIFTY, FINNIFTY)
- FX pairs (USD/INR, EUR/INR, GBP/INR, JPY/INR)
- Commodities (Gold, Crude)

Notes:
- Install yfinance: `pip install yfinance` and `pip install pandas plotly` in your environment.
- yfinance relies on Yahoo; availability of some tickers may vary. The code uses fallbacks and graceful warnings.
- Use reasonable update cadence in Streamlit (e.g. st.experimental_memo or st.cache_data) to avoid excessive requests.

Usage examples (in your main app):

from rbi_yfinance_live_module import render_live_market_tab, render_live_inside_fx_tab, render_live_sidebar_panel

# Option 1 - add as a new tab (recommended):
# overview_tab, rbi_live_tab, live_market_tab, liq_repo_tab, ... = st.tabs([...])
# with live_market_tab:
#     render_live_market_tab()

# Option 2 - call inside the existing FX & Commodities tab:
# with fx_com_tab:
#     render_live_inside_fx_tab()

# Option 3 - draw a compact panel in sidebar:
# with st.sidebar:
#     render_live_sidebar_panel()

"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------
# CONFIG / SYMBOL MAPPINGS
# --------------------------
# Bond candidates: Yahoo ticker availability is inconsistent across countries.
BOND_CANDIDATES = {
    "IN_10Y": ["^IN10Y", "^NSEI10YR", "IN10Y.NS", "^TN10Y"],
    "IN_5Y": ["^IN05Y", "IN05Y.NS"],
    "IN_1Y": ["^IN01Y", "IN01Y.NS"],
}

# Corporate bond proxies: use liquid ETFs as proxies if specific bond tickers unavailable
CORP_BOND_PROXIES = {
    "AAA_Corp_Proxy": ["LIQUIDBEES.NS", "ICICIBANK.NS"],  # fallback: liquid money market ETF or major banks
}

# Indices
INDICES = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "^NSEFIN",
}

# FX
FX = {
    "USDINR": "INR=X",
    "EURINR": "EURINR=X",
    "GBPINR": "GBPINR=X",
    "JPYINR": "JPYINR=X",
}

# Commodities
COMMODITIES = {
    "Gold": ["GOLDMCX", "GC=F"],
    "Crude_Brent": ["BZ=F", "CL=F"],
}

# Plot template
PLOTLY_TEMPLATE = "plotly_white"

# --------------------------
# HELPERS
# --------------------------

def _try_fetch_ticker_history(ticker: str, period: str = "30d", interval: str = "1d") -> pd.DataFrame:
    """Fetch history for a single ticker; return empty DataFrame on failure."""
    try:
        t = yf.Ticker(ticker)
        # Use history -- if interval '1m' with period '1d' may be throttled in some envs
        df = t.history(period=period, interval=interval, actions=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Datetime": "Date"}, inplace=True)
        if "Date" not in df.columns and df.index.name in ["Date", "datetime"]:
            df = df.reset_index()
        # Ensure a Date column
        if "Date" not in df.columns:
            df["Date"] = df.index
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def probe_first_available(candidates: List[str], period: str = "30d", interval: str = "1d") -> Tuple[str, pd.DataFrame]:
    """Return first candidate ticker that yields a non-empty history and its dataframe."""
    for s in candidates:
        df = _try_fetch_ticker_history(s, period=period, interval=interval)
        if not df.empty:
            return s, df
    return "", pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_multiple_symbols(symbols: Dict[str, str], period: str = "30d", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch a dict of symbol -> dataframe using yfinance multi-ticker pull where possible."""
    out = {}
    # Try to fetch in batch for performance
    try:
        tickers = list(symbols.values())
        # Use yf.download for multi-symbols where possible
        raw = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        # If single ticker, structure is different
        if raw is None or raw.empty:
            # fallback: single fetches
            for name, tk in symbols.items():
                out[name] = _try_fetch_ticker_history(tk, period=period, interval=interval)
            return out

        # raw will have columns like (ticker, Open) when multiple
        if isinstance(raw.columns, pd.MultiIndex):
            for name, tk in symbols.items():
                if tk in raw.columns.levels[0]:
                    df = raw[tk].reset_index()
                    df.columns = ["Date"] + [f"{c}" for c in df.columns[1:]]
                    out[name] = df
                else:
                    out[name] = _try_fetch_ticker_history(tk, period=period, interval=interval)
        else:
            # single ticker returned
            # find which symbol it is
            first = list(symbols.keys())[0]
            out[first] = raw.reset_index()
            for name in list(symbols.keys())[1:]:
                out[name] = _try_fetch_ticker_history(symbols[name], period=period, interval=interval)
    except Exception:
        # best-effort fallback
        for name, tk in symbols.items():
            out[name] = _try_fetch_ticker_history(tk, period=period, interval=interval)
    return out


@st.cache_data(ttl=30)
def fetch_live_quote(ticker: str) -> Dict:
    """Fetch a small live summary (current price, change, pct) using Ticker.info or history tail."""
    try:
        t = yf.Ticker(ticker)
        # some tickers support fast info
        info = {}
        # Try history 1d 1m if allowed
        try:
            df = t.history(period="2d", interval="1m")
            if not df.empty:
                last = df.tail(1).iloc[0]
                prev = df.tail(2).iloc[0]
                price = float(last["Close"]) if "Close" in last else np.nan
                prev_price = float(prev["Close"]) if "Close" in prev else np.nan
                change = price - prev_price
                pct = (change / prev_price * 100) if prev_price not in [0, np.nan] else np.nan
                info["price"] = price
                info["change"] = change
                info["pct"] = pct
                info["time"] = df.index[-1].to_pydatetime()
                return info
        except Exception:
            pass

        # fallback: use fast_info or info
        try:
            finf = t.fast_info
            if finf and "last_price" in finf:
                info["price"] = finf["last_price"]
                info["time"] = datetime.now()
                return info
        except Exception:
            pass

        # last-resort: use history 7d 1d
        try:
            df2 = t.history(period="7d", interval="1d")
            if not df2.empty:
                last = df2.tail(1).iloc[0]
                price = float(last["Close"]) if "Close" in last else np.nan
                info["price"] = price
                info["time"] = df2.index[-1].to_pydatetime()
                return info
        except Exception:
            return {}
    except Exception:
        return {}


# --------------------------
# RENDERING / PLOTTING
# --------------------------

def _plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if df is None or df.empty:
        st.info("No data to plot.")
        return
    fig = px.line(df, x=x_col, y=y_col, title=title, template=PLOTLY_TEMPLATE)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# ENTRYPOINTS
# --------------------------

def render_live_market_tab(period_default: str = "90d", interval_default: str = "1d"):
    """Recommended: call this from a NEW tab (Option 1). Renders full page live market UI."""
    st.header("Live Market Data (yfinance)")

    # Controls
    colc1, colc2, colc3 = st.columns([1, 1, 1])
    with colc1:
        period = st.selectbox("Period (history)", options=["7d", "30d", "90d", "180d", "365d", "5y"], index=2)
    with colc2:
        interval = st.selectbox("Interval", options=["1m", "5m", "15m", "1h", "1d"], index=5 if "1d" in ["1m","5m","15m","1h","1d"] else 4)
    with colc3:
        refresh = st.button("Refresh Live Data")

    st.markdown("---")

    # 1) Bond Yields
    st.subheader("Bond Yields (India) — Best-effort from Yahoo Finance")
    bond_cols = st.columns(3)
    bond_dfs = {}
    for i, (k, candidates) in enumerate(BOND_CANDIDATES.items()):
        with bond_cols[i]:
            st.markdown(f"**{k.replace('_', ' ')}**")
            chosen, df = probe_first_available(candidates, period=period, interval=interval)
            if chosen:
                st.write(f"Ticker used: {chosen}")
                # If df has Open/Close columns, use Date and Close
                if "Close" in df.columns:
                    df_plot = df[["Date", "Close"]].rename(columns={"Close": k})
                else:
                    # try to find last numeric col
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    if numeric_cols:
                        df_plot = df[["Date", numeric_cols[-1]]].rename(columns={numeric_cols[-1]: k})
                    else:
                        df_plot = pd.DataFrame()
                if not df_plot.empty:
                    _plot_time_series(df_plot, "Date", k, f"{k} — {chosen}")
                bond_dfs[k] = df
            else:
                st.warning("No ticker available via yfinance for this maturity; consider using an alternate data source.")

    st.markdown("---")

    # 2) Corporate bond proxies
    st.subheader("Corporate Bond Proxies (ETFs) — best-effort")
    corp_cols = st.columns(len(CORP_BOND_PROXIES))
    corp_dfs = {}
    for i, (name, cand) in enumerate(CORP_BOND_PROXIES.items()):
        with corp_cols[i]:
            ticker_used, df = probe_first_available(cand, period=period, interval=interval)
            if ticker_used:
                st.write(f"{name}: {ticker_used}")
                if "Close" in df.columns:
                    df_plot = df[["Date", "Close"]].rename(columns={"Close": name})
                    _plot_time_series(df_plot, "Date", name, f"{name} ({ticker_used})")
                corp_dfs[name] = df
            else:
                st.info(f"No ETF proxy found for {name}.")

    st.markdown("---")

    # 3) Indices
    st.subheader("Indian Indices")
    idx_choice = st.multiselect("Select indices", options=list(INDICES.keys()), default=list(INDICES.keys()))
    idx_syms = {k: INDICES[k] for k in idx_choice}
    idx_dfs = fetch_multiple_symbols(idx_syms, period=period, interval=interval) if idx_syms else {}
    for name, df in idx_dfs.items():
        if not df.empty:
            # prefer Close column
            numeric_cols = [c for c in df.columns if c.lower().startswith("close") or c.lower().endswith("Close") or c.lower()=="close"]
            if numeric_cols:
                y = numeric_cols[0]
            else:
                y = df.select_dtypes(include=["number"]).columns.tolist()[0]
            _plot_time_series(df, "Date", y, f"{name} ({INDICES[name]})")
        else:
            st.warning(f"{name} data not available via yfinance.")

    st.markdown("---")

    # 4) FX
    st.subheader("FX (INR pairs)")
    fx_choice = st.multiselect("Select FX pairs", options=list(FX.keys()), default=list(FX.keys()))
    fx_syms = {k: FX[k] for k in fx_choice}
    fx_dfs = fetch_multiple_symbols(fx_syms, period=period, interval=interval) if fx_syms else {}
    for name, df in fx_dfs.items():
        if not df.empty:
            # use Close price
            col = "Close" if "Close" in df.columns else df.select_dtypes(include=["number"]).columns.tolist()[-1]
            _plot_time_series(df, "Date", col, f"{name} ({fx_syms[name]})")
        else:
            st.warning(f"FX {name} not available via yfinance.")

    st.markdown("---")

    # 5) Commodities
    st.subheader("Commodities")
    com_choice = st.multiselect("Select commodities", options=list(COMMODITIES.keys()), default=list(COMMODITIES.keys()))
    for cname in com_choice:
        candidates = COMMODITIES[cname]
        tk, df = probe_first_available(candidates, period=period, interval=interval)
        if tk:
            st.write(f"{cname}: {tk}")
            if "Close" in df.columns:
                df_plot = df[["Date", "Close"]].rename(columns={"Close": cname})
                _plot_time_series(df_plot, "Date", cname, f"{cname} ({tk})")
        else:
            st.warning(f"{cname} not available via yfinance.")

    st.markdown("---")

    # Quick live summary cards (latest quotes)
    st.subheader("Live Quotes — Snapshot")
    snapshot_cols = st.columns(4)
    snapshot_symbols = {}
    # pick some representative tickers
    # Indices first
    for name in [k for k in INDICES.keys()]:
        snapshot_symbols[name] = INDICES[name]
    # FX
    for name in [k for k in FX.keys()]:
        snapshot_symbols[name] = FX[name]
    # Commodities (pick first fallback)
    for name, cand in COMMODITIES.items():
        snapshot_symbols[name] = cand[0]

    i = 0
    for label, tk in snapshot_symbols.items():
        col = snapshot_cols[i % 4]
        with col:
            q = fetch_live_quote(tk)
            if q:
                price = q.get("price", None)
                pct = q.get("pct", None)
                st.metric(label, f"{price:.4f}" if price is not None else "N/A", f"{pct:.2f}%" if pct is not None else "")
            else:
                st.metric(label, "N/A", "")
        i += 1

    st.caption("Live data fetched via yfinance (Yahoo). Availability of some tickers may vary. Use refresh button to force re-fetch.")


def render_live_inside_fx_tab(period_default: str = "90d", interval_default: str = "1d"):
    """Call this from inside your existing FX & Commodities tab (Option 2). Renders a compact but full-featured view."""
    st.markdown("### Live Market Data (from yfinance)")
    # Reuse same UI but more compact; restrict intervals
    period = st.selectbox("Period", options=["7d", "30d", "90d"], index=2, key="fx_live_period")
    interval = st.selectbox("Interval", options=["1d", "1h"], index=0, key="fx_live_interval")

    # Show indices small
    st.markdown("#### Indices Snapshot")
    idx_syms = {k: INDICES[k] for k in INDICES.keys()}
    idx_dfs = fetch_multiple_symbols(idx_syms, period=period, interval=interval)
    cols = st.columns(len(idx_dfs))
    for i, (name, df) in enumerate(idx_dfs.items()):
        with cols[i]:
            if not df.empty:
                last = df.tail(1)
                val = None
                if "Close" in last.columns:
                    val = last["Close"].values[0]
                else:
                    numeric = last.select_dtypes(include=["number"]).columns.tolist()
                    if numeric:
                        val = last[numeric[-1]].values[0]
                st.metric(name, f"{val:.2f}" if val is not None else "N/A")
            else:
                st.write(name)
                st.write("N/A")

    st.markdown("---")
    # FX mini charts
    st.markdown("#### FX (INR pairs)")
    fx_syms = FX
    fx_dfs = fetch_multiple_symbols(fx_syms, period=period, interval=interval)
    for name, df in fx_dfs.items():
        if not df.empty:
            col = "Close" if "Close" in df.columns else df.select_dtypes(include=["number"]).columns.tolist()[-1]
            fig = px.line(df, x="Date", y=col, title=f"{name} ({FX[name]})", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

    st.caption("Compact live view (yfinance). For full interactivity, open the Live Market Data tab.")


def render_live_sidebar_panel(period_default: str = "30d", interval_default: str = "1d"):
    """Render a small set of live metrics in the sidebar (Option 3)."""
    st.markdown("### Live Market Snapshot")
    # show a couple of quick metrics
    try:
        usd = fetch_live_quote(FX["USDINR"]) or {}
        nifty = fetch_live_quote(INDICES["NIFTY"]) or {}
    except Exception:
        usd = {}
        nifty = {}

    if usd and "price" in usd:
        st.metric("USD/INR", f"{usd['price']:.4f}", f"{usd.get('pct',''):.2f}%" if usd.get("pct") is not None else "")
    else:
        st.write("USD/INR: N/A")

    if nifty and "price" in nifty:
        st.metric("NIFTY", f"{nifty['price']:.2f}")
    else:
        st.write("NIFTY: N/A")

    st.write("---")
    st.write("Open the Live Market Data tab for detailed charts and full history.")


# --------------------------
# END MODULE
# --------------------------
