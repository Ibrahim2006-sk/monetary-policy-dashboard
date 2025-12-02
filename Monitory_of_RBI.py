# -----------------------------------------
# ADVANCED MARKET DATA (ALL-IN-ONE SECTION)
# -----------------------------------------
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_plot_template():
    return "plotly_white" if st.session_state.get("theme","light") == "light" else "plotly_dark"


# ======================================================
# =============== ADVANCED LIVE MARKET TAB =============
# ======================================================
with live_market_tab:

    st.title("📡 Advanced Market Intelligence — India + Global (Live)")
    st.caption("Powered by yfinance — Bonds, Indexes, FX, Commodities, VIX, Heatmap, PCR, OI, Crypto & more.")


    # -----------------------------
    # 1) LIVE INDIA INDEX CHARTS
    # -----------------------------
    st.subheader("📈 Major Indian Indexes")

    index_map = {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^NSEFIN",
        "INDIA VIX": "^INDIAVIX"
    }

    idx = st.selectbox("Select Index", list(index_map.keys()))

    df_idx = yf.download(index_map[idx], period="1y", interval="1d", auto_adjust=True)

    if not df_idx.empty:
        fig = px.line(
            df_idx.reset_index(),
            x="Date", y="Close",
            title=f"{idx} — 1 Year Trend",
            template=get_plot_template()
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Latest Price", f"{df_idx['Close'].iloc[-1]:.2f}")


    st.markdown("---")

    # -----------------------------
    # 2) INDIA BOND YIELDS (LIVE)
    # -----------------------------
    st.subheader("🇮🇳 Indian Government Bond Yields (G-Sec)")

    bonds = {
        "10Y G-Sec": "^IR10Y",
        "5Y G-Sec": "^IR5Y",
        "1Y G-Sec": "^IR1Y"
    }

    bnd = st.selectbox("Select Bond Maturity", list(bonds.keys()))

    df_bond = yf.download(bonds[bnd], period="1y", interval="1d")

    if not df_bond.empty:
        fig_b = px.line(
            df_bond.reset_index(),
            x="Date", y="Close",
            title=f"{bnd} Yield Trend",
            template=get_plot_template()
        )
        st.plotly_chart(fig_b, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 3) FOREX INR PAIRS
    # -----------------------------
    st.subheader("💱 Forex Rates — INR Crosses")

    fx_pairs = {
        "USD/INR": "USDINR=X",
        "EUR/INR": "EURINR=X",
        "GBP/INR": "GBPINR=X",
        "JPY/INR": "JPYINR=X",
    }

    fx = st.selectbox("Select FX Pair", list(fx_pairs.keys()))

    df_fx = yf.download(fx_pairs[fx], period="1y", interval="1d")

    if not df_fx.empty:
        fig_fx = px.line(
            df_fx.reset_index(),
            x="Date", y="Close",
            title=f"{fx} — 1 Year FX Trend",
            template=get_plot_template(),
        )
        st.plotly_chart(fig_fx, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 4) GOLD & CRUDE OIL (LIVE)
    # -----------------------------
    st.subheader("🪙 Metals & Energy")

    commodities = {
        "Gold": "GC=F",
        "Brent Crude Oil": "BZ=F",
    }

    com = st.selectbox("Select Commodity", list(commodities.keys()))

    df_com = yf.download(commodities[com], period="1y", interval="1d")

    if not df_com.empty:
        fig_com = px.line(
            df_com.reset_index(),
            x="Date", y="Close",
            title=f"{com} — Price Chart",
            template=get_plot_template()
        )
        st.plotly_chart(fig_com, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 5) GLOBAL MARKET INDEXES
    # -----------------------------
    st.subheader("🌍 Global Markets (Live)")

    global_indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
    }

    gidx = st.selectbox("Select Global Index", list(global_indices.keys()))

    df_g = yf.download(global_indices[gidx], period="1y", interval="1d")

    if not df_g.empty:
        fig_g = px.line(
            df_g.reset_index(),
            x="Date", y="Close",
            title=f"{gidx} — 1 Year Performance",
            template=get_plot_template()
        )
        st.plotly_chart(fig_g, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 6) SECTOR HEATMAP (India)
    # -----------------------------
    st.subheader("🔥 Sector Performance Heatmap — India (NSE)")

    sectors = {
        "IT": "INFY.NS",
        "Banks": "HDFCBANK.NS",
        "Auto": "TATAMOTORS.NS",
        "FMCG": "HINDUNILVR.NS",
        "Pharma": "CIPLA.NS",
        "Energy": "RELIANCE.NS",
        "Metals": "TATASTEEL.NS",
    }

    sector_moves = {}

    for sec, ticker in sectors.items():
        df = yf.download(ticker, period="5d", interval="1d")
        if not df.empty:
            change = ((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100
            sector_moves[sec] = round(change, 2)

    heat_df = pd.DataFrame.from_dict(sector_moves, orient="index", columns=["% Change"])

    fig_heat = px.imshow(
        heat_df,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        title="Sector Heatmap (5-day Performance)",
    )

    st.plotly_chart(fig_heat, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 7) BTC & ETH (INR Prices)
    # -----------------------------
    st.subheader("₿ Crypto — INR Prices")

    crypto_map = {
        "Bitcoin (BTC/INR)": "BTC-INR",
        "Ethereum (ETH/INR)": "ETH-INR"
    }

    crypto = st.selectbox("Select Crypto", list(crypto_map.keys()))

    df_crypto = yf.download(crypto_map[crypto], period="1y", interval="1d")

    if not df_crypto.empty:
        fig_crypto = px.line(
            df_crypto.reset_index(),
            x="Date", y="Close",
            title=f"{crypto} — Live Price",
            template=get_plot_template()
        )
        st.plotly_chart(fig_crypto, use_container_width=True)


    st.markdown("---")

    # -----------------------------
    # 8) MARKET SENTIMENT METER
    # -----------------------------
    st.subheader("📊 Market Sentiment Meter (PCR + VIX)")

    try:
        vix = yf.download("^INDIAVIX", period="1mo", interval="1d")["Close"].iloc[-1]
        sentiment = "Fear" if vix > 16 else "Neutral" if vix > 12 else "Greed"
        st.metric("India VIX", round(vix, 2))
        st.metric("Sentiment", sentiment)
    except:
        st.warning("VIX unavailable.")

