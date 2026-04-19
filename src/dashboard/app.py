import streamlit as st
import pandas as pd
import sys
import os
import time
import subprocess
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.queries import get_latest_prices, get_latest_news_for_sector, get_market_pulse, get_top_impact_articles
from src.database.connection import get_streamlit_connection
from src.utils.config import API_HOST, TARGET_SECTORS

import socket
import atexit

st.set_page_config(page_title="BSE Sector & Macro News DB", layout="wide")

# Environment Validation
missing_deps = []
for lib in ["yfinance", "transformers", "torch"]:
    try:
        __import__(lib)
    except ImportError:
        missing_deps.append(lib)

if missing_deps:
    st.error(f"Missing required libraries: {', '.join(missing_deps)}. Please install them to continue.")
    st.stop()

# Auto-Launch API
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if 'api_launched' not in st.session_state:
    if not is_port_in_use(8000):
        # Launch API
        api_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        api_process = subprocess.Popen(["uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=api_cwd)
        st.session_state['api_launched'] = True

        def cleanup_api():
            api_process.terminate()
            api_process.wait()
        
        atexit.register(cleanup_api)
    else:
        st.session_state['api_launched'] = True

conn = get_streamlit_connection()

st.title("📈 BSE Macro-Sector Analyzer")

df_prices_init = get_latest_prices(conn)
if not df_prices_init.empty:
    latest_date_str = pd.to_datetime(df_prices_init['date']).max().strftime('%B %d, %Y')
    st.caption(f"**Data as of: {latest_date_str}**")
else:
    st.caption("**Data as of: N/A (Run Sync)**")

@st.cache_data(ttl=3600)
def cached_get_latest_prices():
    return get_latest_prices(conn)

@st.cache_data(ttl=3600)
def cached_get_latest_news_for_sector(sector, limit=20):
    return get_latest_news_for_sector(sector, limit, conn)

@st.cache_data(ttl=3600)
def cached_get_market_pulse():
    return get_market_pulse(conn)

@st.cache_data(ttl=3600)
def cached_get_top_impact_articles(limit=3):
    return get_top_impact_articles(limit, conn)

# Background orchestration
def run_pipeline():
    with st.status("Running One-Click Pipeline...", expanded=True) as status:
        pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils", "master_orchestrator.py"))
        process = subprocess.Popen(
            ["python", pipeline_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                if "PROGRESS:" in line:
                    status.update(label=f"Calculating AI Sentiment: {line.split('PROGRESS:')[1].strip()}", expanded=True)
                elif line.startswith("STEP:"):
                    status.update(label=f"Running: {line.split('STEP:')[1].strip()}", expanded=True)
                st.write(line)
                
        process.wait()
        if process.returncode != 0:
            status.update(label="Pipeline Failed", state="error", expanded=True)
            return

        status.update(label="Sync & Refresh Complete!", state="complete", expanded=False)

    st.cache_data.clear()
    time.sleep(0.5)
    st.rerun()

st.sidebar.header("Controls")
if st.sidebar.button("🔄 Sync & Refresh"):
    run_pipeline()

API_BASE = API_HOST.rstrip('/')

# Health Check logic
api_healthy = False
try:
    health_resp = requests.get(f"{API_BASE}/health", timeout=2)
    if health_resp.status_code == 200:
        api_healthy = True
except requests.exceptions.RequestException:
    pass

# Market Pulse
pulse = cached_get_market_pulse()
if pulse:
    st.header("🌐 Market Pulse")
    cols = st.columns(len(pulse))
    for col, (sector, score) in zip(cols, pulse.items()):
        formatted_score = f"{score:.2f}" if score is not None else "0.00"
        delta = score if score is not None else 0
        col.metric(label=sector, value=formatted_score, delta=f"{delta:.2f} Avg Sentiment")

st.divider()

# High-Impact Articles
st.header("⚡ Top 3 High-Impact Articles")
top_articles = cached_get_top_impact_articles(limit=3)
if not top_articles.empty:
    for idx, row in top_articles.iterrows():
        sentiment_color = "🟢" if row["sentiment_score"] > 0 else "🔴"
        st.markdown(f"### {sentiment_color} {row['headline']}")
        st.markdown(f"**Sector:** {row['sector_index']} | **Score:** {row['sentiment_score']:.2f} ({row['sentiment']})")
else:
    st.write("No configured articles found yet. Run Sync to capture AI sentiment.")

st.divider()

df_prices = cached_get_latest_prices()

st.sidebar.header("Filter by Sector")
sectors = [s for s in df_prices["sector_index"].unique() if s in TARGET_SECTORS] if not df_prices.empty else []
selected_sector = st.sidebar.selectbox("Select Sector", sectors)

if not df_prices.empty and selected_sector:
    sector_data = df_prices[df_prices["sector_index"] == selected_sector].copy()
    
    # Restrict scale to 6 months
    sector_data["date"] = pd.to_datetime(sector_data["date"])
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    sector_data = sector_data[sector_data["date"] >= six_months_ago]
    
    st.subheader(f"{selected_sector} Price Trend (Last 6 Months)")
    
    chart_data = sector_data.set_index("date")["close_price"]
    st.line_chart(chart_data)

    st.subheader(f"Recent Macro News for {selected_sector}")
    df_news = cached_get_latest_news_for_sector(selected_sector)
    
    if not df_news.empty:
        for idx, row in df_news.iterrows():
            with st.expander(f"{row['published_at'][:10]} - {row['headline']}"):
                color = "green" if row['sentiment'] == "positive" else "red" if row['sentiment'] == "negative" else "gray"
                st.markdown(f"**Score:** :{color}[{row['sentiment']} ({row['sentiment_score']:.2f})]")
                st.markdown(f"**Audit Trail:** {row['mapping_reason']}")
    else:
        st.write("No mapped news found for this sector.")
        
    st.subheader(f"🔮 Next-Day Outlook")
    
    if api_healthy:
        try:
            # Pre-compute lag and sentiment to send
            lag1 = float(sector_data['daily_return_pct'].iloc[-2]) if len(sector_data) > 1 else 0.0
            sentiment_val = pulse.get(selected_sector, 0.0) or 0.0
            
            resp = requests.post(f"{API_BASE}/predict", json={
                "lag1_return": lag1,
                "weighted_sentiment": sentiment_val,
                "sector_index": selected_sector
            }, timeout=5)
            
            if resp.status_code == 200:
                pred = resp.json()
                if pred["is_green"]:
                    st.success(f"**Outlook:** GREEN (Expected to rise) with {pred['probability']*100:.1f}% confidence")
                else:
                    red_conf = (1 - pred['probability'])*100
                    st.error(f"**Outlook:** RED (Expected to fall) with {red_conf:.1f}% confidence")
                
                st.subheader("🧠 Strategic Suggestions")
                sentiment_label = "positive" if sentiment_val > 0 else "negative" if sentiment_val < 0 else "neutral"
                if pred["is_green"] and pred["probability"] > 0.8:
                    advice = f"Strong positive outlook with {pred['probability']*100:.1f}% confidence. High {sentiment_label} sentiment suggests capitalizing on momentum in {selected_sector}."
                elif not pred["is_green"] and red_conf > 80:
                    advice = f"High {sentiment_label} sentiment in {selected_sector} suggests a defensive stance; consider reallocating to robust sectors as confidence of a drop exceeds 80%."
                elif pred["is_green"]:
                    advice = f"Moderate positive outlook. Monitor {selected_sector} closely while leveraging steady {sentiment_label} sentiment."
                else:
                    advice = f"Caution advised. Expected to drop with moderate confidence amidst {sentiment_label} sentiment."
                    
                st.info(f"**AI Advisor:** {advice}")
                
                if not pred["is_green"] and red_conf >= 99:
                    st.warning(f"**Pro Tip:** 🔴 Overwhelming 99%+ RED confidence detected. Strongly consider deeply defensive market moves, hedging {selected_sector} exposures, or maintaining heavy cash positions.")

            else:
                st.warning("Prediction API returned error.")
        except requests.exceptions.ConnectionError:
            st.error("Prediction API is unreachable.")
            st.info("💡 **Troubleshooting Guide:** To enable Next-Day Outlook predicting, please launch the FastAPI service in a separate terminal: `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`")
    else:
        st.error("Prediction API Unreachable.")
        st.info("💡 **Troubleshooting Guide:** To enable Next-Day Outlook predicting, please launch the FastAPI service in a separate terminal: `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`")
else:
    st.write("No price data available. Please run data ingestion pipelines via Sync.")
