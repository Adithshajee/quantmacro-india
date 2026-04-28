import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import altair as alt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.connection import get_streamlit_connection
from src.ingestion.news_fetcher import run_ingestion
from src.ingestion.fetch_bse_data import main as fetch_bse_main
from src.insights.engine import generate_insights
from src.insights.llm import explain_market_condition
from src.models.predictor import PricePredictor

# --- Page Config ---
st.set_page_config(page_title="BSE AI Analytics", layout="wide", page_icon="📈")

st.markdown("""
<style>
    /* Force dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Modern Dark Cards for Metrics */
    div[data-testid="stMetric"] {
        background-color: #1c1f26;
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #2d3139;
        box-shadow: none;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #a0a6b5 !important;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #ffffff !important;
    }
    
    .stAlert {
        background-color: #1c1f26 !important;
        border: 1px solid #2d3139 !important;
        color: #e0e6ed !important;
        border-radius: 10px;
    }
    
    h1, h2, h3 { 
        font-family: 'Inter', sans-serif; 
        color: #ffffff;
    }
    
    hr {
        border-color: #2d3139;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 BSE AI Decision Support System")

st.markdown("""
<div style="font-size: 1.1em; color: #a0a6b5; margin-bottom: 20px;">
<b>📌 What this system does:</b><br>
This AI-powered financial platform analyzes the Indian equity market using quantitative and qualitative data. It provides:
<ul>
    <li><b>Market trend analysis:</b> Tracks indices with advanced volatility and moving average indicators.</li>
    <li><b>News sentiment impact:</b> Aggregates and scores real-time news to gauge market emotion.</li>
    <li><b>AI-driven predictions:</b> Uses Machine Learning (Random Forest) and LLMs to forecast next-day movements and explain market behavior.</li>
</ul>
</div>
<hr>
""", unsafe_allow_html=True)

# --- Unified Data Pipeline ---
@st.cache_data(ttl=86400)
def run_all_pipelines():
    try:
        fetch_bse_main() 
        run_ingestion()  
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(f"Error during auto-refresh: {e}")
        return "Failed"

last_updated = run_all_pipelines()

# --- Cached DB Loading ---
@st.cache_data(ttl=300)
def load_data(query, params=()):
    try:
        conn = get_streamlit_connection()
        return pd.read_sql(query, conn, params=params)
    except Exception as e:
        return pd.DataFrame()

# --- Cached ML Model ---
@st.cache_resource
def get_trained_predictor(df):
    predictor = PricePredictor()
    success, test_results = predictor.train_and_evaluate(df)
    return predictor, success, test_results

# --- Sidebar ---
st.sidebar.header("⚙️ Control Panel")
if st.sidebar.button("🔄 Force Refresh Data"):
    with st.spinner("Fetching latest news & market data..."):
        st.cache_data.clear()
        st.cache_resource.clear()
        run_all_pipelines()
        st.rerun()

st.sidebar.caption(f"Last Updated: {last_updated}")
st.sidebar.divider()

# --- Data Loading ---
sector_map = {
    "Banking": {"price": "BANKING_SECTOR", "news": "BSE_BANKEX"},
    "IT": {"price": "IT_SECTOR", "news": "BSE_IT"},
    "Energy": {"price": "ENERGY_SECTOR", "news": "BSE_ENERGY"},
    "Market (Sensex)": {"price": "BSE_SENSEX", "news": "BSE_SENSEX"},
}

sel = st.sidebar.selectbox("🎯 Select Market Sector", list(sector_map.keys()))
p_key = sector_map[sel]["price"]
n_key = sector_map[sel]["news"]

pdf = load_data("SELECT date, close_price, daily_return_pct FROM bse_sector_prices WHERE sector_index = ? ORDER BY date", (p_key,))
ndf = load_data("SELECT r.published_at, r.headline, r.sentiment, r.sentiment_score FROM raw_news r JOIN news_sector_mapping m ON r.id = m.news_id WHERE m.sector_index = ? ORDER BY r.published_at DESC LIMIT 100", (n_key,))

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Trends", "📰 Sentiment Analysis", "🤖 AI Insights", "🔮 Predictions"])

if not pdf.empty:
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values('date')
    pdf['MA50'] = pdf['close_price'].rolling(window=50).mean()
    pdf['MA200'] = pdf['close_price'].rolling(window=200).mean()
    pdf['Volatility_20d'] = pdf['daily_return_pct'].rolling(window=20).std() * np.sqrt(252)
    latest_price = pdf.iloc[-1]["close_price"]
    prev_price = pdf.iloc[-2]["close_price"] if len(pdf) > 1 else latest_price
    delta_pct = ((latest_price - prev_price) / prev_price) * 100
    current_vol = pdf.iloc[-1]['Volatility_20d']

with tab1:
    st.header(f"📈 {sel} Market Overview")
    if not pdf.empty:
        c1, c2 = st.columns(2)
        c1.metric(f"{sel} Index", f"₹{latest_price:,.2f}", f"{delta_pct:.2f}%")
        c2.metric("20-Day Volatility", f"{current_vol:.2f}%" if pd.notnull(current_vol) else "N/A", "Annualized")
        st.subheader("Price Action & Moving Averages")
        st.line_chart(pdf.set_index("date")[['close_price', 'MA50', 'MA200']])
    else:
        st.warning("No price data available.")

with tab2:
    st.header(f"📰 {sel} AI Sentiment Feed")
    if not ndf.empty:
        avg_sent = ndf.head(10)['sentiment_score'].mean()
        sent_delta = avg_sent - (ndf.tail(10)['sentiment_score'].mean() if len(ndf)>10 else 0)
        st.metric("Avg Sentiment Score (7d)", f"{avg_sent:.2f}", f"{sent_delta:.2f} shift")
        st.divider()
        for index, row in ndf.head(10).iterrows():
            label = "🟢 Pos" if row["sentiment"].lower() == "positive" else "🔴 Neg" if row["sentiment"].lower() == "negative" else "⚪ Neu"
            st.markdown(f"**{label}** | {row['headline']} _({str(row['published_at'])[:10]})_")
    else:
        st.info("No news found for this sector. Try refreshing the data.")

with tab3:
    st.header("🧠 AI Synthesized Insights")
    if not pdf.empty and not ndf.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("🤖 LLM Market Summary")
            with st.spinner("Generating deep explanation..."):
                insights = generate_insights(pdf, ndf)
                explanation = explain_market_condition(sel, insights, ndf['headline'].tolist())
                st.info(explanation)
        with c2:
            st.subheader("⚠️ Divergence Analysis")
            for ins in insights: st.markdown(ins)
    else:
        st.warning("Insufficient data to generate AI insights.")

with tab4:
    st.header("🔮 Next-Day Forecasting")
    if not pdf.empty:
        with st.spinner("Training Machine Learning Models..."):
            predictor, success, test_results = get_trained_predictor(pdf)
        if success and predictor.trained:
            pred_trend, pred_price, confidence = predictor.predict_next_day(pdf)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Next Day Trend", "UP ⬆️" if pred_trend == 1 else "DOWN ⬇️")
            c2.metric("Predicted Next Day Price", f"₹{pred_price:,.2f}")
            c3.metric("Model Confidence", f"{confidence:.1f}%")
            st.divider()
            
            ch1, ch2 = st.columns([2, 1])
            with ch1:
                st.subheader("🧪 Backtesting: AI Strategy vs Buy & Hold")
                plot_df = test_results[['bnh_cumulative', 'strategy_cumulative']].copy()
                plot_df.columns = ["Buy & Hold", "AI Strategy"]
                st.line_chart(plot_df)
            with ch2:
                st.subheader("🧠 Feature Importance")
                feat_imp = predictor.trend_model.feature_importances_
                features = ['daily_return_pct', 'lag1_return', 'lag2_return', 'rolling_mean_return_5', 'volatility_5']
                imp_df = pd.DataFrame({"Feature": features, "Importance": feat_imp})
                
                # Dark theme Altair chart
                chart = alt.Chart(imp_df).mark_bar().encode(
                    x=alt.X("Importance:Q", title="Importance Score", axis=alt.Axis(gridColor="#2d3139", titleColor="#a0a6b5", labelColor="#a0a6b5")),
                    y=alt.Y("Feature:N", sort="-x", title="Feature", axis=alt.Axis(titleColor="#a0a6b5", labelColor="#a0a6b5")),
                    color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues"), legend=None)
                ).properties(height=300, background="#0e1117").configure_view(strokeOpacity=0)
                
                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Not enough historical data to train the model robustly (Need at least 20 days).")
    else:
        st.warning("No price data.")
