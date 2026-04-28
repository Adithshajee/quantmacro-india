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
from src.utils.config import TARGET_SECTORS
from src.insights.engine import generate_insights
from src.insights.llm import explain_market_condition
from src.models.predictor import PricePredictor

# --- Page Config ---
st.set_page_config(page_title="BSE AI Analytics", layout="wide", page_icon="📈")

# Custom CSS for modern layout
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stAlert { border-radius: 8px; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .intro-text { font-size: 1.1em; color: #444; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 BSE AI Decision Support System")

st.markdown("""
<div class="intro-text">
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

# --- Refresh Logic ---
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

# --- Database helpers ---
@st.cache_data(ttl=300) # Cache DB queries for 5 mins
def load_data(query, params=()):
    try:
        conn = get_streamlit_connection()
        return pd.read_sql(query, conn, params=params)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# --- ML caching ---
@st.cache_resource
def get_trained_predictor(df):
    predictor = PricePredictor()
    success, test_results = predictor.train_and_evaluate(df)
    return predictor, success, test_results

# --- Sidebar ---
st.sidebar.header("⚙️ Control Panel")
if st.sidebar.button("🔄 Force Refresh Data"):
    with st.spinner("Fetching latest news & market data..."):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            run_all_pipelines()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to refresh: {e}")

st.sidebar.caption(f"Last Updated: {last_updated}")
st.sidebar.divider()

# --- Sector Mapping ---
sector_map = {
    "Banking": {"price": "BANKING_SECTOR", "news": "BSE_BANKEX"},
    "IT": {"price": "IT_SECTOR", "news": "BSE_IT"},
    "Energy": {"price": "ENERGY_SECTOR", "news": "BSE_ENERGY"},
    "Market (Sensex)": {"price": "BSE_SENSEX", "news": "BSE_SENSEX"},
}

sel = st.sidebar.selectbox("🎯 Select Market Sector", list(sector_map.keys()))
p_key = sector_map[sel]["price"]
n_key = sector_map[sel]["news"]

# Fetch Data
pdf = load_data(
    "SELECT date, close_price, daily_return_pct FROM bse_sector_prices WHERE sector_index = ? ORDER BY date",
    (p_key,),
)

ndf = load_data(
    """
    SELECT r.published_at, r.headline, r.sentiment, r.sentiment_score 
    FROM raw_news r 
    JOIN news_sector_mapping m ON r.id = m.news_id 
    WHERE m.sector_index = ? 
    ORDER BY r.published_at DESC
    LIMIT 100
    """,
    (n_key,),
)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Trends", "📰 Sentiment Analysis", "🤖 AI Insights", "🔮 Predictions"])

# Ensure data exists
if not pdf.empty:
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values('date')
    
    # Calculate Advanced Indicators
    pdf['MA50'] = pdf['close_price'].rolling(window=50).mean()
    pdf['MA200'] = pdf['close_price'].rolling(window=200).mean()
    pdf['Volatility_20d'] = pdf['daily_return_pct'].rolling(window=20).std() * np.sqrt(252)
    
    latest_price = pdf.iloc[-1]["close_price"]
    prev_price = pdf.iloc[-2]["close_price"] if len(pdf) > 1 else latest_price
    delta_pct = ((latest_price - prev_price) / prev_price) * 100
    current_vol = pdf.iloc[-1]['Volatility_20d']
else:
    latest_price = 0
    delta_pct = 0
    current_vol = 0

with tab1:
    st.header(f"📈 {sel} Market Overview")
    if not pdf.empty:
        c1, c2 = st.columns(2)
        c1.metric(f"{sel} Index", f"₹{latest_price:,.2f}", f"{delta_pct:.2f}%")
        c2.metric("20-Day Volatility", f"{current_vol:.2f}%" if pd.notnull(current_vol) else "N/A", "Annualized")
        
        st.subheader("Price Action & Moving Averages")
        chart_data = pdf.set_index("date")[['close_price', 'MA50', 'MA200']]
        st.line_chart(chart_data, use_container_width=True)
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
            sentiment = str(row["sentiment"]).lower()
            label = "🟢 Pos" if sentiment == "positive" else "🔴 Neg" if sentiment == "negative" else "⚪ Neu"
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
                headlines = ndf['headline'].tolist()
                insights = generate_insights(pdf, ndf)
                explanation = explain_market_condition(sel, insights, headlines)
                st.info(explanation)
                
                st.markdown("**Why this matters:**")
                st.caption("LLM summaries synthesize hundreds of data points into actionable intelligence, saving analysts hours of reading while rapidly surfacing hidden market drivers.")
        
        with c2:
            st.subheader("⚠️ Divergence Analysis")
            for ins in insights:
                st.markdown(ins)
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
            trend_label = "UP ⬆️" if pred_trend == 1 else "DOWN ⬇️"
            
            c1.metric("Predicted Next Day Trend", trend_label)
            c2.metric("Predicted Next Day Price", f"₹{pred_price:,.2f}")
            c3.metric("Model Confidence", f"{confidence:.1f}%")
            
            st.divider()
            
            # Layout for Charts
            ch1, ch2 = st.columns([2, 1])
            
            with ch1:
                st.subheader("🧪 Backtesting: AI Strategy vs Buy & Hold")
                if test_results is not None and not test_results.empty:
                    plot_df = test_results[['bnh_cumulative', 'strategy_cumulative']].copy()
                    plot_df.columns = ["Buy & Hold", "AI Strategy"]
                    st.line_chart(plot_df, use_container_width=True)
            
            with ch2:
                st.subheader("🧠 Feature Importance")
                feat_imp = predictor.trend_model.feature_importances_
                features = ['daily_return_pct', 'lag1_return', 'lag2_return', 'rolling_mean_return_5', 'volatility_5']
                imp_df = pd.DataFrame({"Feature": features, "Importance": feat_imp})
                
                chart = alt.Chart(imp_df).mark_bar().encode(
                    x=alt.X("Importance:Q", title="Importance Score"),
                    y=alt.Y("Feature:N", sort="-x", title="Feature"),
                    color=alt.Color("Importance:Q", scale=alt.Scale(scheme="blues"), legend=None)
                ).properties(height=300)
                
                st.altair_chart(chart, use_container_width=True)
                
        else:
            st.warning("Not enough historical data to train the model robustly (Need at least 20 days).")
    else:
        st.warning("No price data.")
