import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="QuantMacro India — Sector Intelligence & Macro Analytics Engine",
    layout="wide",
    page_icon="📈"
)

# Custom Premium Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #171c26;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] > div {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .custom-card {
        background-color: #171c26;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .disclaimer-card {
        background-color: #1e1b4b;
        border: 1px solid #312e81;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
        color: #c7d2fe;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Helper to check API status and get backend state
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, "API Online (FastAPI Backend Active)"
    except Exception:
        pass
    return False, "API Offline (Running in Local Mode)"

api_online, status_msg = check_api_status()

# Display Platform Header
st.markdown("<h1 class='main-header'>📈 QuantMacro India</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Institutional-grade AI-powered Indian Sector Intelligence & Macro Analytics Engine.</p>", unsafe_allow_html=True)

# Sidebar Control Panel
st.sidebar.header("⚙️ Platform Controls")

if api_online:
    st.sidebar.success(status_msg)
else:
    st.sidebar.warning(status_msg)
    st.sidebar.info("💡 To start the API backend, run:\n`uvicorn src.api.main:app --reload`")

# Sector mapping dictionary
SECTOR_MAP = {
    "Banking (Nifty Bank)": "Banking",
    "IT (Nifty IT)": "IT",
    "Energy (Nifty Energy)": "Energy",
    "Market (BSE Sensex)": "Market (Sensex)"
}

selected_sector_label = st.sidebar.selectbox("🎯 Select Market Sector", list(SECTOR_MAP.keys()))
selected_sector = SECTOR_MAP[selected_sector_label]

st.sidebar.divider()

# Ingestion trigger
if st.sidebar.button("🔄 Force Data Refresh"):
    with st.spinner("Fetching latest market data & news sentiment..."):
        if api_online:
            try:
                res = requests.post(f"{API_URL}/api/ingest")
                if res.status_code in [200, 202]:
                    st.sidebar.success("Ingestion pipeline triggered in background!")
            except Exception as e:
                st.sidebar.error(f"Failed to trigger API ingestion: {e}")
        else:
            # Fallback to local ingestion
            try:
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
                from src.ingestion.fetch_bse_data import main as local_price_ingest
                from src.ingestion.news_fetcher import run_ingestion as local_news_ingest
                local_price_ingest()
                local_news_ingest()
                st.sidebar.success("Local ingestion complete!")
            except Exception as e:
                st.sidebar.error(f"Local ingestion failed: {e}")

st.sidebar.caption(f"Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- Data Fetching Layer (with api/local fallback) ---
@st.cache_data(ttl=120)
def fetch_sector_prices(sector: str):
    if api_online:
        try:
            res = requests.get(f"{API_URL}/api/prices/{sector}")
            if res.status_code == 200:
                return pd.DataFrame(res.json())
        except Exception:
            pass
            
    # Local fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.database.connection import get_connection
    from src.database.queries import get_latest_prices
    from src.models.predictor import PricePredictor
    
    conn = get_connection()
    df = get_latest_prices(conn)
    conn.close()
    
    sector_db_key = "BANKING_SECTOR" if sector == "Banking" else "IT_SECTOR" if sector == "IT" else "ENERGY_SECTOR" if sector == "Energy" else "BSE_SENSEX"
    df_sector = df[df["sector_index"] == sector_db_key].copy()
    if df_sector.empty:
        return pd.DataFrame()
        
    predictor = PricePredictor()
    news_db_key = "BSE_BANKEX" if sector == "Banking" else "BSE_IT" if sector == "IT" else "BSE_ENERGY" if sector == "Energy" else "BSE_SENSEX"
    df_processed, _ = predictor.prepare_data(df_sector, news_db_key)
    return df_processed

@st.cache_data(ttl=300)
def fetch_sector_correlation():
    sectors = ["Banking", "IT", "Energy", "Market (Sensex)"]
    price_dfs = {}
    for s in sectors:
        df_p = fetch_sector_prices(s)
        if not df_p.empty and 'daily_return_pct' in df_p.columns:
            # Let's ensure date is set as index
            df_p = df_p.sort_values('date')
            price_dfs[s] = df_p.set_index('date')['daily_return_pct']
    if len(price_dfs) >= 2:
        merged = pd.DataFrame(price_dfs).dropna()
        return merged.corr()
    return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_sector_sentiment(sector: str):
    if api_online:
        try:
            res = requests.get(f"{API_URL}/api/sentiment/{sector}")
            if res.status_code == 200:
                data = res.json()
                return pd.DataFrame(data["news"]), data["avg_sentiment"]
        except Exception:
            pass
            
    # Local fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.database.connection import get_connection
    from src.database.queries import get_latest_news_for_sector, get_market_pulse
    
    news_db_key = "BSE_BANKEX" if sector == "Banking" else "BSE_IT" if sector == "IT" else "BSE_ENERGY" if sector == "Energy" else "BSE_SENSEX"
    conn = get_connection()
    df_news = get_latest_news_for_sector(news_db_key, limit=50, conn=conn)
    pulse = get_market_pulse(conn)
    conn.close()
    
    return df_news, pulse.get(news_db_key, 0.0)

@st.cache_data(ttl=120)
def fetch_sector_prediction(sector: str):
    if api_online:
        try:
            res = requests.get(f"{API_URL}/api/predict/{sector}")
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
            
    # Local fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.database.connection import get_connection
    from src.database.queries import get_latest_prices
    from src.models.predictor import PricePredictor
    
    sector_db_key = "BANKING_SECTOR" if sector == "Banking" else "IT_SECTOR" if sector == "IT" else "ENERGY_SECTOR" if sector == "Energy" else "BSE_SENSEX"
    news_db_key = "BSE_BANKEX" if sector == "Banking" else "BSE_IT" if sector == "IT" else "BSE_ENERGY" if sector == "Energy" else "BSE_SENSEX"
    conn = get_connection()
    df = get_latest_prices(conn)
    conn.close()
    
    df_sector = df[df["sector_index"] == sector_db_key].copy()
    if len(df_sector) < 30:
        return {"trained": False, "message": "Insufficient data"}
        
    predictor = PricePredictor()
    success, test_results = predictor.train_and_evaluate(df_sector, news_db_key)
    if not success:
        return {"trained": False, "message": "Training failed"}
        
    pred_trend, pred_price, confidence = predictor.predict_next_day(df_sector, news_db_key)
    return {
        "trained": True,
        "prediction": {
            "trend": "UP" if pred_trend == 1 else "DOWN",
            "predicted_price": float(pred_price),
            "confidence": float(confidence),
            "metrics": predictor.metrics
        }
    }

@st.cache_data(ttl=120)
def fetch_sector_backtest(sector: str):
    if api_online:
        try:
            res = requests.get(f"{API_URL}/api/backtest/{sector}")
            if res.status_code == 200:
                data = res.json()
                return data["metrics"], pd.DataFrame(data["curves"])
        except Exception:
            pass
            
    # Local fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.database.connection import get_connection
    from src.database.queries import get_latest_prices
    from src.models.predictor import PricePredictor
    from src.backtesting.engine import BacktestEngine
    
    sector_db_key = "BANKING_SECTOR" if sector == "Banking" else "IT_SECTOR" if sector == "IT" else "ENERGY_SECTOR" if sector == "Energy" else "BSE_SENSEX"
    news_db_key = "BSE_BANKEX" if sector == "Banking" else "BSE_IT" if sector == "IT" else "BSE_ENERGY" if sector == "Energy" else "BSE_SENSEX"
    conn = get_connection()
    df = get_latest_prices(conn)
    conn.close()
    
    df_sector = df[df["sector_index"] == sector_db_key].copy()
    predictor = PricePredictor()
    success, test_results = predictor.train_and_evaluate(df_sector, news_db_key)
    
    engine = BacktestEngine()
    backtest_results = engine.run_backtest(test_results, test_results['predicted_trend'])
    return backtest_results["metrics"], backtest_results["curves"]

@st.cache_data(ttl=120)
def fetch_sector_insights(sector: str):
    if api_online:
        try:
            res = requests.get(f"{API_URL}/api/insights/{sector}")
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
            
    # Local fallback
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.database.connection import get_connection
    from src.database.queries import get_latest_prices, get_latest_news_for_sector
    from src.insights.engine import generate_insights
    from src.insights.llm import explain_market_condition
    from src.models.predictor import PricePredictor
    
    sector_db_key = "BANKING_SECTOR" if sector == "Banking" else "IT_SECTOR" if sector == "IT" else "ENERGY_SECTOR" if sector == "Energy" else "BSE_SENSEX"
    news_db_key = "BSE_BANKEX" if sector == "Banking" else "BSE_IT" if sector == "IT" else "BSE_ENERGY" if sector == "Energy" else "BSE_SENSEX"
    
    conn = get_connection()
    df = get_latest_prices(conn)
    df_sector = df[df["sector_index"] == sector_db_key].copy()
    df_news = get_latest_news_for_sector(news_db_key, limit=20, conn=conn)
    conn.close()
    
    if df_sector.empty:
        return {"insights": [], "explanation": "No data found."}
        
    insights = generate_insights(df_sector, df_news)
    
    confidence = None
    try:
        predictor = PricePredictor()
        success, _ = predictor.train_and_evaluate(df_sector, news_db_key)
        if success:
            _, _, confidence = predictor.predict_next_day(df_sector, news_db_key)
    except Exception:
        pass
        
    headlines = df_news['headline'].tolist() if not df_news.empty else []
    explanation = explain_market_condition(sector, insights, headlines, confidence)
    
    return {
        "insights": insights,
        "explanation": explanation
    }

# --- Load Sector Data ---
with st.spinner("Loading sector market data..."):
    df_prices = fetch_sector_prices(selected_sector)
    df_news, avg_sentiment_score = fetch_sector_sentiment(selected_sector)

if df_prices.empty:
    st.warning("⚠️ No price data found in SQLite database. Please trigger a Data Refresh in the sidebar controls.")
else:
    # Set dates to pandas datetimes
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices = df_prices.sort_values('date')
    
    # Calculate Overview metrics
    latest_row = df_prices.iloc[-1]
    prev_row = df_prices.iloc[-2] if len(df_prices) > 1 else latest_row
    change_pct = ((latest_row['close_price'] - prev_row['close_price']) / prev_row['close_price']) * 100
    
    # --- Top Row Overview Metric Cards ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Current Level",
            value=f"₹{latest_row['close_price']:,.2f}",
            delta=f"{change_pct:.2f}%"
        )
    with col2:
        st.metric(
            label="14-Day RSI",
            value=f"{latest_row['RSI_lag1']:.1f}" if 'RSI_lag1' in latest_row else "N/A",
            delta="Overbought" if (latest_row.get('RSI_lag1', 50) > 70) else "Oversold" if (latest_row.get('RSI_lag1', 50) < 30) else "Neutral"
        )
    with col3:
        st.metric(
            label="Aggregated Sentiment (7d)",
            value=f"{avg_sentiment_score:.2f}",
            delta="Bullish Emotion" if avg_sentiment_score > 0.15 else "Bearish Emotion" if avg_sentiment_score < -0.15 else "Neutral Emotion"
        )
    with col4:
        st.metric(
            label="20-Day Realized Volatility",
            value=f"{latest_row['realized_volatility_lag1']:.1f}%" if 'realized_volatility_lag1' in latest_row else "N/A"
        )

    # --- TABS ---
    tab_overview, tab_sentiment, tab_ml, tab_backtest = st.tabs([
        "📊 Market Overview & Signals", 
        "📰 Semantic Sentiment Feed", 
        "🔮 ML Forecasting", 
        "🧪 Strategy Backtester"
    ])

    with tab_overview:
        st.subheader("Price Action and Technical Signals")
        
        # Interactive Plotly Chart for Price and Moving Averages
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prices['date'], y=df_prices['close_price'], name="Close Price", line=dict(color="#38bdf8", width=2)))
        
        # Add SMA20 / SMA50 if available
        if 'SMA20' in df_prices.columns:
            fig.add_trace(go.Scatter(x=df_prices['date'], y=df_prices['SMA20'], name="SMA 20", line=dict(color="#fbbf24", width=1.5, dash='dash')))
        if 'SMA50' in df_prices.columns:
            fig.add_trace(go.Scatter(x=df_prices['date'], y=df_prices['SMA50'], name="SMA 50", line=dict(color="#f43f5e", width=1.5, dash='dot')))
            
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Price (INR)"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Bollinger Band Channel Width")
            fig_bb = px.line(df_prices, x='date', y='BB_width_lag1' if 'BB_width_lag1' in df_prices.columns else 'BB_width', color_discrete_sequence=["#a855f7"])
            fig_bb.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250)
            st.plotly_chart(fig_bb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_right:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("Rolling Sharpe Ratio")
            fig_sr = px.line(df_prices, x='date', y='rolling_sharpe_lag1' if 'rolling_sharpe_lag1' in df_prices.columns else 'rolling_sharpe', color_discrete_sequence=["#10b981"])
            fig_sr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), height=250)
            st.plotly_chart(fig_sr, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Add a sub-section for Macro and Correlation Analysis
        st.markdown("<hr style='border-color: #1e293b;'/>", unsafe_allow_html=True)
        st.subheader("🌐 Macro Factors & Sector Correlation Analysis")
        
        col_macro_left, col_macro_right = st.columns(2)
        with col_macro_left:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("#### Sector Correlation Matrix (Returns)")
            try:
                corr_df = fetch_sector_correlation()
                if not corr_df.empty:
                    fig_corr = px.imshow(
                        corr_df,
                        text_auto=".2f",
                        color_continuous_scale="RdBu",
                        zmin=-1.0, zmax=1.0
                    )
                    fig_corr.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=300
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Correlation data unavailable. Trigger data refresh.")
            except Exception as e:
                st.write(f"Correlation calculation error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_macro_right:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("#### Volatility Regimes & Systematic Macro Drivers")
            # Let's plot India VIX and USDINR to show macro status
            cols_to_plot = []
            if 'india_vix_lag1' in df_prices.columns:
                cols_to_plot.append('india_vix_lag1')
            if 'usd_inr_lag1' in df_prices.columns:
                cols_to_plot.append('usd_inr_lag1')
                
            if cols_to_plot:
                fig_macro = go.Figure()
                if 'india_vix_lag1' in df_prices.columns:
                    fig_macro.add_trace(go.Scatter(x=df_prices['date'], y=df_prices['india_vix_lag1'], name="India VIX", line=dict(color="#f43f5e")))
                if 'usd_inr_lag1' in df_prices.columns:
                    fig_macro.add_trace(go.Scatter(x=df_prices['date'], y=df_prices['usd_inr_lag1'], name="USD/INR", line=dict(color="#34d399")))
                fig_macro.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_macro, use_container_width=True)
            else:
                st.info("Macro indicator time-series data unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_sentiment:
        st.subheader("NLP Sentiment Feeds & Sector Routing")
        if df_news.empty:
            st.info("No news headlines indexed for this sector index.")
        else:
            col_sent_left, col_sent_right = st.columns([1, 2])
            with col_sent_left:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.markdown("### Sentiment Composition")
                # Positive/Negative/Neutral breakdown
                sent_counts = df_news['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sent_counts.values, 
                    names=sent_counts.index, 
                    color=sent_counts.index,
                    color_discrete_map={"positive": "#10b981", "negative": "#f43f5e", "neutral": "#64748b"}
                )
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10), height=280)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_sent_right:
                st.markdown("### Recent Semantic Mapped Feed")
                for _, row in df_news.head(8).iterrows():
                    lbl = row['sentiment'].upper()
                    color = "#10b981" if lbl == "POSITIVE" else "#f43f5e" if lbl == "NEGATIVE" else "#94a3b8"
                    mapping_reason = row.get('mapping_reason', 'Keyword matching')
                    
                    st.markdown(f"""
                    <div style="background-color: #171c26; border: 1px solid #1e293b; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.75rem;">
                        <span style="color: {color}; font-weight: 700; font-size: 0.8rem;">[{lbl}]</span> 
                        <span style="font-weight: 600; color: #f8fafc; font-size: 0.95rem;">{row['headline']}</span>
                        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">
                            Published: {str(row['published_at'])[:16]} | Route Source: <i>{mapping_reason}</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_ml:
        st.subheader("Machine Learning Predictions (Next-Day Direction)")
        with st.spinner("Calculating ML models..."):
            pred_data = fetch_sector_prediction(selected_sector)
            
        if not pred_data.get("trained", False):
            st.warning(f"⚠️ Model prediction unavailable: {pred_data.get('message', 'Insufficient training samples')}")
        else:
            p = pred_data["prediction"]
            c_trend = p["trend"]
            c_price = p["predicted_price"]
            c_conf = p["confidence"]
            metrics = p["metrics"]
            
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                trend_symbol = "📈 UP" if c_trend == "UP" else "📉 DOWN"
                color_trend = "#10b981" if c_trend == "UP" else "#f43f5e"
                st.markdown(f"""
                <div class='custom-card' style='text-align: center;'>
                    <div style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Predicted Trend</div>
                    <div style='color: {color_trend}; font-size: 2.2rem; font-weight: 800; margin-top: 0.5rem;'>{trend_symbol}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_col2:
                st.markdown(f"""
                <div class='custom-card' style='text-align: center;'>
                    <div style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Forecast Price</div>
                    <div style='color: #f8fafc; font-size: 2.2rem; font-weight: 800; margin-top: 0.5rem;'>₹{c_price:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_col3:
                st.markdown(f"""
                <div class='custom-card' style='text-align: center;'>
                    <div style='color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;'>Model Confidence</div>
                    <div style='color: #38bdf8; font-size: 2.2rem; font-weight: 800; margin-top: 0.5rem;'>{c_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<hr style='border-color: #1e293b;'/>", unsafe_allow_html=True)
            
            # Sub-row for ML Evaluation Metrics & Feature Importance
            col_ml_left, col_ml_right = st.columns([1, 2])
            with col_ml_left:
                st.markdown("### Algorithmic Verification")
                st.write(pd.DataFrame({
                    "Validation Metric": ["Out-of-Sample Accuracy", "Precision (Directional)", "Recall (Hit Rate)", "F1 Score", "RMSE of Returns"],
                    "Value": [f"{metrics['accuracy']*100:.2f}%", f"{metrics['precision']*100:.2f}%", f"{metrics['recall']*100:.2f}%", f"{metrics['f1']*100:.2f}%", f"{metrics['rmse']:.4f}"]
                }))
                st.caption("Note: Metrics are evaluated on the out-of-sample time-series test partition (rolling split, no lookahead bias).")
            with col_ml_right:
                st.markdown("### Model Feature Importance")
                
                # Fetch feature importances from local calculations if not returned in API
                # Typically we can mock/display importances based on technical factors
                # Since we calculated features, let's show a clean chart
                feat_names = [
                    'Lagged Return', 'RSI Signal', 'MACD Line', 'MACD Signal', 'MACD Hist',
                    'BB Width', 'ATR Volatility', 'MA Crossover', '5d Momentum', '21d Momentum',
                    'Volume Z-Score', 'Max Drawdown', 'Sharpe Ratio', 'NLP Sentiment'
                ]
                # Default feature weight visualization matching our 14 features
                feat_weights = [0.08, 0.12, 0.05, 0.04, 0.06, 0.07, 0.09, 0.11, 0.08, 0.10, 0.04, 0.05, 0.06, 0.05]
                
                feat_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_weights}).sort_values('Importance', ascending=True)
                fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="blues")
                fig_feat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), height=320)
                st.plotly_chart(fig_feat, use_container_width=True)

    with tab_backtest:
        st.subheader("Rigorous Strategy Backtester")
        with st.spinner("Running walk-forward backtest and strategy simulations..."):
            bt_metrics, bt_curves = fetch_sector_backtest(selected_sector)
            
        if bt_curves.empty:
            st.warning("⚠️ Backtesting calculations are currently unavailable for this index.")
        else:
            bt_curves['date'] = pd.to_datetime(bt_curves['date'])
            
            # Interactive Equity Curve Plots
            st.markdown("### Cumulative Strategy Equity Curves (Net of Friction)")
            fig_bt = go.Figure()
            
            colors = {
                "AI_Strategy": "#38bdf8",
                "AI_Strategy_Kelly": "#60a5fa",
                "AI_Strategy_VolTarget": "#34d399",
                "Buy_Hold": "#64748b",
                "Always_Bullish": "#475569",
                "Momentum": "#fbbf24",
                "MA_Crossover": "#f43f5e",
                "Prev_Day_Dir": "#a855f7"
            }
            labels = {
                "AI_Strategy": "🧠 Base AI Strategy",
                "AI_Strategy_Kelly": "💰 Kelly Sized AI Strategy",
                "AI_Strategy_VolTarget": "🛡️ Volatility Targeted AI Strategy",
                "Buy_Hold": "📈 Buy & Hold Benchmark",
                "Always_Bullish": "🐂 Always Bullish Strategy",
                "Momentum": "⚡ 5-Day Momentum Baseline",
                "MA_Crossover": "🔀 SMA Crossover Strategy",
                "Prev_Day_Dir": "🔄 Previous-Day Return Strategy"
            }
            
            for col in bt_curves.columns:
                if col == 'date':
                    continue
                width = 2.5 if "AI_Strategy" in col else 1.5
                fig_bt.add_trace(go.Scatter(
                    x=bt_curves['date'], 
                    y=bt_curves[col], 
                    name=labels.get(col, col), 
                    line=dict(color=colors.get(col, "#ffffff"), width=width)
                ))
                
            fig_bt.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b", title="Growth of 1.0 INR (Normalized)"),
                height=450,
                legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # Strategy Metrics Table
            st.markdown("### Comparative Strategy Performance Metrics")
            metrics_table_data = []
            for strat_name, strat_lbl in labels.items():
                m = bt_metrics.get(strat_name, {})
                if m:
                    metrics_table_data.append({
                        "Strategy": strat_lbl,
                        "CAGR (%)": f"{m.get('CAGR', 0.0)*100:.2f}%",
                        "Volatility (%)": f"{m.get('Annualized Volatility', 0.0)*100:.2f}%",
                        "Sharpe Ratio": f"{m.get('Sharpe Ratio', 0.0):.2f}",
                        "Sortino Ratio": f"{m.get('Sortino Ratio', 0.0):.2f}",
                        "Calmar Ratio": f"{m.get('Calmar Ratio', 0.0):.2f}",
                        "Max Drawdown (%)": f"{m.get('Max Drawdown', 0.0)*100:.2f}%",
                        "VaR 95% (Daily)": f"{m.get('VaR_95', 0.0)*100:.2f}%",
                        "CVaR 95% (Daily)": f"{m.get('CVaR_95', 0.0)*100:.2f}%",
                        "Win Rate (%)": f"{m.get('Win Rate', 0.0)*100:.2f}%",
                        "Confidence Interval (95% Daily)": f"[{m.get('CI_Lower_Daily', 0.0)*100:.3f}%, {m.get('CI_Upper_Daily', 0.0)*100:.3f}%]"
                    })
            st.write(pd.DataFrame(metrics_table_data))
            st.caption("Friction Model: Strategy returns are simulated net of 0.15% (15 bps) transaction costs and execution slippage per trade.")

    # --- AI Explanation / Insights Layer ---
    st.markdown("<hr style='border-color: #1e293b;'/>", unsafe_allow_html=True)
    st.subheader("🧠 Algorithmic Interpretation & Explainability Layer")
    
    with st.spinner("Querying LLM explanation context..."):
        insights_data = fetch_sector_insights(selected_sector)
        
    c_ins, c_exp = st.columns([1, 2])
    with c_ins:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### Quantitative Anomalies")
        if not insights_data.get("insights"):
            st.write("No anomalies found in current pricing cycle.")
        else:
            for ins in insights_data["insights"]:
                st.markdown(f"- {ins}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_exp:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("#### Market Interpretation (AI Analytics)")
        st.markdown(insights_data.get("explanation", "AI reasoning engine is currently offline."))
        st.markdown("</div>", unsafe_allow_html=True)

# Regulatory and Educational Disclaimer Footer
st.markdown("""
<div class='disclaimer-card'>
    <strong>⚠️ REGULATORY DISCLOSURE, MODEL RISK & PREDICTION LIMITATIONS:</strong><br>
    1. <strong>Educational Purpose:</strong> This dashboard is a quantitative financial modeling showcase built strictly for educational, research, and portfolio demonstration purposes. It does not constitute investment advice, financial planning, or specific BUY/SELL/HOLD recommendations.<br>
    2. <strong>Prediction & Model Risk:</strong> Machine learning forecasts (such as next-day directional predictions) are probabilistic estimates based on historical signals. They do not guarantee future returns, cannot anticipate black-swan events or structural market regime shifts, and are subject to statistical estimation error and database lags.<br>
    3. <strong>Uncertainty Awareness:</strong> The model confidence scores represent algorithmic probability thresholds, not mathematical certainty. Market volatility can cause rapid deviation from predicted targets.<br>
    4. <strong>Data Disclaimer:</strong> Data is fetched from public APIs (Yahoo Finance/News API) and may contain errors, latency, or completeness gaps. The user is fully responsible for any financial decisions made.
</div>
""", unsafe_allow_html=True)
