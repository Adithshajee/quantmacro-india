import streamlit as st
import sqlite3
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# --- Load env ---
load_dotenv()

# --- Base Directory Fix (VERY IMPORTANT for cloud) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "project.db")


# --- Ensure data folder exists ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# --- AUTO DB INITIALIZATION ---
@st.cache_resource
def initialize_database():
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Database not found. Creating it now... (first run only)")

        try:
            # Add project root to Python path
            sys.path.append(BASE_DIR)

            # Import your ingestion script
            from src.ingestion.fetch_bse_data import main as fetch_data

            fetch_data()  # This must create DB + tables + insert data

            st.success("✅ Database created successfully!")

        except Exception as e:
            st.error(f"❌ Failed to initialize database: {e}")
            st.stop()


# --- LOAD DATA FUNCTION ---
def load_data(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn, params=params)


# --- Initialize DB before anything ---
initialize_database()


# --- Page Config ---
st.set_page_config(page_title="BSE Macro-Sector Analyzer", layout="wide")
st.title("📊 BSE Macro-Sector Analyzer")


# --- Sector Mapping ---
sector_map = {
    "Banking": {"price": "BANKING_SECTOR", "news": "BSE_BANKEX"},
    "IT": {"price": "IT_SECTOR", "news": "BSE_IT"},
    "Energy": {"price": "ENERGY_SECTOR", "news": "BSE_ENERGY"},
    "Market (Sensex)": {"price": "BSE_SENSEX", "news": "BSE_SENSEX"},
}


# --- Sidebar ---
st.sidebar.header("Control Panel")
sel = st.sidebar.selectbox("Select Sector", list(sector_map.keys()))
p_key = sector_map[sel]["price"]
n_key = sector_map[sel]["news"]


# --- Layout ---
col1, col2 = st.columns([3, 2])


# =========================
# 📈 PRICE TREND
# =========================
with col1:
    st.subheader(f"📈 {sel} Price Trend")

    pdf = load_data(
        """
        SELECT date, close_price 
        FROM bse_sector_prices 
        WHERE sector_index = ? 
        ORDER BY date
        """,
        (p_key,),
    )

    if not pdf.empty:
        pdf["date"] = pd.to_datetime(pdf["date"])

        st.line_chart(pdf.set_index("date"))

        # Metrics
        latest_price = pdf.iloc[-1]["close_price"]
        prev_price = pdf.iloc[-2]["close_price"] if len(pdf) > 1 else latest_price
        delta = ((latest_price - prev_price) / prev_price) * 100

        st.metric(
            label=f"Current {sel} Index",
            value=f"{latest_price:,.2f}",
            delta=f"{delta:.2f}%",
        )
    else:
        st.warning(f"Price data missing for {p_key}.")


# =========================
# 📰 NEWS SENTIMENT
# =========================
with col2:
    st.subheader(f"📰 {sel} AI Sentiment Feed")

    ndf = load_data(
        """
        SELECT r.published_at, r.headline, r.sentiment, r.sentiment_score 
        FROM raw_news r 
        JOIN news_sector_mapping m ON r.id = m.news_id 
        WHERE m.sector_index = ? 
        ORDER BY r.published_at DESC
        """,
        (n_key,),
    )

    if not ndf.empty:
        for _, row in ndf.iterrows():
            sentiment = str(row["sentiment"]).lower()

            if sentiment == "positive":
                label = "🟢 Positive"
            elif sentiment == "negative":
                label = "🔴 Negative"
            else:
                label = "⚪ Neutral"

            with st.container():
                st.markdown(f"**{row['headline']}**")

                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"📅 {row['published_at'][:10]}")
                with c2:
                    st.caption(f"AI: {label} ({row['sentiment_score']:.2f})")

                st.divider()
    else:
        st.info(f"No news found for {n_key}.")
