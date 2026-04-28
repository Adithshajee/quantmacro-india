import streamlit as st
import sqlite3
import pandas as pd
import os
import sys

# --- Base Path Fix ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "project.db")

# --- Ensure data directory exists ---
os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# 🔥 HARD DB INITIALIZATION
# =========================
@st.cache_resource
def initialize_database():
    try:
        # Step 1: Force create DB file
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Step 2: Create tables if not exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bse_sector_prices (
            date TEXT,
            close_price REAL,
            sector_index TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY,
            published_at TEXT,
            headline TEXT,
            sentiment TEXT,
            sentiment_score REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_sector_mapping (
            news_id INTEGER,
            sector_index TEXT
        )
        """)

        conn.commit()

        # Step 3: Check if data exists
        cursor.execute("SELECT COUNT(*) FROM bse_sector_prices")
        count = cursor.fetchone()[0]

        # Step 4: If empty → run ingestion
        if count == 0:
            st.warning("⚠️ No data found. Fetching fresh data...")

            sys.path.append(BASE_DIR)
            from src.ingestion.fetch_bse_data import main

            main()

            st.success("✅ Data loaded successfully!")

        conn.close()

    except Exception as e:
        st.error(f"❌ DB Initialization failed: {e}")
        st.stop()


# =========================
# 📦 LOAD DATA
# =========================
def load_data(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


# 🔥 MUST RUN BEFORE UI
initialize_database()


# =========================
# 🎨 UI STARTS HERE
# =========================
st.set_page_config(page_title="BSE Macro-Sector Analyzer", layout="wide")
st.title("📊 BSE Macro-Sector Analyzer")

sector_map = {
    "Banking": {"price": "BANKING_SECTOR", "news": "BSE_BANKEX"},
    "IT": {"price": "IT_SECTOR", "news": "BSE_IT"},
    "Energy": {"price": "ENERGY_SECTOR", "news": "BSE_ENERGY"},
    "Market (Sensex)": {"price": "BSE_SENSEX", "news": "BSE_SENSEX"},
}

st.sidebar.header("Control Panel")
sel = st.sidebar.selectbox("Select Sector", list(sector_map.keys()))
p_key = sector_map[sel]["price"]
n_key = sector_map[sel]["news"]

col1, col2 = st.columns([3, 2])


# =========================
# 📈 PRICE
# =========================
with col1:
    st.subheader(f"📈 {sel} Price Trend")

    pdf = load_data(
        "SELECT date, close_price FROM bse_sector_prices WHERE sector_index = ? ORDER BY date",
        (p_key,),
    )

    if not pdf.empty:
        pdf["date"] = pd.to_datetime(pdf["date"])
        st.line_chart(pdf.set_index("date"))

        latest = pdf.iloc[-1]["close_price"]
        prev = pdf.iloc[-2]["close_price"] if len(pdf) > 1 else latest
        delta = ((latest - prev) / prev) * 100

        st.metric("Current Index", f"{latest:,.2f}", f"{delta:.2f}%")
    else:
        st.warning("No price data available.")


# =========================
# 📰 NEWS
# =========================
with col2:
    st.subheader(f"📰 {sel} AI Sentiment")

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

            label = (
                "🟢 Positive"
                if sentiment == "positive"
                else "🔴 Negative"
                if sentiment == "negative"
                else "⚪ Neutral"
            )

            st.markdown(f"**{row['headline']}**")
            st.caption(
                f"{row['published_at'][:10]} | {label} ({row['sentiment_score']:.2f})"
            )
            st.divider()
    else:
        st.info("No news data available.")
