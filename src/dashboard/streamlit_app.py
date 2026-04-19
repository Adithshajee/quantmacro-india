import streamlit as st
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data/project.db")


def load_data(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn, params=params)


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

# --- Sidebar Selection ---
st.sidebar.header("Control Panel")
sel = st.sidebar.selectbox("Select Sector", list(sector_map.keys()))
p_key = sector_map[sel]["price"]
n_key = sector_map[sel]["news"]

# --- Layout Columns ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(f"📈 {sel} Price Trend")
    pdf = load_data(
        "SELECT date, close_price FROM bse_sector_prices WHERE sector_index = ? ORDER BY date",
        (p_key,),
    )

    if not pdf.empty:
        pdf["date"] = pd.to_datetime(pdf["date"])
        st.line_chart(pdf.set_index("date"))

        # Quick Metric Calculation
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

with col2:
    st.subheader(f"📰 {sel} AI Sentiment Feed")

    # Updated Query to include Sentiment columns
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
        for index, row in ndf.iterrows():
            # Logic for color-coded sentiment labels
            sentiment = str(row["sentiment"]).lower()
            if sentiment == "positive":
                label = "🟢 Positive"
            elif sentiment == "negative":
                label = "🔴 Negative"
            else:
                label = "⚪ Neutral"

            # Display news card
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
