import sqlite3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_streamlit_connection():
    import streamlit as st
    @st.cache_resource
    def _create_persistent_connection():
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    return _create_persistent_connection()

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bse_sector_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        sector_index TEXT,
        close_price REAL,
        daily_return_pct REAL,
        UNIQUE(date, sector_index)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        published_at TEXT,
        source TEXT,
        headline TEXT,
        url TEXT UNIQUE,
        is_macro BOOLEAN,
        sentiment TEXT,
        sentiment_score REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_sector_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_id INTEGER NOT NULL,
        sector_index TEXT NOT NULL,
        mapping_reason TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(news_id, sector_index),
        FOREIGN KEY(news_id) REFERENCES raw_news(id)
    )
    """)

    conn.commit()
    conn.close()

# Auto-initialize on import wrapper
init_db()
