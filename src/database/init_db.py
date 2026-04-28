import sqlite3
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH

logger = logging.getLogger(__name__)

def initialize_database(conn=None):
    """Creates tables if they do not exist."""
    close_conn = False
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    try:
        cursor = conn.cursor()
        
        # Prices table
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

        # News table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT,
            source TEXT,
            headline TEXT,
            content TEXT,
            url TEXT UNIQUE,
            is_macro BOOLEAN,
            sentiment TEXT,
            sentiment_score REAL
        )
        """)

        # Mapping table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_sector_mapping (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER,
            sector_index TEXT,
            mapping_reason TEXT,
            FOREIGN KEY(news_id) REFERENCES raw_news(id),
            UNIQUE(news_id, sector_index)
        )
        """)

        conn.commit()
    except Exception as e:
        logger.error(f"Error initializing database schema: {e}")
    finally:
        if close_conn:
            conn.close()

if __name__ == "__main__":
    initialize_database()
