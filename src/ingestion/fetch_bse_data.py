import yfinance as yf
import pandas as pd
import sqlite3
import os
import sys
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.connection import get_connection

try:
    from src.utils.logger import get_logger
    logger = get_logger("fetch_bse_data")
except ImportError:
    logger = logging.getLogger("fetch_bse_data")
    logger.setLevel(logging.INFO)

# =========================
# 📊 SECTOR TICKERS
# =========================
SECTOR_TICKERS = {
    "BSE_SENSEX": "^BSESN",
    "BSE_BANKEX": "^NSEBANK",
    "BSE_IT": "^CNXIT",
    "BSE_ENERGY": "^CNXENERGY",
    "BANKING_SECTOR": "^NSEBANK",
    "IT_SECTOR": "^CNXIT",
    "ENERGY_SECTOR": "^CNXENERGY",
}

def fetch_and_process(sector, ticker, retries=3):
    logger.info(f"Fetching data for {sector} ({ticker})...")
    
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period="2y", progress=False)

            if df.empty:
                logger.warning(f"No data returned for {sector} on attempt {attempt+1}.")
                time.sleep(2)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Close"]].copy()
            df["daily_return_pct"] = df["Close"].pct_change() * 100
            df = df.dropna()

            return df
        except Exception as e:
            logger.error(f"yfinance error for {sector} on attempt {attempt+1}: {e}")
            time.sleep(2)
            
    logger.error(f"Failed to fetch data for {sector} after {retries} retries.")
    return pd.DataFrame()

def save_to_db(sector, df):
    if df.empty:
        return

    conn = get_connection()
    cursor = conn.cursor()

    inserted = 0
    for date, row in df.iterrows():
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO bse_sector_prices 
                (date, sector_index, close_price, daily_return_pct) 
                VALUES (?, ?, ?, ?)
                """,
                (
                    date.strftime("%Y-%m-%d"),
                    sector,
                    float(row["Close"]),
                    float(row["daily_return_pct"]),
                ),
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Error inserting row for {sector} on {date}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Successfully saved {inserted} records for {sector}.")

def main():
    logger.info("Starting BSE Data Ingestion...")
    for sector, ticker in SECTOR_TICKERS.items():
        data = fetch_and_process(sector, ticker)
        save_to_db(sector, data)
        time.sleep(1) # Be nice to Yahoo Finance API
    logger.info("BSE Data Ingestion complete.")

if __name__ == "__main__":
    main()
