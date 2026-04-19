import yfinance as yf
import pandas as pd
import sqlite3
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger("fetch_bse_data")

SECTOR_TICKERS = {
    "BSE_SENSEX": "^BSESN",
    "BSE_BANKEX": "^NSEBANK", # Assuming using NSE indices as proxy or mapping
    "BSE_IT": "^CNXIT",
    "BSE_ENERGY": "^CNXENERGY",
    "BANKING_SECTOR": "^NSEBANK",
    "IT_SECTOR": "^CNXIT",
    "ENERGY_SECTOR": "^CNXENERGY",
}

def fetch_and_process(sector, ticker):
    logger.info(f"Fetching data for {sector} ({ticker})")
    df = yf.download(ticker, period="2y", progress=False)

    if df.empty:
        logger.warning(f"No data found for {ticker}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        logger.warning(f"'Close' column not found for {ticker}")
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df["daily_return_pct"] = df["Close"].pct_change() * 100
    df = df.dropna()
    return df

def save_to_db(sector, df):
    if df.empty:
        logger.warning(f"Skipping {sector}: DataFrame is empty.")
        return

    from src.database.connection import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0

    for date, row in df.iterrows():
        try:
            price = float(row["Close"])
            ret = float(row["daily_return_pct"])
            d_str = date.strftime("%Y-%m-%d")

            cursor.execute(
                "INSERT OR REPLACE INTO bse_sector_prices (date, sector_index, close_price, daily_return_pct) VALUES (?, ?, ?, ?)",
                (d_str, sector, price, ret),
            )
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            logger.error(f"Error inserting row for {sector} on {date}: {e}")
            continue

    conn.commit()
    conn.close()
    logger.info(f"Saved {inserted} new rows for {sector}.")
    
def run_ingestion():
    # Only fetch for BSE_BANKEX, BSE_IT, BSE_ENERGY
    from src.utils.config import TARGET_SECTORS
    for sector in TARGET_SECTORS:
        ticker = SECTOR_TICKERS.get(sector)
        if not ticker:
            logger.warning(f"No ticker mapping found for {sector}")
            continue
        data = fetch_and_process(sector, ticker)
        save_to_db(sector, data)
    logger.info("BSE Sector Data ingestion complete.")

if __name__ == "__main__":
    run_ingestion()
