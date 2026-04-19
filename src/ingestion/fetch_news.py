import requests
import sqlite3
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH, NEWS_API_KEY
from src.utils.logger import get_logger

logger = get_logger("fetch_news")
NEWS_URL = "https://newsapi.org/v2/everything"

def get_last_fetched_date():
    from src.database.connection import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date(published_at)) FROM raw_news")
    result = cursor.fetchone()
    conn.close()
    if result and result[0]:
        return result[0]
    return None

def fetch_news_page(page: int, from_date: str = None):
    params = {
        "q": "economy OR inflation OR interest rates OR GDP OR banking OR technology OR energy",
        "language": "en",
        "pageSize": 50,
        "page": page,
        "apiKey": NEWS_API_KEY,
    }
    if from_date:
        params["from"] = from_date
    params["to"] = datetime.now().strftime("%Y-%m-%d")

    try:
        response = requests.get(NEWS_URL, params=params)

        if response.status_code != 200:
            logger.error(f"API ERROR: HTTP {response.status_code} - {response.text}")
            return []

        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"Invalid API response: {data}")
            return []

        return data.get("articles", [])
    except Exception as e:
        logger.error(f"Fetch error on page {page}: {e}")
        return []

def normalize_article(article):
    try:
        # TODO: Implement robust source validation and deduplication
        # Require publishedAt, source.name, title, and url
        if not all([article.get("publishedAt"), article.get("source", {}).get("name"), article.get("title"), article.get("url")]):
            return None
        return (
            article["publishedAt"],
            article["source"]["name"],
            article["title"],
            article["url"],
            True,
        )
    except:
        return None

def save_news(articles):
    from src.database.connection import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    inserted = 0
    skipped = 0
    failed = 0

    for article in articles:
        clean = normalize_article(article)
        if not clean:
            failed += 1
            continue

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO raw_news
                (published_at, source, headline, url, is_macro)
                VALUES (?, ?, ?, ?, ?)
                """,
                clean,
            )

            if cursor.rowcount == 0:
                skipped += 1
            else:
                inserted += 1
        except Exception as e:
            logger.error(f"Insert error: {e}")
            failed += 1

    conn.commit()
    conn.close()
    return inserted, skipped, failed

def run_ingestion():
    start_time = datetime.now()
    logger.info(f"Starting news ingestion at {start_time}")

    total_inserted = 0
    total_skipped = 0
    total_failed = 0

    # Overcome April 10 stall: Fetch up to current date regardless of last entry
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = "2026-04-10" # Force past the stall
    logger.info(f"Force fetching from {from_date} to {to_date} to overcome stall.")

    max_pages = 10
    for page in range(1, max_pages + 1):
        logger.info(f"Fetching page {page}/{max_pages}...")
        articles = fetch_news_page(page, from_date)

        if not articles:
            logger.info("No more articles found or rate limited, stopping early.")
            break

        inserted, skipped, failed = save_news(articles)
        logger.info(f"Page {page} stats -> Inserted: {inserted}, Skipped: {skipped}, Failed: {failed}")

        total_inserted += inserted
        total_skipped += skipped
        total_failed += failed
        
        # Sleep slightly to avoid rapid rate limiting
        time.sleep(1)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("----- FINAL INGESTION SUMMARY -----")
    logger.info(f"Start Time: {start_time}")
    logger.info(f"End Time: {end_time} (Duration: {duration})")
    logger.info(f"Total Inserted: {total_inserted}")
    logger.info(f"Total Skipped: {total_skipped}")
    logger.info(f"Total Failed: {total_failed}")
    logger.info("News ingestion completed.")

if __name__ == "__main__":
    run_ingestion()
