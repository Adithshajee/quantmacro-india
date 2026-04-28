import requests
import sqlite3
import os
import sys
import time
from datetime import datetime
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH, NEWS_API_KEY, TARGET_SECTORS
from src.database.connection import get_connection
from src.utils.sentiment import SentimentAnalyzer

try:
    from src.utils.logger import get_logger
    logger = get_logger("news_fetcher")
except ImportError:
    import logging
    logger = logging.getLogger("news_fetcher")
    logger.setLevel(logging.INFO)

NEWS_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

def get_last_fetched_date():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date(published_at)) FROM raw_news")
        result = cursor.fetchone()
        conn.close()
        if result and result[0]:
            return result[0]
    except Exception as e:
        logger.error(f"Error fetching last fetched date: {e}")
    return None

def fetch_from_news_api(page: int, from_date: str = None):
    if not NEWS_API_KEY:
        return []
        
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
        response = requests.get(NEWS_URL, params=params, timeout=10)
        if response.status_code != 200:
            logger.error(f"NewsAPI ERROR: HTTP {response.status_code} - {response.text}")
            return []

        data = response.json()
        if data.get("status") != "ok":
            logger.error(f"Invalid API response: {data}")
            return []

        return data.get("articles", [])
    except Exception as e:
        logger.error(f"NewsAPI Fetch error on page {page}: {e}")
        return []

def fetch_from_google_news_rss():
    articles = []
    queries = ["India Economy", "Banking Sector India", "IT Sector India", "Energy Sector India"]
    
    for query in queries:
        try:
            params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
            response = requests.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('./channel/item')[:15]:  # Get top 15 per category
                    title = item.find('title').text if item.find('title') is not None else ""
                    link = item.find('link').text if item.find('link') is not None else ""
                    pubDate = item.find('pubDate').text if item.find('pubDate') is not None else ""
                    source = item.find('source').text if item.find('source') is not None else "Google News"
                    
                    try:
                        # Convert RFC822 to ISO 8601
                        dt = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %Z")
                        iso_date = dt.isoformat()
                    except:
                        iso_date = datetime.now().isoformat()
                        
                    articles.append({
                        "title": title,
                        "url": link,
                        "publishedAt": iso_date,
                        "source": {"name": source},
                        "content": title  # RSS often doesn't have full content
                    })
            time.sleep(1) # Be nice
        except Exception as e:
            logger.error(f"Google News RSS Fetch error for {query}: {e}")
            
    return articles

def normalize_article(article):
    try:
        if not all([article.get("publishedAt"), article.get("title"), article.get("url")]):
            return None
            
        source_name = article.get("source", {}).get("name") or "Unknown"
        content = article.get("content") or article.get("description") or ""
        
        return {
            "published_at": article["publishedAt"],
            "source": source_name,
            "headline": article["title"],
            "content": content,
            "url": article["url"],
            "is_macro": True
        }
    except Exception:
        return None

def save_news(articles, analyzer: SentimentAnalyzer):
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

        # Sentiment Analysis
        text_to_analyze = f"{clean['headline']} {clean['content']}"
        sentiment_label, sentiment_score = analyzer.analyze(text_to_analyze)

        try:
            cursor.execute(
                """
                INSERT INTO raw_news
                (published_at, source, headline, content, url, is_macro, sentiment, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean["published_at"],
                    clean["source"],
                    clean["headline"],
                    clean["content"],
                    clean["url"],
                    clean["is_macro"],
                    sentiment_label,
                    sentiment_score
                ),
            )
            
            news_id = cursor.lastrowid
            
            # Simple Sector Mapping
            headline_lower = clean["headline"].lower()
            if "bank" in headline_lower or "finance" in headline_lower or "rbi" in headline_lower:
                sector = "BSE_BANKEX"
            elif "tech" in headline_lower or "it " in headline_lower or "software" in headline_lower:
                sector = "BSE_IT"
            elif "energy" in headline_lower or "oil" in headline_lower or "power" in headline_lower:
                sector = "BSE_ENERGY"
            else:
                sector = "BSE_SENSEX" # Default to general market

            cursor.execute(
                """
                INSERT INTO news_sector_mapping
                (news_id, sector_index, mapping_reason)
                VALUES (?, ?, ?)
                """,
                (news_id, sector, "Keyword match in headline")
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # URL already exists
            skipped += 1
        except Exception as e:
            logger.error(f"Insert error: {e}")
            failed += 1

    conn.commit()
    conn.close()
    return inserted, skipped, failed

def run_ingestion():
    start_time = datetime.now()
    logger.info(f"Starting news ingestion at {start_time}")
    
    analyzer = SentimentAnalyzer()

    total_inserted = 0
    total_skipped = 0
    total_failed = 0

    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = get_last_fetched_date() or "2024-01-01"

    # Try NewsAPI first
    if NEWS_API_KEY:
        max_pages = 2  # Keep it small to save API credits and run fast
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page}/{max_pages} from NewsAPI...")
            articles = fetch_from_news_api(page, from_date)

            if not articles:
                break

            inserted, skipped, failed = save_news(articles, analyzer)
            total_inserted += inserted
            total_skipped += skipped
            total_failed += failed
            time.sleep(1)
            
    # Fallback / Supplement with Google News RSS
    logger.info("Fetching from Google News RSS...")
    rss_articles = fetch_from_google_news_rss()
    if rss_articles:
        inserted, skipped, failed = save_news(rss_articles, analyzer)
        total_inserted += inserted
        total_skipped += skipped
        total_failed += failed

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("----- FINAL INGESTION SUMMARY -----")
    logger.info(f"Total Inserted: {total_inserted}")
    logger.info(f"Total Skipped: {total_skipped}")
    logger.info(f"Total Failed: {total_failed}")
    logger.info(f"Duration: {duration}")
    
    return total_inserted

if __name__ == "__main__":
    run_ingestion()
