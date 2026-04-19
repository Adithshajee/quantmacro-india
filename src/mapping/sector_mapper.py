import sqlite3
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger("sector_mapper")

KEYWORD_DICT = {
    "BSE_BANKEX": [r"\bbank(s|ing)?\b", r"\bfinancial\b", r"\binterest rate(s)?\b", r"\brbi\b", r"\bloan(s)?\b", r"\bcredit\b", r"\bmortgage\b", r"\bfinance\b"],
    "BSE_IT": [r"\btechnology\b", r"\bsoftware\b", r"\bai\b", r"\bcyber\b", r"\btech\b", r"\bit\b", r"\binfosys\b", r"\btcs\b", r"\bwipro\b", r"\bcloud\b"],
    "BSE_ENERGY": [r"\benergy\b", r"\boil\b", r"\bgas\b", r"\bpetroleum\b", r"\brenewable(s)?\b", r"\bcoal\b", r"\bpower\b", r"\bsolar\b", r"\bfuel\b"]
}

def map_sectors():
    logger.info("Starting sector mapping...")
    from src.database.connection import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    # Get all news that haven't been mapped recently, or just get all news
    # To keep it idempotency, we can just fetch all news and try to insert or ignore mappings
    cursor.execute("SELECT id, headline FROM raw_news")
    news_items = cursor.fetchall()

    if not news_items:
        logger.warning("No news to map.")
        conn.close()
        return

    inserted = 0

    for news_id, headline in news_items:
        lower_hl = headline.lower()
        for sector, patterns in KEYWORD_DICT.items():
            matched_keywords = []
            for pattern in patterns:
                if re.search(pattern, lower_hl):
                    # extract the matched word purely for reason
                    matched_word = re.search(pattern, lower_hl).group(0)
                    if matched_word not in matched_keywords:
                        matched_keywords.append(matched_word)
            
            if matched_keywords:
                reason = f"Matched keywords: {', '.join(matched_keywords)}"
                try:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO news_sector_mapping (news_id, sector_index, mapping_reason)
                        VALUES (?, ?, ?)
                        """,
                        (news_id, sector, reason)
                    )
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    logger.error(f"Error mapping news_id {news_id} to {sector}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Sector mapping complete. Added {inserted} new mappings.")

if __name__ == "__main__":
    map_sectors()
