import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import sys
import torch

# Standardize paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import MODEL_NAME
from src.utils.logger import get_logger
from src.database.connection import get_connection

logger = get_logger("sentiment_analyzer")

def run_sentiment_analysis():
    # Use Railway Volume or local 'models' folder for caching weights
    cache_folder = os.getenv("TRANSFORMERS_CACHE", "./models")
    
    # Check Hardware
    device_id = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device_id == 0 else "CPU"
    
    logger.info(f"Loading {MODEL_NAME} on {device_name} (Cache: {cache_folder})...")

    # Load Model with caching logic
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_folder)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=cache_folder)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device_id)

    conn = get_connection()
    news_df = pd.read_sql("SELECT id, headline FROM raw_news WHERE sentiment IS NULL", conn)

    if news_df.empty:
        logger.info("Database up to date. No new headlines to score.")
        conn.close()
        return

    logger.info(f"Batch processing {len(news_df)} headlines...")

    headlines = news_df["headline"].fillna("No headline").tolist()
    batch_size = 16
    results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        batch_preds = nlp(batch)
        
        for j, pred in enumerate(batch_preds):
            results.append({
                "id": int(news_df.iloc[i + j]["id"]),
                "sentiment": pred["label"],
                "score": float(pred["score"])
            })
        
        percent = int(((i + len(batch)) / len(headlines)) * 100)
        print(f"PROGRESS: {percent}%", flush=True)

    # Atomic Updates
    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE raw_news SET sentiment = ?, sentiment_score = ? WHERE id = ?",
        [(r["sentiment"], r["score"], r["id"]) for r in results]
    )
    
    conn.commit()
    conn.close()
    logger.info("Pipeline Complete: Database updated.")

if __name__ == "__main__":
    run_sentiment_analysis()