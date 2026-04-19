import sqlite3
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH, MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger("sentiment_analyzer")

def run_sentiment_analysis():
    import torch
    
    device_id = -1
    device_name = "CPU"
    
    if torch.cuda.is_available():
        device_id = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"PROGRESS: 0% - Hardware Accelerated Model Loading on {device_name}", flush=True)
        logger.info(f"Using CUDA hardware accelerated device: {device_name}")
    else:
        logger.info("CUDA not available. Falling back to CPU mode.")

    logger.info(f"Loading FinBERT Model ({MODEL_NAME}) on device {device_id} ({device_name})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device_id)

    from src.database.connection import get_connection
    conn = get_connection()
    news_df = pd.read_sql("SELECT id, headline FROM raw_news WHERE sentiment IS NULL", conn)

    if news_df.empty:
        logger.info("No unscored news found in database.")
        conn.close()
        return

    logger.info(f"Found {len(news_df)} headlines to score. Starting batch analysis...")

    results = []
    
    # Process in batches
    headlines = [str(hl) if hl else "No headline" for hl in news_df["headline"]]
    predictions = []
    
    batch_size = 16
    total_batches = (len(headlines) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(headlines))
        batch = headlines[start_idx:end_idx]
        
        batch_preds = nlp(batch)
        predictions.extend(batch_preds)
        
        percent = int(((i + 1) / total_batches) * 100)
        print(f"PROGRESS: {percent}% - Scored {end_idx}/{len(headlines)} headlines", flush=True)

    for idx, pred in enumerate(predictions):
        results.append({
            "id": int(news_df.iloc[idx]["id"]),
            "sentiment": pred["label"],
            "sentiment_score": float(pred["score"])
        })


    # Update Database
    for res in results:
        conn.execute(
            "UPDATE raw_news SET sentiment = ?, sentiment_score = ? WHERE id = ?",
            (res["sentiment"], res["sentiment_score"], res["id"]),
        )
    
    conn.commit()
    conn.close()
    logger.info("AI Sentiment Analysis Complete! Database updated.")

if __name__ == "__main__":
    run_sentiment_analysis()
