import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger("train_model")

def sentiment_to_numeric(label):
    if label.lower() == "positive": return 1.0
    elif label.lower() == "negative": return -1.0
    return 0.0

def build_training_frame():
    from src.database.connection import get_connection
    conn = get_connection()
    
    # Needs to handle cases where there is no data
    prices_df = pd.read_sql("SELECT date, sector_index, daily_return_pct FROM bse_sector_prices ORDER BY date ASC", conn)
    
    query_news = """
    SELECT date(r.published_at) as date, m.sector_index, r.sentiment, r.sentiment_score
    FROM raw_news r
    JOIN news_sector_mapping m ON r.id = m.news_id
    WHERE r.sentiment IS NOT NULL
    """
    news_df = pd.read_sql(query_news, conn)
    conn.close()

    if prices_df.empty:
        logger.warning("No price data. Cannot build frame.")
        return pd.DataFrame()

    prices_df["date"] = pd.to_datetime(prices_df["date"])
    
    # We want to predict if the NEXT day will close Green.
    prices_df = prices_df.sort_values(by=["sector_index", "date"])
    prices_df["target_next_day_green"] = (prices_df.groupby("sector_index")["daily_return_pct"].shift(-1) > 0).astype(int)
    prices_df["lag1_return"] = prices_df.groupby("sector_index")["daily_return_pct"].shift(1)
    
    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"])
        news_df["sentiment_num"] = news_df["sentiment"].apply(sentiment_to_numeric)
        news_df["weighted_sentiment"] = news_df["sentiment_num"] * news_df["sentiment_score"]
        
        # Aggregate daily sentiment by sector
        daily_sentiment = news_df.groupby(["date", "sector_index"])["weighted_sentiment"].mean().reset_index()
        
        # Merge
        merged = pd.merge(prices_df, daily_sentiment, on=["date", "sector_index"], how="left")
        merged["weighted_sentiment"] = merged["weighted_sentiment"].fillna(0.0) # Fill days with no news as neutral
    else:
        # No news at all
        merged = prices_df.copy()
        merged["weighted_sentiment"] = 0.0

    merged = merged.dropna() # drops the last row for each sector where target is NaN
    
    # One hot encode sector
    merged = pd.get_dummies(merged, columns=["sector_index"], drop_first=True)
    merged = merged.sort_values("date")
    return merged

def train_and_evaluate():
    logger.info("Building training frame...")
    df = build_training_frame()
    if df.empty:
        logger.warning("Empty training frame. Model not trained.")
        return
    
    features = [c for c in df.columns if c not in ["date", "target_next_day_green", "daily_return_pct"]]
    X = df[features]
    y = df["target_next_day_green"]

    logger.info("Training using TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Time-series cross validation manually or just one final train-test
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Fold {fold+1} Accuracy: {acc:.4f}")

    # Retrain on all data for deployment
    model.fit(X, y)
    
    # Ensure models dir exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "models"), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rf_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Final model saved to {model_path}.")
    
    # Also save the features column order to use when predicting
    features_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "features.pkl")
    joblib.dump(features, features_path)

if __name__ == "__main__":
    train_and_evaluate()
