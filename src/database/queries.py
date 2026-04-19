from src.database.connection import get_connection
import pandas as pd

def get_latest_prices(conn=None):
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
        
    df = pd.read_sql("SELECT * FROM bse_sector_prices ORDER BY date ASC", conn)
    
    if close_conn:
        conn.close()
    return df

def get_latest_news_for_sector(sector, limit=20, conn=None):
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
        
    query = """
    SELECT r.published_at, r.headline, m.mapping_reason, r.sentiment, r.sentiment_score
    FROM raw_news r
    INNER JOIN news_sector_mapping m ON r.id = m.news_id
    WHERE m.sector_index = ?
    ORDER BY r.published_at DESC
    LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(sector, limit))
    
    if close_conn:
        conn.close()
    return df

def get_top_impact_articles(limit=3, conn=None):
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
        
    query = """
    SELECT r.headline, r.sentiment, r.sentiment_score, m.sector_index
    FROM raw_news r
    INNER JOIN news_sector_mapping m ON r.id = m.news_id
    WHERE r.sentiment_score IS NOT NULL
    ORDER BY ABS(r.sentiment_score) DESC
    LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(limit,))
    
    if close_conn:
        conn.close()
    return df

def get_market_pulse(conn=None):
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
        
    query = """
    SELECT m.sector_index, AVG(CASE 
        WHEN r.sentiment = 'positive' THEN r.sentiment_score
        WHEN r.sentiment = 'negative' THEN -r.sentiment_score
        ELSE 0 END) as avg_sentiment
    FROM raw_news r
    INNER JOIN news_sector_mapping m ON r.id = m.news_id
    WHERE r.sentiment_score IS NOT NULL
    GROUP BY m.sector_index
    """
    df = pd.read_sql(query, conn)
    
    if close_conn:
        conn.close()
    return dict(zip(df['sector_index'], df['avg_sentiment']))
