import pandas as pd
import numpy as np

def generate_insights(price_df, news_df):
    """
    Analyzes divergence between price trend and sentiment trend.
    Outputs human readable insight messages and stats.
    """
    insights = []
    
    if price_df.empty or len(price_df) < 5:
        return ["Not enough data to generate insights."]

    # Calculate short term (5-day) price return
    current_price = price_df['close_price'].iloc[-1]
    past_price = price_df['close_price'].iloc[-5] if len(price_df) >= 5 else price_df['close_price'].iloc[0]
    price_return = (current_price - past_price) / past_price * 100
    
    # Analyze Sentiment
    recent_sentiment = 0.0
    sentiment_trend = "neutral"
    if not news_df.empty:
        # Get average sentiment of last 10 articles
        recent_news = news_df.head(10)
        recent_sentiment = recent_news['sentiment_score'].mean()
        
        if recent_sentiment > 0.3:
            sentiment_trend = "positive"
        elif recent_sentiment < -0.3:
            sentiment_trend = "negative"

    # Detect Divergence
    if price_return > 2.0 and sentiment_trend == "negative":
        insights.append("⚠️ **Bearish Divergence Detected**: Price is rising, but news sentiment is overwhelmingly negative. Market risk is increasing.")
    elif price_return < -2.0 and sentiment_trend == "positive":
        insights.append("🚀 **Bullish Divergence Detected**: Price is falling, but news sentiment is highly positive. Potential accumulation zone.")
    elif price_return > 1.0 and sentiment_trend == "positive":
        insights.append("✅ **Strong Bullish Confirmation**: Both price action and news sentiment are positive. Momentum is strong.")
    elif price_return < -1.0 and sentiment_trend == "negative":
        insights.append("🚨 **Strong Bearish Confirmation**: Both price action and news sentiment are negative. Extreme caution advised.")
    else:
        insights.append("📊 **Consolidation Phase**: Market is moving sideways with mixed or neutral sentiment signals.")

    # Correlation (if we had enough aligned daily sentiment, simplify here)
    insights.append(f"Recent Price Change (5-day): **{price_return:.2f}%**")
    insights.append(f"Average Recent Sentiment: **{recent_sentiment:.2f}** ({sentiment_trend})")

    return insights
