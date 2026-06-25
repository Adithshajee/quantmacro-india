import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.state import AgentState
from src.utils.sentiment import SentimentAnalyzer  # import existing sentiment utility as requested

# Try importing get_latest_predictions from src.models.predictor
try:
    from src.models.predictor import get_latest_predictions
except ImportError:
    # Define fallback function since get_latest_predictions is not present in predictor.py
    def get_latest_predictions(sector: str) -> dict:
        """
        Fallback function to get real predictions from the PricePredictor ensemble.
        """
        from src.models.predictor import PricePredictor
        from src.database.connection import get_connection
        from src.database.queries import get_latest_prices
        
        sector_lower = sector.lower() if sector else ""
        price_key = None
        news_key = None
        
        if "banking" in sector_lower:
            price_key, news_key = "BANKING_SECTOR", "BSE_BANKEX"
        elif "it" in sector_lower or "tech" in sector_lower:
            price_key, news_key = "IT_SECTOR", "BSE_IT"
        elif "energy" in sector_lower or "power" in sector_lower:
            price_key, news_key = "ENERGY_SECTOR", "BSE_ENERGY"
        elif "sensex" in sector_lower or "market" in sector_lower:
            price_key, news_key = "BSE_SENSEX", "BSE_SENSEX"
        else:
            return {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
            
        conn = get_connection()
        try:
            df = get_latest_prices(conn)
            df_sector = df[df["sector_index"] == price_key].copy()
            if len(df_sector) < 30:
                return {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
                
            predictor = PricePredictor()
            success, _ = predictor.train_and_evaluate(df_sector, news_key)
            if not success:
                return {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
                
            pred_trend, _, confidence = predictor.predict_next_day(df_sector, news_key)
            
            features = getattr(predictor, "feature_names", [])
            clean_features = [f.replace("_lag1", "") for f in features]
            
            return {
                "direction": "UP" if pred_trend == 1 else "DOWN",
                "probability": float(confidence) / 100.0 if confidence else 0.5,
                "features_used": clean_features
            }
        except Exception:
            return {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
        finally:
            conn.close()

def _get_average_sentiment(sector: str) -> float:
    """
    Computes average news sentiment for the sector using get_market_pulse.
    """
    from src.database.connection import get_connection
    from src.database.queries import get_market_pulse
    
    sector_lower = sector.lower() if sector else ""
    news_key = "BSE_SENSEX"
    if "banking" in sector_lower:
        news_key = "BSE_BANKEX"
    elif "it" in sector_lower or "tech" in sector_lower:
        news_key = "BSE_IT"
    elif "energy" in sector_lower or "power" in sector_lower:
        news_key = "BSE_ENERGY"
        
    conn = get_connection()
    try:
        pulse = get_market_pulse(conn)
        return float(pulse.get(news_key, 0.0))
    except Exception:
        return 0.0
    finally:
        conn.close()

def quant_node(state: AgentState) -> AgentState:
    """
    Invokes quantitative model prediction and news sentiment aggregation.
    """
    try:
        sector = state.get("sector", "")
        
        # 1. Fetch ML predictions
        try:
            ml_signal = get_latest_predictions(sector)
            if not ml_signal:
                ml_signal = {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
        except Exception:
            ml_signal = {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
            
        state["ml_signal"] = ml_signal
        
        # 2. Fetch news sentiment score
        try:
            sentiment_val = _get_average_sentiment(sector)
        except Exception:
            sentiment_val = 0.0
            
        state["news_sentiment"] = sentiment_val
        
    except Exception as e:
        state["error"] = str(e)
        state["ml_signal"] = {"direction": "UNKNOWN", "probability": 0.5, "features_used": []}
        state["news_sentiment"] = 0.0
        
    return state
