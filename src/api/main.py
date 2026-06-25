import os
import sys
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class IngestResponse(BaseModel):
    status: str
    message: str

class SectorInfo(BaseModel):
    name: str
    price_key: str
    news_key: str

class PredictionMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    rmse: float

class PredictionDetails(BaseModel):
    trend: str
    target_trend_code: int
    predicted_price: float
    confidence: float
    metrics: PredictionMetrics

class PredictionResponse(BaseModel):
    trained: bool
    prediction: Optional[PredictionDetails] = None
    message: Optional[str] = None

class BacktestMetricDetail(BaseModel):
    total_return: float = Field(alias="Total Return")
    cagr: float = Field(alias="CAGR")
    annualized_volatility: float = Field(alias="Annualized Volatility")
    sharpe_ratio: float = Field(alias="Sharpe Ratio")
    sortino_ratio: float = Field(alias="Sortino Ratio")
    calmar_ratio: float = Field(alias="Calmar Ratio")
    max_drawdown: float = Field(alias="Max Drawdown")
    var_95: float = Field(alias="VaR_95")
    cvar_95: float = Field(alias="CVaR_95")
    win_rate: float = Field(alias="Win Rate")
    ci_lower_daily: float = Field(alias="CI_Lower_Daily")
    ci_upper_daily: float = Field(alias="CI_Upper_Daily")

    model_config = {
        "populate_by_name": True
    }

class BacktestResponse(BaseModel):
    metrics: Dict[str, BacktestMetricDetail]
    curves: List[Dict[str, Any]]

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.connection import get_connection
from src.database.queries import get_latest_prices, get_latest_news_for_sector, get_market_pulse
from src.ingestion.news_fetcher import run_ingestion
from src.ingestion.fetch_bse_data import main as run_price_ingestion
from src.models.predictor import PricePredictor
from src.backtesting.engine import BacktestEngine
from src.insights.engine import generate_insights
from src.insights.llm import explain_market_condition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_backend")

app = FastAPI(
    title="QuantMacro India API",
    description="Production-grade API backend providing institutional market intelligence, sentiment analytics, machine learning forecasts, and backtesting metrics for Indian equities.",
    version="3.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECTOR_MAP = {
    "Banking": {"price": "BANKING_SECTOR", "news": "BSE_BANKEX"},
    "IT": {"price": "IT_SECTOR", "news": "BSE_IT"},
    "Energy": {"price": "ENERGY_SECTOR", "news": "BSE_ENERGY"},
    "Market (Sensex)": {"price": "BSE_SENSEX", "news": "BSE_SENSEX"},
}

class IngestResponse(BaseModel):
    status: str
    message: str

class SectorInfo(BaseModel):
    name: str
    price_key: str
    news_key: str

@app.get("/")
def read_root():
    return {
        "title": "Indian Sector Market Intelligence Platform API",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/api/sectors",
            "/api/prices/{sector}",
            "/api/sentiment/{sector}",
            "/api/predict/{sector}",
            "/api/backtest/{sector}",
            "/api/insights/{sector}",
            "/api/ingest"
        ]
    }

@app.get("/health")
def health_check():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/api/sectors", response_model=List[SectorInfo])
def get_sectors():
    return [
        SectorInfo(name=k, price_key=v["price"], news_key=v["news"])
        for k, v in SECTOR_MAP.items()
    ]

@app.get("/api/prices/{sector}")
def get_prices(sector: str):
    if sector not in SECTOR_MAP:
        raise HTTPException(status_code=404, detail="Sector not found")
    
    price_key = SECTOR_MAP[sector]["price"]
    conn = get_connection()
    try:
        df = get_latest_prices(conn)
        df_sector = df[df["sector_index"] == price_key].copy()
        if df_sector.empty:
            return []
        
        # Prepare data with technical indicators
        predictor = PricePredictor()
        df_processed, _ = predictor.prepare_data(df_sector, SECTOR_MAP[sector]["news"])
        
        # Convert date to string
        df_processed['date'] = df_processed['date'].dt.strftime('%Y-%m-%d')
        # Convert NaN values to None for JSON compliance
        import numpy as np
        df_clean = df_processed.replace({np.nan: None})
        return df_clean.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching prices for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/sentiment/{sector}")
def get_sentiment(sector: str):
    if sector not in SECTOR_MAP:
        raise HTTPException(status_code=404, detail="Sector not found")
    
    news_key = SECTOR_MAP[sector]["news"]
    conn = get_connection()
    try:
        df_news = get_latest_news_for_sector(news_key, limit=50, conn=conn)
        
        # Get overall market sentiment pulse
        pulse = get_market_pulse(conn)
        
        return {
            "news": df_news.to_dict(orient="records") if not df_news.empty else [],
            "avg_sentiment": float(pulse.get(news_key, 0.0))
        }
    except Exception as e:
        logger.error(f"Error fetching sentiment for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/predict/{sector}", response_model=PredictionResponse)
def get_predictions(sector: str):
    if sector not in SECTOR_MAP:
        raise HTTPException(status_code=404, detail="Sector not found")
        
    price_key = SECTOR_MAP[sector]["price"]
    news_key = SECTOR_MAP[sector]["news"]
    conn = get_connection()
    
    try:
        df = get_latest_prices(conn)
        df_sector = df[df["sector_index"] == price_key].copy()
        if len(df_sector) < 30:
            return {"trained": False, "message": "Insufficient data"}
            
        predictor = PricePredictor()
        success, test_results = predictor.train_and_evaluate(df_sector, news_key)
        
        if not success:
            return {"trained": False, "message": "Failed to train model"}
            
        pred_trend, pred_price, confidence = predictor.predict_next_day(df_sector, news_key)
        
        return {
            "trained": True,
            "prediction": {
                "trend": "UP" if pred_trend == 1 else "DOWN",
                "target_trend_code": int(pred_trend),
                "predicted_price": float(pred_price),
                "confidence": float(confidence),
                "metrics": predictor.metrics
            }
        }
    except Exception as e:
        logger.error(f"Prediction error for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/backtest/{sector}", response_model=BacktestResponse)
def get_backtest(sector: str):
    if sector not in SECTOR_MAP:
        raise HTTPException(status_code=404, detail="Sector not found")
        
    price_key = SECTOR_MAP[sector]["price"]
    news_key = SECTOR_MAP[sector]["news"]
    conn = get_connection()
    
    try:
        df = get_latest_prices(conn)
        df_sector = df[df["sector_index"] == price_key].copy()
        if len(df_sector) < 30:
            raise HTTPException(status_code=400, detail="Insufficient data to backtest")
            
        # Train model
        predictor = PricePredictor()
        success, test_results = predictor.train_and_evaluate(df_sector, news_key)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to train model for backtest")
            
        # Run Backtester
        engine = BacktestEngine(transaction_cost=0.001, slippage=0.0005)
        backtest_results = engine.run_backtest(test_results, test_results['predicted_trend'])
        
        # Serialize curves
        curves_df = backtest_results["curves"]
        curves_df['date'] = curves_df['date'].astype(str)
        import numpy as np
        curves_clean = curves_df.replace({np.nan: None})
        
        return {
            "metrics": backtest_results["metrics"],
            "curves": curves_clean.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Backtesting error for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/insights/{sector}")
def get_insights(sector: str):
    if sector not in SECTOR_MAP:
        raise HTTPException(status_code=404, detail="Sector not found")
        
    price_key = SECTOR_MAP[sector]["price"]
    news_key = SECTOR_MAP[sector]["news"]
    conn = get_connection()
    
    try:
        df = get_latest_prices(conn)
        df_sector = df[df["sector_index"] == price_key].copy()
        df_news = get_latest_news_for_sector(news_key, limit=20, conn=conn)
        
        if df_sector.empty:
            return {"insights": [], "explanation": "No price data found."}
            
        insights = generate_insights(df_sector, df_news)
        
        # Get model confidence to pass to explanation generator
        confidence = None
        try:
            predictor = PricePredictor()
            success, _ = predictor.train_and_evaluate(df_sector, news_key)
            if success:
                _, _, confidence = predictor.predict_next_day(df_sector, news_key)
        except Exception:
            pass
            
        headlines = df_news['headline'].tolist() if not df_news.empty else []
        explanation = explain_market_condition(sector, insights, headlines, confidence)
        
        return {
            "insights": insights,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Insights generation error for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

def async_ingestion_task():
    logger.info("Executing async background ingestion task...")
    try:
        # Run price and news ingestions sequentially
        run_price_ingestion()
        run_ingestion()
        logger.info("Background ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error in background ingestion: {e}")

@app.post("/api/ingest", response_model=IngestResponse)
def trigger_ingestion(background_tasks: BackgroundTasks):
    background_tasks.add_task(async_ingestion_task)
    return IngestResponse(
        status="accepted",
        message="Background ingestion pipeline triggered successfully."
    )

from src.agents.graph import run_analysis
from pydantic import BaseModel

class AgentQueryRequest(BaseModel):
    question: str
    sector: str = ""

class AgentQueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: str
    ml_direction: str
    ml_probability: float
    news_sentiment: float
    error: str = ""

@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    try:
        result = run_analysis(query=request.question, sector=request.sector)
        return AgentQueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/health")
async def agent_health():
    return {"status": "ok", "agents": ["retriever", "quant", "analyst"], "llm": "gemini-2.5-flash"}
