# QuantMacro India — AI-Powered Sector Intelligence & Macro Analytics Engine

QuantMacro India is an institutional-grade, research-ready, and microservice-friendly market intelligence and quantitative research platform. The system ingests financial news and sectoral index data, scores sentiment semantically using FinBERT, aligns macro indicators (VIX, USDINR, crude oil, yields, CPI, repo rates), and leverages calibrated Voting Ensembles (RandomForest + ExtraTrees + XGBoost/LightGBM) to forecast next-day trend directions. It includes a walk-forward, friction-aware backtesting engine to evaluate advanced allocation strategies (Kelly Criterion, Volatility Targeting) against standard benchmarks.

```text
                                  [DATA SOURCES]
                        (yfinance, NewsAPI, RSS Search)
                                       │
                                       ▼
                             [INGESTION WORKERS]
                  (fetch_bse_data.py, news_fetcher.py)
                                       │
                                       ▼
                             [SQLITE DATABASE]
                         (bse_sector_prices, raw_news)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [NLP & SEMANTIC LAYER]                     [QUANT & FEATURES LAYER]
      (news_mapper.py, sentiment.py)               (predictor.py indicators)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                            [MACHINE LEARNING ENGINE]
                       (Calibrated Voting Ensemble in predictor.py)
                                       │
                                       ▼
                           [QUANTITATIVE BACKTESTER]
                         (Sizing & metrics in engine.py)
                                       │
                                       ▼
                             [FASTAPI BACKEND REST]
                                  (main.py)
                                       │
                                       ▼
                          [STREAMLIT TERM PANEL UI]
                                  (app.py)
```

---

## Key Technical Features

### 1. NLP & Semantic Routing Engine
* **FinBERT Sentiment Analysis**: Ingests financial news headlines and scores sentiment from `-1.0` (most bearish) to `+1.0` (most bullish) using the **FinBERT** (`ProsusAI/finbert`) GPU/CPU inference pipeline, falling back to VADER for low-compute environments.
* **Semantic News Mapper**: Routes headlines to target sectoral categories (`BSE_BANKEX`, `BSE_IT`, `BSE_ENERGY`) using **SentenceTransformers** (`all-MiniLM-L6-v2`) cosine-similarity clustering. Headlines failing similarity thresholds default to `BSE_SENSEX` (general macro).

### 2. Multi-Dimensional Feature Engineering Engine
All engineered signals are **shifted by 1 day (`lag1`)** before model target alignment to eliminate any lookahead bias in forecasting.
* **Trend & Momentum**: 14-day RSI, MACD Line/Signal/Histogram, 5-day, 10-day & 21-day price momentum.
* **Volatility & Risk**: Bollinger Bands width, ATR (Average True Range), realized volatility (rolling 20-day annualized standard deviation of returns), rolling Sortino ratio, rolling 20-day skewness and kurtosis.
* **Volume & State**: Volume Z-Score, rolling 20-day Sharpe ratio, VWAP (Volume Weighted Average Price) proxy.
* **Macro Factors**: Daily integrated metrics: USD/INR exchange rate, Crude Oil spot prices, India VIX Index, 10-Year Government Bond Yields, synthetic Inflation CPI baseline, and synthetic Repo Rate baseline.

### 3. Calibrated Ensemble Forecasting
* **Voting Ensemble**: Fits a soft voting classifier and voting regressor combining `RandomForest`, `ExtraTrees` (for variance reduction), and dynamic hooks for `XGBoost`/`LightGBM` (integrated automatically if available).
* **Platt Scaling Calibration**: Fits `CalibratedClassifierCV` (sigmoid mapping) to ensure output directional probabilities represent calibrated, empirical frequencies.

### 4. Friction-Aware Backtesting Engine
* **Advanced Allocation Strategies**:
  * **AI Kelly Sizing**: Dynamically scales transaction position sizes based on model probabilities using the Kelly Criterion ($2 \times P(\text{UP}) - 1$).
  * **Volatility Targeting**: Adjusts portfolio exposure to target a constant 15% annualized volatility.
  * **Baselines**: Buy & Hold, 5-Day Momentum, SMA Crossover, and Previous-Day return direction.
* **Friction Modeling**: Simulates net-of-cost returns by factoring in **15 bps (0.15%)** of transaction fees and bid-ask slippage per trade.
* **Portfolio Metrics**: Calculates CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Value at Risk (VaR 95%), and Conditional VaR (CVaR 95%).

### 5. Decoupled Architecture
* **FastAPI Backend REST API**: Features asynchronous handlers, strict Pydantic V2 validations, and **NaN serialization cleaning** to safely convert pandas `NaN` values to JSON-compliant `null` formats.
* **Streamlit Terminal**: Dark-themed portfolio dashboard displaying Plotly correlation heatmaps, systematic macro timelines, custom feature weights, and interactive backtesting equity curves.

---

## Project Folder Structure

```text
├── data/                   # SQLite database directory (gitignored)
├── models/                 # Persistent cache for NLP models (gitignored)
├── src/
│   ├── api/                # FastAPI backend endpoints and schemas (main.py)
│   ├── backtesting/        # Quantitative strategy evaluation (engine.py)
│   ├── dashboard/          # Streamlit UI dashboard (app.py)
│   ├── database/           # SQLite connection and tables initialization
│   ├── ingestion/          # Yahoo Finance & news scraping scripts
│   ├── insights/           # Multi-agent AI explanation engine (llm.py)
│   ├── models/             # ML predictor with quantitative features (predictor.py)
│   └── utils/              # Configuration, logging, and sentiment utilities
├── requirements.txt        # Python package dependencies
└── .env.example            # Environment configurations blueprint
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- SQLite3

### 1. Local Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Adithshajee/bse-macro-sector-analyzer.git
   cd bse-macro-sector-analyzer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

---

## Running the Platform

### Unified Local Mode (Recommended)
You can launch the entire platform (both the FastAPI backend and Streamlit dashboard) with a **single command**. The Streamlit frontend automatically checks the health of the FastAPI server and spawns it programmatically in the background if it is offline:
```bash
streamlit run src/dashboard/app.py
```
* Once launched, the dashboard will be available at [http://localhost:8501](http://localhost:8501) and the FastAPI server at [http://127.0.0.1:8000](http://127.0.0.1:8000) (Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)).

### Manual Decoupled Mode (Optional)
If you prefer to run the services in separate processes manually:
1. **Start the FastAPI Backend**:
   ```bash
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Start the Streamlit Frontend**:
   In a separate terminal, run:
   ```bash
   streamlit run src/dashboard/app.py
   ```


---

## Data Ingestion & Inferences
The platform automatically refreshes data on startup or via the **"Force Data Refresh"** button in the dashboard. You can also run ingestion manually:
```bash
# Ingest historical prices from Yahoo Finance & macro indices
python src/ingestion/fetch_bse_data.py

# Ingest and score news headlines semantically
python src/ingestion/news_fetcher.py
```

---

## Model Risks & Uncertainty Disclosures

> [!WARNING]
> **Probabilistic Model Predictions**:
> Machine learning forecasts (such as next-day directional predictions) are probabilistic estimates based on historical signals. They do not guarantee future returns, cannot anticipate black-swan events or structural market regime shifts, and are subject to statistical estimation error.

> [!IMPORTANT]
> **Regulatory Disclosure**:
> This dashboard is a quantitative financial modeling showcase built strictly for educational, research, and portfolio demonstration purposes. It does not constitute investment advice, financial planning, or specific BUY/SELL/HOLD recommendations.

---

## Author

**Adith Shajee**  
B.Tech in Artificial Intelligence and Machine Learning
