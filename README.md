# QuantMacro India — AI-Powered Sector Intelligence & Macro Analytics Engine

A technically rigorous, portfolio-grade Indian market intelligence and quantitative research platform. The system ingests financial news and sectoral index data, scores sentiment semantically using FinBERT, aligns macro indicators (VIX, USDINR, crude oil, yields), and leverages calibrated Voting Ensembles (RandomForest + ExtraTrees + XGBoost/LightGBM) to forecast next-day trend directions. It includes a walk-forward, friction-aware backtesting engine to evaluate advanced allocation strategies (Kelly Criterion, Volatility Targeting) against standard benchmarks.

---

## Key Technical Features

1. **Dual Sentiment Architecture**: Features a high-performance **FinBERT** (`ProsusAI/finbert`) GPU/CPU inference pipeline using PyTorch and Hugging Face `transformers` with a fully automated VADER fallback for lightweight environments.
2. **Semantic News Routing**: Routes news headlines to relevant sector mappings using **SentenceTransformers** cosine similarity (`all-MiniLM-L6-v2`) with a robust keyword fallback.
3. **Advanced Quantitative Feature Engineering**: Computes:
   - **Trend/Momentum**: 14-day RSI, MACD Line/Signal/Histogram, 5-day, 10-day & 21-day momentum.
   - **Volatility/Risk**: Bollinger Bands width, ATR (Average True Range), realized volatility (rolling 20-day annualized std dev), rolling Sortino ratio, rolling skewness/kurtosis, and max drawdown.
   - **Volume/State**: Volume Z-Score, rolling Sharpe ratio, VWAP (Volume Weighted Average Price) proxy.
   - **Macro Factors**: Integrates USD/INR, Crude Oil, India VIX, 10-Year yields, inflation CPI, and repo rates.
4. **No-Lookahead ML Pipeline**: Shifting all engineered features by 1 day (`lag1`) to completely eliminate lookahead bias in training and evaluation.
5. **Calibrated Ensemble Model**: Integrates a Voting Ensemble classifier and regressor with Platt Scaling probability calibration (`CalibratedClassifierCV`) to ensure model confidence reflects true empirical frequencies.
6. **Advanced Backtesting & Sizing**: Simulates trading strategies net of **15 bps (0.15%)** transaction friction and bid-ask slippage. Includes **Kelly Criterion** and **Volatility Targeting** sizing models, calculating CAGR, Sharpe, Sortino, Calmar, VaR (95%), and CVaR (95%).
7. **Multi-Agent Explanations**: Features a collaborative AI analyst team prompt structure (Macro Analyst, Sector Analyst, Risk Management, Coordinator) to explain forecasts.
8. **Decoupled API Architecture**: Versioned FastAPI REST API backend with strict Pydantic v2 validation models and NaN serialization cleaning, serving a visual Streamlit dashboard with Plotly correlation heatmaps, macro timelines, and equity curves.

---

## Folder Structure

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
├── Dockerfile              # Build configuration for containerization
├── docker-compose.yml      # Service definitions for backend + frontend
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

### Option A: Standalone Local Mode (Automatic Fallback)
You can launch the dashboard directly. It will detect that the backend API is offline and compute all ML and backtesting metrics locally:
```bash
streamlit run src/dashboard/app.py
```

### Option B: Decoupled REST Mode (Recommended)
1. **Start the FastAPI Backend**:
   ```bash
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *The Swagger API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

2. **Start the Streamlit Frontend**:
   In a separate terminal, run:
   ```bash
   streamlit run src/dashboard/app.py
   ```

### Option C: Containerized Mode (Docker Compose)
To run the entire platform (FastAPI + Streamlit) inside isolated Docker containers:
```bash
docker-compose up --build
```
- Streamlit dashboard: [http://localhost:8501](http://localhost:8501)
- FastAPI backend: [http://localhost:8000](http://localhost:8000)

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

## Disclaimer
*This platform is designed for research, education, and portfolio demonstration purposes. It does not provide financial advice, recommendations, or endorsements for active trading.*

---

## Author

**Adith Shajee**  
B.Tech in Artificial Intelligence and Machine Learning
