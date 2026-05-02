# BSE Macro-to-Sector Analyzer

BSE Macro-to-Sector Analyzer is a financial intelligence platform that studies how macroeconomic developments influence sector-level market behavior in Indian equity markets.

The system collects BSE sector market data and macroeconomic news, transforms them into structured signals, applies FinBERT-based sentiment analysis and machine learning, and delivers sector-level insights through an interactive Streamlit dashboard and FastAPI prediction service.

---

## Features

- Automated BSE sector market data ingestion
- Financial news collection pipeline
- Rule-based news-to-sector mapping
- FinBERT-powered sentiment analysis
- Random Forest trend prediction
- Structured SQLite database
- Interactive Streamlit dashboard
- FastAPI prediction API
- Docker-ready deployment

---

## Project Structure

```text
src/
├── api/           # FastAPI prediction endpoints
├── dashboard/     # Streamlit dashboard
├── database/      # Database setup and schema
├── ingestion/     # Market and news data collection
├── mapping/       # News-to-sector mapping
├── modeling/      # Sentiment analysis and ML training
└── utils/         # Configuration and logging
How It Works
Collect BSE sector historical market data
Collect macroeconomic and financial news
Map news articles to relevant sectors
Generate sentiment scores using FinBERT
Train a Random Forest model on market and sentiment features
Display sector-level insights through the dashboard
Installation

Clone the repository:

git clone https://github.com/Adithshajee/bse-macro-sector-analyzer.git
cd bse-macro-sector-analyzer

Install dependencies:

pip install -r requirements.txt

Create a .env file:

DB_PATH=data/project.db
NEWS_API_KEY=your_key_here
MODEL_NAME=ProsusAI/finbert
TARGET_SECTORS=BSE_BANKEX,BSE_IT,BSE_ENERGY
Run From Scratch

Initialize the database:

python src/database/db_setup.py

Fetch market data:

python src/ingestion/fetch_bse_data.py

Fetch news:

python src/ingestion/fetch_news.py

Map news to sectors:

python src/mapping/sector_mapper.py

Generate sentiment scores:

python src/modeling/sentiment.py

Train the model:

python src/modeling/train_model.py
Run Dashboard
streamlit run src/dashboard/app.py
Run API
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
Docker

Build:

docker build -t bse-analyzer .

Run:

docker run -p 8000:8000 bse-analyzer
Live Demo

https://bse-ai-decision-support-system.streamlit.app/

Author

Adith Shajee
B.Tech — Artificial Intelligence and Machine Learning
