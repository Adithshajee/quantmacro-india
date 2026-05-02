# BSE Macro-to-Sector Analyzer

A financial intelligence platform for analyzing how macroeconomic developments influence sector-level market behavior in Indian equity markets.

The project collects BSE sector market data and macroeconomic news, applies FinBERT-based sentiment analysis and machine learning, and provides sector-level analytical insights through a Streamlit dashboard and FastAPI prediction service.

## Features

* Automated BSE sector market data ingestion
* Financial news collection pipeline
* Rule-based news-to-sector mapping
* FinBERT-powered sentiment analysis
* Random Forest trend prediction
* SQLite data storage
* Interactive Streamlit dashboard
* FastAPI prediction API
* Docker-ready deployment

## Project Structure

```text
src/
├── api/           # FastAPI prediction endpoints
├── dashboard/     # Streamlit dashboard
├── database/      # Database schema and setup
├── ingestion/     # Market and news ingestion
├── mapping/       # News-to-sector mapping
├── modeling/      # Sentiment analysis and ML training
└── utils/         # Configuration and logging
```

## Installation

```bash
git clone https://github.com/Adithshajee/bse-macro-sector-analyzer.git
cd bse-macro-sector-analyzer
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
DB_PATH=data/project.db
NEWS_API_KEY=your_key_here
MODEL_NAME=ProsusAI/finbert
TARGET_SECTORS=BSE_BANKEX,BSE_IT,BSE_ENERGY
```

## Run the Pipeline

Initialize the database:

```bash
python src/database/db_setup.py
```

Fetch market data:

```bash
python src/ingestion/fetch_bse_data.py
```

Fetch news:

```bash
python src/ingestion/fetch_news.py
```

Map news to sectors:

```bash
python src/mapping/sector_mapper.py
```

Generate sentiment scores:

```bash
python src/modeling/sentiment.py
```

Train the model:

```bash
python src/modeling/train_model.py
```

## Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Run the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Docker

```bash
docker build -t bse-analyzer .
docker run -p 8000:8000 bse-analyzer
```

## Live Demo

[https://bse-ai-decision-support-system.streamlit.app/](https://bse-ai-decision-support-system.streamlit.app/)

## Author

**Adith Shajee**
B.Tech in Artificial Intelligence and Machine Learning
