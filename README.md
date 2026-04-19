# BSE Macro-to-Sector Analyzer

A financial data engineering project that collects macroeconomic news and BSE sector market data to analyze relationships between macro events and sector performance.

## Features

• Automated BSE sector data ingestion  
• Financial news collection pipeline  
• Structured SQLite database  
• Modular Python architecture  
• Streamlit Dashboard
• Machine Learning model using FinBERT and Random Forest for trend prediction

## Folder Structure
- `src/api/` - FastAPI predictions
- `src/dashboard/` - Streamlit UI
- `src/database/` - Data definitions
- `src/ingestion/` - News & market data gathering
- `src/mapping/` - Rule-based sector mapping
- `src/modeling/` - Sentimental analysis & ML model
- `src/utils/` - Config and logging

## How to run from scratch

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables in `.env`:
   ```env
   DB_PATH=data/project.db
   NEWS_API_KEY=your_key_here
   MODEL_NAME=ProsusAI/finbert
   TARGET_SECTORS=BSE_BANKEX,BSE_IT,BSE_ENERGY
   ```
3. Initialize the database:
   ```bash
   python src/database/db_setup.py
   ```
4. Fetch Data:
   ```bash
   python src/ingestion/fetch_bse_data.py
   python src/ingestion/fetch_news.py
   ```
5. Map News to Sectors:
   ```bash
   python src/mapping/sector_mapper.py
   ```
6. Score Sentiment:
   ```bash
   python src/modeling/sentiment.py
   ```
7. Train the Model:
   ```bash
   python src/modeling/train_model.py
   ```
8. Run the Dashboard:
   ```bash
   streamlit run src/dashboard/app.py
   ```
9. Run the API (Deployment):
   ```bash
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
   Or using Docker:
   ```bash
   docker build -t bse-analyzer .
   docker run -p 8000:8000 bse-analyzer
   ```