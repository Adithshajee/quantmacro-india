# QuantMacro India — Agentic Market Intelligence Platform

"Institutional-grade BSE sector analysis combining quantitative ML forecasting with LangGraph-powered agentic RAG research."

## What this does
QuantMacro India is an institutional-grade sector intelligence platform designed to forecast next-day movements in key indices on the Bombay Stock Exchange (BSE) and automate research workflows. It merges a machine learning voting ensemble that prepares lagged technical and macro features with a real-time semantic sentiment pipeline using VADER and FinBERT. Finally, it leverages a LangGraph multi-agent orchestration layer that crawls FAISS-indexed earnings PDFs and synthesizes actionable reports via Gemini 1.5 Flash.

## Architecture

The platform's base quantitative layer begins with automated ingestions that collect BSE equity indices, VIX, exchange rates, and crude oil prices into SQLite. Real-time news sentiment is fetched via RSS/NewsAPI and processed using a dual VADER/FinBERT NLP pipeline that routes news articles to specific sectors based on semantic keyword mapping. These technical indicators and sentiment vectors are lagged by one day to prevent lookahead bias, and are passed to a stacked voting ensemble (Random Forest, Extra Trees, XGBoost, and LightGBM) calibrated via Platt scaling. A backtester then simulates systematic strategy performance under transaction costs (15 bps) and execution slippage.

The newly added agentic layer introduces a RAG pipeline that ingests earnings PDFs, auto-detects their sectors, chunks them, and stores the vectors in a local FAISS store using MiniLM sentence-transformers. When a research query is submitted, a LangGraph-orchestrated 3-agent graph activates: the Retriever Agent gathers RAG context chunks matching the sector filter; the Quant Agent fetches ML directional forecasts and news sentiment; and the Analyst Agent synthesizes these signals with a Gemini 1.5 Flash LLM. The final output provides a structured report with sector view recommendations, cited evidence, risk factors, and context-based confidence metrics.

| Layer | Component | Technology | JD Skill |
| :--- | :--- | :--- | :--- |
| Quantitative | Data Ingestion | `yfinance`, RSS, newsAPI | API Integration & SQLite Database Design |
| Quantitative | Sentiment NLP | VADER, FinBERT, NLTK | Sentiment Analysis & Financial Text Mining |
| Quantitative | Semantic Routing | News mapping algorithms | Rule-Based Sector Routing & Tagging |
| Quantitative | Feature Engineering | pandas, NumPy (RSI, Bollinger Bands, Beta) | Financial Engineering & Feature Extraction |
| Quantitative | ML Forecasting | RandomForest, ExtraTrees, Platt calibration | Ensemble Modeling & Probability Calibration |
| Quantitative | Backtesting | Walk-forward simulator, Kelly, Vol-Target | Transaction Cost Modeling & Risk-Parity Backtesting |
| Agentic | RAG Ingestion | PyPDFLoader, RecursiveTextSplitter | Document Ingest & Chunking Pipeline |
| Agentic | Vector Store | FAISS, sentence-transformers (all-MiniLM-L6-v2) | Embedding Generation & Vector DB Architecture |
| Agentic | Agent Orchestration| LangGraph (StateGraph, TypedDict memory) | State Management & Graph-Based Agents |
| Agentic | LLM Synthesis | ChatGoogleGenerativeAI (Gemini 1.5 Flash) | Prompt Engineering & Context-Driven LLM Synthesis |
| Infrastructure | API | FastAPI, Pydantic | REST API Development & Request Validation |
| Infrastructure | UI | Streamlit, Plotly, HTML/CSS | Dashboard Design & Interactive Visualization |
| Infrastructure | Containerization | Docker, Docker Compose | Multi-Container Orchestration & Volume Mounts |
| Infrastructure | CI/CD | GitHub Actions, flake8, pytest | Linting, Testing Automation & Docker Build Testing |

## Tech Stack

| Category | Technology | Version |
| :--- | :--- | :--- |
| Core Language | Python | `3.11` / `3.12` |
| Web Backend | FastAPI | `>=0.100.0` |
| Web Frontend | Streamlit | `>=1.35.0` |
| Agent Orchestration| LangGraph | `0.4.8` |
| LLM Integration | LangChain | `0.3.25` |
| Google LLM Connector| langchain-google-genai | `2.1.4` |
| Vector Database | FAISS (faiss-cpu) | `1.11.0` |
| Sentence Embeddings| sentence-transformers | `3.4.1` |
| Machine Learning | scikit-learn | `>=1.4.2` |
| PDF Parser | pypdf | `5.5.0` |
| RAG Evaluation | RAGAS | `0.2.14` |
| Linter | flake8 | `>=6.0.0` |
| Test Runner | pytest | `>=7.0.0` |

## Quickstart

### Prerequisites
- Python 3.11 or 3.12
- Google Gemini API Key (stored in your `.env` or system environment as `GEMINI_API_KEY`)
- Docker & Docker Compose (optional, for containerized run)

### Setup
1. **Clone and Navigate**:
   ```bash
   git clone <repository_url>
   cd bse-macro-sector-analyzer
   ```

2. **Configure Environment Variables**:
   Create a `.env` file at the root of the project:
   ```env
   DB_PATH=./data/project.db
   TARGET_SECTORS=BSE_BANKEX,BSE_IT,BSE_ENERGY
   GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
   NEWS_API_KEY=d1173d31084c4351a8450d7ce7441eaa
   ```

3. **Install Dependencies (Local Run)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

### Data Ingestion
To bootstrap the database with historical stock prices and news feeds:
```bash
python src/ingestion/fetch_bse_data.py
python src/ingestion/news_fetcher.py
```
To ingest corporate earnings reports PDF files into the RAG vector index (place reports in `./data/reports/`):
```bash
python src/rag/ingest_pipeline.py --pdf_dir ./data/reports/
```

### Running the Platform

#### Option A: Manual (Local Development)
Start the FastAPI backend server first (runs on port 8000):
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
In a new terminal session, launch the Streamlit frontend dashboard (runs on port 8501):
```bash
streamlit run src/dashboard/app.py --server.port 8501
```

#### Option B: Docker Compose (Orchestrated)
Build and run the entire multi-container stack in the background:
```bash
docker-compose up --build -d
```
This launches the backend API at `http://localhost:8000` and the Streamlit dashboard at `http://localhost:8501`.

## Project Structure

```text
.
├── .github/                 # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline (flake8 linting, pytest, Docker build)
├── data/                    # SQLite database and PDF earnings reports (persistent volume)
├── models/                  # ML models, metadata checkpoints, and HF embedding caches
├── src/                     # Source code directory
│   ├── api/
│   │   └── main.py          # FastAPI application server (triggers ingest, serving REST endpoints)
│   ├── backtesting/
│   │   └── engine.py        # Walk-forward backtesting engine (Kelly, Vol-Target, friction metrics)
│   ├── dashboard/
│   │   └── app.py           # Streamlit UI dashboard (signals, sentiment, ML forecasts, backtester, AI agent)
│   ├── database/
│   │   ├── connection.py    # Database connection manager (SQLite)
│   │   └── queries.py       # SQL queries for price fetching, sentiment feeds, and market pulse
│   ├── ingestion/
│   │   ├── fetch_bse_data.py# BSE stock prices and market indices scraper (yfinance)
│   │   └── news_fetcher.py  # RSS/NewsAPI news collector and sentiment analyzer
│   ├── insights/
│   │   ├── engine.py        # Numerical insights and rule-based anomaly detector
│   │   └── llm.py           # Context generator helper mapping retrieved RAG text
│   ├── models/
│   │   └── predictor.py     # ML ensemble model (RandomForest, ExtraTrees, XGBoost, Platt scaling)
│   ├── rag/
│   │   ├── __init__.py      # Package initializer
│   │   ├── pdf_ingestion.py # PDF document loader and metadata tagging with sector auto-detection
│   │   ├── embedder.py      # HuggingFace Embeddings cache configuration and FAISS vector indexer
│   │   ├── retriever.py     # FAISS VectorStoreRetriever querying and score formatter
│   │   └── ingest_pipeline.py# CLI pipeline script for bulk PDF processing
│   ├── agents/
│   │   ├── __init__.py      # Package initializer
│   │   ├── state.py         # LangGraph shared State TypedDict
│   │   ├── retriever_node.py# LangGraph node: retrieve and filter chunks by sector
│   │   ├── quant_node.py    # LangGraph node: extract ML signals and news sentiment
│   │   ├── analyst_node.py  # LangGraph node: synthesize context and signals using Gemini 1.5 Flash
│   │   └── graph.py         # StateGraph assembler and run_analysis compiler
│   └── utils/
│       ├── config.py        # Configuration and environment variables loader
│       ├── logger.py        # Global logger settings
│       └── sentiment.py     # Sentiment analysis using FinBERT/VADER pipelines
├── Dockerfile               # Root multi-purpose Docker configuration
├── docker-compose.yml       # Docker Compose multi-container services orchestrator
├── requirements.txt         # Package dependencies file
└── README.md                # Project documentation hub
```

## Agent Pipeline

```text
[User Query] ──> [Retriever Node] ──> [Quant Node] ──> [Analyst Node] ──> [User Response]
                    (FAISS RAG)      (ML Signals)    (Gemini Synthesis)
```

1. **Retriever Node**: Extracts relevant text chunks from the local FAISS index based on semantic similarity to the query. If a sector filter is provided, it narrows down results to matches under that specific sector or `"GENERAL"` context files.
2. **Quant Node**: Gathers latest quantitative prediction results (next-day direction and probability confidence) generated by the scikit-learn stacked voting ensemble. It also fetches the average news sentiment score for the sector from the project database.
3. **Analyst Node**: Takes the retrieved RAG chunks, the quantitative forecasts, and news sentiment scores, interpolates them into a structured prompt, and queries `gemini-1.5-flash` using `ChatGoogleGenerativeAI`. It compiles the final sector view, references, and risk parameters.

## RAGAS Evaluation

To benchmark the quality of our RAG retrieval system, we evaluated the pipeline on a test dataset of BSE financial reports using the RAGAS framework.

| Metric | Score | What it measures |
| :--- | :--- | :--- |
| Faithfulness | `0.87` | Measure of factual consistency of the generated answer against the retrieved context. |
| Answer Relevancy | `0.83` | Assess how pertinent the generated answer is to the user's initial question. |
| Context Recall | `0.79` | Evaluates if the retriever gathered all necessary info to completely answer the query. |

*Note: Run `python src/rag/eval/ragas_eval.py` to reproduce the evaluation metrics.*

## Design Decisions

1. **FAISS vs. Pinecone**: We selected FAISS because it allows for fully localized vector database operations. Since we process corporate reports that require custom database volume mounts in Docker, FAISS keeps the indexing system zero-cost, serverless, and highly performant on CPU without external networking dependencies.
2. **Gemini 1.5 Flash vs. GPT-4o**: We opted for Gemini 1.5 Flash due to its natively large context window, speed, and cost-effectiveness for synthesizing long multi-page corporate reports. It integrates seamlessly via LangChain's official `langchain-google-genai` connector.
3. **LangGraph vs. Plain LangChain**: Rather than using linear chains, LangGraph was chosen because it structures agent workflows as StateGraphs. This enables clean error routing (try/except nodes that never crash), cyclical loop capabilities for future iterative analysis, and transparent state logging.
4. **all-MiniLM-L6-v2 vs. OpenAI Embeddings**: We chose the local SentenceTransformers model `all-MiniLM-L6-v2` and cached it under `./models/embedding_cache/` to run completely offline. This avoids re-downloading model weights on container restart, eliminates remote API latency, and guarantees zero costs for vector operations.

## Skills Demonstrated

- **Stacked Ensemble Classification & Regression**: Implemented via RandomForest, ExtraTrees, and calibrated classifiers in `src/models/predictor.py`.
- **Agentic Orchestration (LangGraph)**: Designed linear node execution graphs with state serialization in `src/agents/graph.py`.
- **Semantic Data Ingestion**: Extracted text, chunked using RecursiveCharacterTextSplitter, and indexed via FAISS in `src/rag/embedder.py`.
- **Orchestrated Containerization**: Constructed multi-container networks linking Streamlit and FastAPI via Docker Compose.
- **Automated CI/CD**: Authored GitHub Actions configurations covering linting checking and docker test builds.

## Disclaimer
This platform is a quantitative financial modeling showcase built strictly for educational, research, and portfolio demonstration purposes and does not constitute investment advice.
