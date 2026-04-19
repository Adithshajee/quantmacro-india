from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "/app/data/project.db")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "ProsusAI/finbert")
API_HOST = os.getenv("API_HOST", "http://127.0.0.1:8000")

if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY missing")

TARGET_SECTORS = os.getenv("TARGET_SECTORS", "BSE_BANKEX,BSE_IT,BSE_ENERGY").split(",")
