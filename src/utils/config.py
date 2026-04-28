import os
from dotenv import load_dotenv

load_dotenv()

IS_CLOUD = os.getenv("STREAMLIT") == "true" or not os.name == "nt"

DB_PATH = "/tmp/project.db" if IS_CLOUD else os.getenv("DB_PATH", "data/project.db")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
TARGET_SECTORS = os.getenv("TARGET_SECTORS", "BSE_BANKEX,BSE_IT,BSE_ENERGY").split(",")

# LLM Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

API_KEYS = {
    "gemini": GEMINI_API_KEY,
    "groq": GROQ_API_KEY,
    "openai": OPENAI_API_KEY,
}
