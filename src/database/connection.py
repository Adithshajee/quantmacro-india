import sqlite3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH
from src.database.init_db import init_db

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if not os.path.exists(DB_PATH):
        init_db()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def get_streamlit_connection():
    import streamlit as st
    @st.cache_resource
    def _create_persistent_connection():
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        if not os.path.exists(DB_PATH):
            init_db()
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    return _create_persistent_connection()
