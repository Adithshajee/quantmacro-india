import sqlite3
import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH
from src.database.init_db import initialize_database

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    initialize_database(conn) # Ensure schema exists every time we connect
    return conn

@st.cache_resource
def get_streamlit_connection():
    return get_connection()
