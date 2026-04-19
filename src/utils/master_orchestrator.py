import os
import sys
import subprocess
import requests
import sqlite3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DB_PATH, TARGET_SECTORS

def run_step(step_name, cmd):
    print(f"STEP: {step_name}", flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(line, flush=True)
    process.wait()
    if process.returncode != 0:
        print(f"ERROR: failed at {step_name}", flush=True)
        sys.exit(1)

def clean_database():
    print("STEP: Scrubbing Database", flush=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(TARGET_SECTORS))
    
    cursor.execute(f"DELETE FROM bse_sector_prices WHERE sector_index NOT IN ({placeholders})", tuple(TARGET_SECTORS))
    deleted_prices = cursor.rowcount
    
    cursor.execute(f"DELETE FROM news_sector_mapping WHERE sector_index NOT IN ({placeholders})", tuple(TARGET_SECTORS))
    deleted_mappings = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"Scrubbed {deleted_prices} invalid prices and {deleted_mappings} invalid mappings.", flush=True)

def main():
    clean_database()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    scripts = [
        ("Ingestion (News)", ["python", os.path.join(base_dir, "ingestion", "fetch_news.py")]),
        ("Ingestion (BSE Data)", ["python", os.path.join(base_dir, "ingestion", "fetch_bse_data.py")]),
        ("Mapping Sectors", ["python", os.path.join(base_dir, "mapping", "sector_mapper.py")]),
        ("Calculating AI Sentiment", ["python", os.path.join(base_dir, "modeling", "sentiment_engine.py")]),
        ("Re-training Model", ["python", os.path.join(base_dir, "modeling", "train_model.py")])
    ]
    
    for name, cmd in scripts:
        run_step(name, cmd)
        
    print("STEP: Reloading API Model", flush=True)
    try:
        resp = requests.post("http://127.0.0.1:8000/reload", timeout=5)
        if resp.status_code == 200:
            print("API Model Reloaded successfully.", flush=True)
        else:
            print(f"FAILED to reload API model: {resp.status_code}", flush=True)
    except Exception as e:
        print(f"FAILED to reach API for reload: {e}", flush=True)

if __name__ == "__main__":
    main()
