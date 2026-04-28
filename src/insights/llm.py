import requests
import logging
from typing import List
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import API_KEYS

logger = logging.getLogger(__name__)

# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------
def explain_market_condition(
    sector: str, insights_list: List[str], recent_news_headlines: List[str]
) -> str:
    """
    Generate AI explanation using available LLMs with safe fallback.
    """
    prompt = build_prompt(sector, insights_list, recent_news_headlines)

    # Try LLMs in priority order
    for provider in ["groq", "gemini", "openai"]:
        try:
            if API_KEYS.get(provider):
                logger.info(f"Using {provider} for LLM explanation")
                response = call_llm(provider, prompt)
                if response:
                    return response
        except Exception as e:
            logger.warning(f"{provider} failed: {e}")

    # Fallback (ALWAYS RETURN STRING)
    return rule_based_fallback(sector, insights_list)


# ----------------------------------------
# PROMPT BUILDER
# ----------------------------------------
def build_prompt(sector, insights, news):
    return (
        f"You are a professional financial analyst.\n\n"
        f"Sector: {sector}\n\n"
        f"Quantitative Insights:\n- " + "\n- ".join(insights[:5]) + "\n\n"
        f"Recent News:\n- " + "\n- ".join(news[:5]) + "\n\n"
        f"Task:\n"
        f"1. Summarize market condition\n"
        f"2. Identify trend (bullish/bearish/neutral)\n"
        f"3. Give a short recommendation (BUY/HOLD/SELL)\n\n"
        f"Keep it concise (3-4 sentences)."
    )

# ----------------------------------------
# LLM ROUTER
# ----------------------------------------
def call_llm(provider, prompt):
    if provider == "groq":
        return call_groq(prompt)
    elif provider == "gemini":
        return call_gemini(prompt)
    elif provider == "openai":
        return call_openai(prompt)

# ----------------------------------------
# GROQ
# ----------------------------------------
def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEYS['groq']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
        },
        timeout=10,
    )
    return safe_extract(response)

# ----------------------------------------
# GEMINI
# ----------------------------------------
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={API_KEYS['gemini']}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=10,
    )
    try:
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            logger.warning(f"Gemini API Error: {response.status_code}")
    except Exception as e:
        logger.warning(f"Gemini parsing failed: {e}")
    return None

# ----------------------------------------
# OPENAI
# ----------------------------------------
def call_openai(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {API_KEYS['openai']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
        },
        timeout=10,
    )
    return safe_extract(response)

# ----------------------------------------
# SAFE RESPONSE PARSER
# ----------------------------------------
def safe_extract(response):
    try:
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.warning(f"LLM API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.warning(f"Response parsing failed: {e}")
    return None

# ----------------------------------------
# FALLBACK
# ----------------------------------------
def rule_based_fallback(sector, insights):
    return "AI insights currently unavailable"
