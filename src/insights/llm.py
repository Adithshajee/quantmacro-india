import requests
import logging
from typing import List
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
# PROMPT BUILDER (CLEAN SEPARATION)
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
        f"Keep it concise (3–4 sentences)."
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
            return response.json()["candidates"][0]["content"]["parts"][0][
                "text"
            ].strip()
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
            logger.warning(f"LLM API error: {response.status_code}")
    except Exception as e:
        logger.warning(f"Response parsing failed: {e}")
    return None


# ----------------------------------------
# FALLBACK (SMARTER VERSION)
# ----------------------------------------
def rule_based_fallback(sector, insights):
    sentiment_signal = any("bullish" in i.lower() for i in insights)
    risk_signal = any("risk" in i.lower() or "bearish" in i.lower() for i in insights)

    if sentiment_signal and not risk_signal:
        recommendation = "BUY"
        tone = "positive momentum"
    elif risk_signal:
        recommendation = "SELL"
        tone = "increasing downside risk"
    else:
        recommendation = "HOLD"
        tone = "mixed signals"

    return (
        f"The {sector} sector currently shows {tone} based on internal indicators. "
        f"{' '.join(insights[:2])}. "
        f"Overall recommendation: {recommendation}. "
        f"(Rule-based fallback – add API key for advanced AI analysis)"
    )
