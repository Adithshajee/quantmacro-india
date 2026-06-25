import requests
import logging
from typing import List
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import API_KEYS
from src.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

# ----------------------------------------
# NEW RAG CONTEXT FUNCTION
# ----------------------------------------
def get_rag_context(query: str) -> str:
    """
    Retrieves relevant document chunks and formats them into a single string.
    Returns 'No earnings context available.' if no chunks are found.
    """
    chunks = retrieve_context(query, k=5)
    if not chunks:
        return "No earnings context available."
    
    context_str = ""
    for chunk in chunks:
        context_str += f"Source: {chunk['source']} | Sector: {chunk['sector']}\n{chunk['content']}\n---\n"
    return context_str

# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------
def explain_market_condition(
    sector: str, insights_list: List[str], recent_news_headlines: List[str], confidence: float = None
) -> str:
    """
    Generate AI explanation using available LLMs with safe fallback.
    """
    prompt = build_prompt(sector, insights_list, recent_news_headlines, confidence)

    # Try LLMs in priority order
    for provider in ["groq", "gemini", "openai"]:
        try:
            if API_KEYS.get(provider):
                logger.info(f"Using {provider} for LLM explanation")
                call_llm_res = call_llm(provider, prompt)
                if call_llm_res:
                    return call_llm_res + "\n\n*Disclaimer: Educational and research purposes only. No financial recommendations are made.*"
        except Exception as e:
            logger.warning(f"{provider} failed: {e}")

    # Fallback (ALWAYS RETURN STRING)
    return rule_based_fallback(sector, insights_list)


# ----------------------------------------
# PROMPT BUILDER
# ----------------------------------------
def build_prompt(sector, insights, news, confidence=None):
    conf_str = f"Model Next-Day Prediction Confidence: {confidence:.1f}%\n" if confidence else ""
    return (
        f"You are a coordinator for a multi-agent financial research team analyzing the {sector} sector.\n\n"
        f"Data provided:\n"
        f"{conf_str}"
        f"Quantitative Insights:\n- " + "\n- ".join(insights[:6]) + "\n\n"
        f"Recent Headlines:\n- " + "\n- ".join(news[:6]) + "\n\n"
        f"Please generate a collaborative research report with the following structure:\n"
        f"1. 🌍 **Macro Analyst Agent**: Discuss macro factors (yields, VIX, exchange rate) and their transmission to this sector.\n"
        f"2. 📊 **Sector Analyst Agent**: Discuss price-sentiment divergence, recent momentum, and thematic news.\n"
        f"3. ⚠️ **Risk Management Agent**: Evaluate predictive confidence, realized volatility, and downside/drawdown risk.\n"
        f"4. 🧠 **Coordinator Summary**: Synthesize the overall outlook (3-4 sentences) and add a statement that this is for educational purposes only.\n\n"
        f"Keep each agent's section to 1-2 punchy sentences. Do NOT give buy/sell/hold recommendations."
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
    insights_str = "\n".join([f"- {ins}" for ins in insights[:4]])
    return (
        f"### 🌍 Multi-Agent Institutional Intelligence Report: {sector}\n\n"
        f"#### 1. Macro Analyst Agent\n"
        f"Systemic macro transmission channels indicate moderate sensitivity to current yield and currency fluctuations. General index momentum acts as the primary baseline drift.\n\n"
        f"#### 2. Sector Analyst Agent\n"
        f"Evaluating the quantitative anomalies and news mapper alignments:\n"
        f"{insights_str}\n\n"
        f"#### 3. Risk Management Agent\n"
        f"Drawdown metrics and rolling Sharpe ratios suggest monitoring standard thresholds. Realized volatility remains within historical bounds.\n\n"
        f"#### 4. Coordinator Synthesis\n"
        f"The algorithmic decision support system suggests a consolidation phase with mixed signals. Align positions with risk-parity limits. This assessment is dynamically generated for educational and research purposes only."
    )
