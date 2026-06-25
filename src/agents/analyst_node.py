import os
import sys
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agents.state import AgentState

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def analyst_node(state: AgentState) -> AgentState:
    """
    Synthesizes RAG contexts and ML/sentiment quantitative signals using Gemini.
    """
    try:
        # Build prompt using f-string exactly as specified
        prompt = f"""SYSTEM: You are a senior BSE sector analyst at an institutional fund. 
You synthesize quantitative signals and fundamental research into clear, actionable sector views.
Always cite your evidence. Never fabricate data.

USER:
Query: {state["query"]}
Sector: {state["sector"] or "Multi-sector"}

=== QUANTITATIVE SIGNALS ===
ML Model Direction: {state["ml_signal"].get("direction", "N/A")}
Model Confidence: {state["ml_signal"].get("probability", 0.5):.1%}
News Sentiment Score: {state["news_sentiment"]:.3f} (range -1 to +1)
Key Features Used: {", ".join(state["ml_signal"].get("features_used", [])[:5])}

=== EARNINGS & FILING CONTEXT (RAG) ===
{chr(10).join([f"[{i+1}] {c['source']} ({c['sector']}): {c['content'][:300]}" for i, c in enumerate(state["retrieved_chunks"][:4])])}

Provide:
1. SECTOR VIEW (1 sentence: Bullish / Neutral / Bearish + reason)
2. KEY EVIDENCE (2-3 bullet points from the context above)
3. RISK FACTORS (1-2 specific risks)
4. CONFIDENCE: HIGH / MEDIUM / LOW (based on context quality)"""

        # Retrieve Google API key
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Call ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        
        response = llm.invoke(prompt)
        response_text = response.content
        
        # Parse CONFIDENCE from response text
        match = re.search(r"CONFIDENCE[:\*\s]*(HIGH|MEDIUM|LOW)", response_text, re.IGNORECASE)
        confidence = match.group(1).upper() if match else "LOW"
        
        # Extract sources (preserve duplicates for counting occurrences in UI)
        sources = [c["source"] for c in state["retrieved_chunks"] if "source" in c]
        
        state["answer"] = response_text
        state["sources"] = sources
        state["confidence"] = confidence

    except Exception as e:
        state["answer"] = f"Analysis unavailable — LLM error: {str(e)}"
        state["confidence"] = "LOW"
        state["sources"] = []
        
    return state
