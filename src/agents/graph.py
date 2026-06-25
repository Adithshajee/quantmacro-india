import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.retriever_node import retriever_node
from src.agents.quant_node import quant_node
from src.agents.analyst_node import analyst_node

# Define StateGraph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("retriever", retriever_node)
graph.add_node("quant", quant_node)
graph.add_node("analyst", analyst_node)

# Add linear flow edges
graph.set_entry_point("retriever")
graph.add_edge("retriever", "quant")
graph.add_edge("quant", "analyst")
graph.add_edge("analyst", END)

# Compile
agent_app = graph.compile()

def run_analysis(query: str, sector: str = "") -> dict:
    """
    Initializes state and runs the compiled LangGraph flow for the query and sector.
    """
    initial_state = AgentState(
        query=query,
        sector=sector,
        retrieved_chunks=[],
        ml_signal={},
        news_sentiment=0.0,
        answer="",
        sources=[],
        confidence="LOW",
        error=""
    )
    result = agent_app.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "ml_direction": result["ml_signal"].get("direction", "N/A"),
        "ml_probability": result["ml_signal"].get("probability", 0.5),
        "news_sentiment": result["news_sentiment"],
        "error": result.get("error", "")
    }

if __name__ == "__main__":
    print("--- LangGraph Agent System Standalone Demo ---")
    
    # Try to load environment variables from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    query = "banking sector NPA outlook"
    sector = "Banking"
    
    print(f"Running run_analysis for Sector: '{sector}' | Query: '{query}'...")
    res = run_analysis(query=query, sector=sector)
    
    print("\n--- RESULTS ---")
    print(f"Confidence: {res['confidence']}")
    print(f"ML Direction: {res['ml_direction']}")
    print(f"ML Probability: {res['ml_probability']:.2f}")
    print(f"News Sentiment: {res['news_sentiment']:.4f}")
    print(f"Sources: {res['sources']}")
    print(f"Error: {res['error']}")
    print("\nAnswer Summary:")
    print(res["answer"])
