import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.state import AgentState
from src.rag.retriever import retrieve_context

def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves document chunks relevant to the query and optionally filters them by sector.
    """
    try:
        query = state.get("query", "")
        sector = state.get("sector", "")
        
        # Retrieve context (k=5)
        chunks = retrieve_context(query, k=5)
        
        # Filter chunks if sector is provided
        if sector and sector.strip():
            # Filter where chunk["sector"] == state["sector"] OR chunk["sector"] == "GENERAL"
            filtered_chunks = []
            target_sec = sector.strip()
            for chunk in chunks:
                chunk_sec = chunk.get("sector", "GENERAL")
                if (chunk_sec == target_sec or 
                    chunk_sec == "GENERAL" or 
                    chunk_sec.upper() == target_sec.upper() or 
                    chunk_sec.upper() == "GENERAL"):
                    filtered_chunks.append(chunk)
            state["retrieved_chunks"] = filtered_chunks
        else:
            state["retrieved_chunks"] = chunks
            
    except Exception as e:
        state["error"] = str(e)
        state["retrieved_chunks"] = []
        
    return state
