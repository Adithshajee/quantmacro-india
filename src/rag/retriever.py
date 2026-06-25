import os
import sys
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.vectorstores import VectorStoreRetriever
from src.rag.embedder import load_vector_store, get_embeddings

def get_retriever(k: int = 5) -> VectorStoreRetriever | None:
    """
    Loads the FAISS vector store and returns it as a retriever configured to fetch k documents.
    Returns None if the vector store does not exist.
    """
    embeddings = get_embeddings()
    vectorstore = load_vector_store(embeddings)
    if vectorstore is not None:
        return vectorstore.as_retriever(search_kwargs={"k": k})
    return None

def retrieve_context(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the top k relevant chunks matching the query.
    Returns a list of dicts: {"content": chunk_text, "source": filename, "sector": sector, "score": float}
    """
    retriever = get_retriever(k=k)
    if retriever is None:
        return []
    
    # Fetch relevant documents using the retriever as required
    docs = retriever.get_relevant_documents(query)
    
    # Attempt to retrieve scores from the underlying vector store
    scores = {}
    try:
        vectorstore = retriever.vectorstore
        doc_scores = vectorstore.similarity_search_with_score(query, k=k)
        for d, s in doc_scores:
            scores[d.page_content] = float(s)
    except Exception:
        pass
        
    results = []
    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "UNKNOWN")
        sector = doc.metadata.get("sector", "GENERAL")
        score = scores.get(content, 0.0)
        results.append({
            "content": content,
            "source": source,
            "sector": sector,
            "score": score
        })
        
    return results

if __name__ == "__main__":
    print("--- Retriever Standalone Demo ---")
    
    query = "banking sector NPA outlook"
    print(f"Retrieving context for query: '{query}'")
    
    results = retrieve_context(query, k=5)
    
    if not results:
        print("No index found or no documents retrieved. Please run the embedder demo first to build the index.")
    else:
        print(f"Retrieved {len(results)} chunks:")
        for idx, res in enumerate(results):
            print(f"\nResult {idx + 1}:")
            print(f"Source: {res['source']}")
            print(f"Sector: {res['sector']}")
            print(f"Score (Distance): {res['score']:.4f}")
            print(f"Content: {res['content']}")
