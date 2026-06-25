import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Global singleton cached embeddings
_embeddings_singleton = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings singleton instance using
    sentence-transformers/all-MiniLM-L6-v2. Cache directory is set to
    ./models/embedding_cache/ to prevent repeated downloads.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        cache_dir = "./models/embedding_cache/"
        os.makedirs(cache_dir, exist_ok=True)
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=cache_dir
        )
    return _embeddings_singleton

def embed_documents(docs: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Builds a FAISS index from the provided document list, saves it locally
    to ./data/faiss_index/, and returns the vectorstore object.
    """
    vectorstore = FAISS.from_documents(docs, embeddings)
    index_path = "./data/faiss_index/"
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS | None:
    """
    Loads and returns the FAISS vector store from ./data/faiss_index/ if it exists.
    Returns None otherwise.
    """
    index_path = "./data/faiss_index/"
    # FAISS files generated are index.faiss and index.pkl
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")
    
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return None

if __name__ == "__main__":
    print("--- Embedder Standalone Demo ---")
    
    # Define 5 fake documents
    fake_docs = [
        Document(
            page_content="Infosys software export revenue grew 5% in constant currency. IT sector outlook is stable.",
            metadata={"source": "tcs_q4.pdf", "type": "earnings_report", "sector": "IT"}
        ),
        Document(
            page_content="HDFC credit growth is strong, led by home loans. Banking sector NPA numbers are healthy.",
            metadata={"source": "hdfc_q4.pdf", "type": "earnings_report", "sector": "BANKING"}
        ),
        Document(
            page_content="Sun Pharma launched a new generic drug for chronic diseases. Pharma R&D remains high.",
            metadata={"source": "sun_q4.pdf", "type": "earnings_report", "sector": "PHARMA"}
        ),
        Document(
            page_content="Maruti Suzuki passenger vehicle sales went up by 8% year-over-year. Auto demand is robust.",
            metadata={"source": "maruti_q4.pdf", "type": "earnings_report", "sector": "AUTO"}
        ),
        Document(
            page_content="Reliance Industries reports higher refining margins. Energy sector crude prices are steady.",
            metadata={"source": "reliance_q4.pdf", "type": "earnings_report", "sector": "ENERGY"}
        ),
    ]
    
    # Initialize embeddings
    print("Initializing embeddings model...")
    embeddings = get_embeddings()
    print("Embeddings loaded successfully.")
    
    # Embed documents
    print("Building FAISS index...")
    db = embed_documents(fake_docs, embeddings)
    print("FAISS index saved successfully.")
    
    # Search simulation
    query = "IT sector outlook"
    print(f"\nRunning similarity search for: '{query}'")
    results = db.similarity_search(query, k=2)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
