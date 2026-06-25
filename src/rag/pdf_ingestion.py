import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def auto_detect_sector(text: str) -> str:
    """
    Scans the first 200 characters of a text chunk to detect the corresponding sector.
    """
    sample = text[:200].lower()
    
    it_keywords = ["infosys", "tcs", "wipro", "hcl", "tech mahindra", "it sector", "software"]
    banking_keywords = ["hdfc", "icici", "sbi", "kotak", "axis bank", "banking", "npa", "credit"]
    pharma_keywords = ["sun pharma", "cipla", "dr reddy", "pharma", "api", "drug"]
    auto_keywords = ["maruti", "tata motors", "bajaj", "hero", "m&m", "automobile"]
    energy_keywords = ["reliance", "ongc", "ntpc", "power", "crude", "energy"]
    
    if any(kw in sample for kw in it_keywords):
        return "IT"
    elif any(kw in sample for kw in banking_keywords):
        return "BANKING"
    elif any(kw in sample for kw in pharma_keywords):
        return "PHARMA"
    elif any(kw in sample for kw in auto_keywords):
        return "AUTO"
    elif any(kw in sample for kw in energy_keywords):
        return "ENERGY"
    
    return "GENERAL"

def load_bse_pdf(pdf_path: str) -> list[Document]:
    """
    Loads a BSE earnings PDF report, splits it into chunks, and enriches each chunk
    with metadata, including the source filename, report type, and auto-detected sector.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    
    filename = os.path.basename(pdf_path)
    for chunk in chunks:
        sector = auto_detect_sector(chunk.page_content)
        chunk.metadata.update({
            "source": filename,
            "type": "earnings_report",
            "sector": sector
        })
        
    return chunks

if __name__ == "__main__":
    print("--- PDF Ingestion Standalone Demo ---")
    
    # Create dummy 2-page text content
    page1_text = (
        "Infosys Q4 results review: The IT sector giant posted a resilient set of numbers. "
        "With strong pipeline growth in cloud transformation, Infosys, TCS, and Wipro are leading "
        "the software services sector. Demand for generative AI integrations in software is rising."
    )
    page2_text = (
        "HDFC Bank credit growth updates: The banking sector is witnessing stable credit trends. "
        "HDFC Bank, ICICI Bank, and SBI show healthy loan books. Realized NPAs have declined "
        "substantially this quarter, indicating strong credit quality across retail segments."
    )
    
    # Convert into mock documents
    dummy_docs = [
        Document(page_content=page1_text, metadata={"page": 1}),
        Document(page_content=page2_text, metadata={"page": 2})
    ]
    
    # Split using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(dummy_docs)
    
    # Add metadata manually to mock chunks
    dummy_filename = "mock_bse_earnings_report.pdf"
    for chunk in chunks:
        sector = auto_detect_sector(chunk.page_content)
        chunk.metadata.update({
            "source": dummy_filename,
            "type": "earnings_report",
            "sector": sector
        })
        
    print(f"Total chunks created: {len(chunks)}")
    for idx, chunk in enumerate(chunks):
        print(f"\nChunk {idx + 1} Metadata: {chunk.metadata}")
        print(f"Chunk {idx + 1} Snippet: {chunk.page_content[:100]}...")
