import os
import sys
import argparse
import glob

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.rag.pdf_ingestion import load_bse_pdf
from src.rag.embedder import get_embeddings, load_vector_store, embed_documents

def main():
    parser = argparse.ArgumentParser(description="Ingest BSE Earnings PDFs into FAISS Vector Store")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="./data/reports/",
        help="Directory containing BSE earnings PDF reports (default: ./data/reports/)"
    )
    args = parser.parse_args()

    # Ensure pdf_dir exists
    if not os.path.exists(args.pdf_dir):
        print(f"Creating reports directory at: {args.pdf_dir}")
        os.makedirs(args.pdf_dir, exist_ok=True)

    # Scan for PDF files (case-insensitive)
    pdf_files = []
    for ext in ("*.pdf", "*.PDF"):
        pdf_files.extend(glob.glob(os.path.join(args.pdf_dir, ext)))

    if not pdf_files:
        print(f"No PDF files found in directory: {args.pdf_dir}")
        # Load index to report current size even if no new files are added
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings)
        total_vectors = vectorstore.index.ntotal if vectorstore is not None else 0
        print(f"Index updated. Total vectors: {total_vectors}")
        return

    print(f"Found {len(pdf_files)} PDF report(s) in {args.pdf_dir}. Starting ingestion...")

    embeddings = get_embeddings()
    vectorstore = load_vector_store(embeddings)

    for pdf_path in pdf_files:
        report_name = os.path.basename(pdf_path)
        try:
            # load pdf and chunk it
            chunks = load_bse_pdf(pdf_path)
            num_chunks = len(chunks)
            
            if num_chunks == 0:
                print(f"Processing {report_name}... added 0 chunks (empty file)")
                continue

            # incremental indexing
            if vectorstore is None:
                vectorstore = embed_documents(chunks, embeddings)
            else:
                vectorstore.add_documents(chunks)
                # save updated index
                index_path = "./data/faiss_index/"
                vectorstore.save_local(index_path)
                
            print(f"Processing {report_name}... added {num_chunks} chunks")
        except Exception as e:
            print(f"Error processing {report_name}: {e}")

    total_vectors = vectorstore.index.ntotal if vectorstore is not None else 0
    print(f"Index updated. Total vectors: {total_vectors}")

if __name__ == "__main__":
    main()
