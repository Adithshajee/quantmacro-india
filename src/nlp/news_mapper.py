import logging
import numpy as np

logger = logging.getLogger(__name__)

# Sector profiles with key semantic descriptors for embedding similarity
SECTOR_PROFILES = {
    "BSE_BANKEX": "banking sector finance RBI banks interest rates credit loan deposits NPA fintech credit cards lending financial policy",
    "BSE_IT": "information technology software computers AI cloud chips coding services TCS Infosys tech companies software developer semiconductor digital",
    "BSE_ENERGY": "energy sector oil gas electricity power renewable green solar wind thermal petrol coal crude refining power grid green hydrogen",
    "BSE_SENSEX": "general economy stock market Sensex macroeconomics inflation global markets GDP trade corporate earnings BSE index"
}

class SemanticNewsMapper:
    def __init__(self, use_embeddings=True, cache_dir="./models"):
        self.use_embeddings = use_embeddings
        self.model = None
        self.sector_embeddings = {}
        self.sectors = list(SECTOR_PROFILES.keys())
        self.sector_descriptions = list(SECTOR_PROFILES.values())

        if self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Initializing SentenceTransformer (all-MiniLM-L6-v2) for semantic news routing...")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
                
                # Precompute sector embeddings
                embeddings = self.model.encode(self.sector_descriptions, convert_to_numpy=True)
                for sector, emb in zip(self.sectors, embeddings):
                    self.sector_embeddings[sector] = emb / np.linalg.norm(emb) # Normalized
                    
                logger.info("SentenceTransformer initialized and sector profiles pre-computed.")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}. Falling back to keyword-based routing.")
                self.use_embeddings = False

    def map_headline_to_sector(self, headline: str) -> tuple:
        """
        Maps a news headline to the most relevant market sector.
        Returns: (sector_index, mapping_reason)
        """
        if not headline or not headline.strip():
            return "BSE_SENSEX", "Default (empty headline)"

        # If semantic embeddings are loaded, use cosine similarity
        if self.use_embeddings and self.model is not None:
            try:
                headline_emb = self.model.encode(headline, convert_to_numpy=True)
                headline_emb = headline_emb / np.linalg.norm(headline_emb) # Normalize
                
                best_sector = "BSE_SENSEX"
                best_similarity = -1.0
                
                for sector, emb in self.sector_embeddings.items():
                    sim = np.dot(headline_emb, emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_sector = sector
                
                # Use a threshold to prevent irrelevant matching. 
                # If similarity is too low, default to general macro index (SENSEX)
                threshold = 0.22
                if best_similarity >= threshold:
                    return best_sector, f"Semantic similarity ({best_similarity:.2f})"
                else:
                    return "BSE_SENSEX", f"Low semantic similarity ({best_similarity:.2f}), routed to macro"
            except Exception as e:
                logger.error(f"Semantic mapping error: {e}. Falling back to keywords.")

        # Robust keyword-based fallback
        headline_lower = headline.lower()
        
        bank_keywords = ["bank", "finance", "rbi", "loan", "lending", "interest rate", "npa", "hdfc", "sbi", "icici"]
        it_keywords = ["tech", "software", "it ", "cloud", "ai ", "tcs", "infosys", "wipro", "computer", "semiconductor"]
        energy_keywords = ["energy", "oil", "gas", "power", "solar", "wind", "coal", "petrol", "reliance", "ongc", "ntpc"]

        bank_score = sum(1 for kw in bank_keywords if kw in headline_lower)
        it_score = sum(1 for kw in it_keywords if kw in headline_lower)
        energy_score = sum(1 for kw in energy_keywords if kw in headline_lower)

        max_score = max(bank_score, it_score, energy_score)
        
        if max_score > 0:
            if max_score == bank_score:
                return "BSE_BANKEX", "Keyword match (Banking)"
            elif max_score == it_score:
                return "BSE_IT", "Keyword match (IT)"
            elif max_score == energy_score:
                return "BSE_ENERGY", "Keyword match (Energy)"

        return "BSE_SENSEX", "Keyword match (General Macro)"
