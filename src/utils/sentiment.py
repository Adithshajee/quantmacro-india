import os
import logging
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self, use_finbert=True, model_name="ProsusAI/finbert", cache_dir="./models"):
        self.use_finbert = use_finbert
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.finbert_pipeline = None
        self.vader_analyzer = SentimentIntensityAnalyzer()

        if self.use_finbert:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                
                device = 0 if torch.cuda.is_available() else -1
                device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
                logger.info(f"Initializing FinBERT ({model_name}) on {device_name} (Cache: {cache_dir})...")
                
                os.makedirs(cache_dir, exist_ok=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
                
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
                logger.info("FinBERT initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}. Falling back to VADER sentiment.")
                self.use_finbert = False

    def analyze(self, text: str) -> tuple:
        """
        Analyzes the text and returns (sentiment_label, sentiment_score).
        sentiment_score ranges from -1.0 (most negative) to 1.0 (most positive).
        """
        if not text or not text.strip():
            return "neutral", 0.0

        if self.use_finbert and self.finbert_pipeline:
            try:
                # Run batch-friendly inference on truncated text (max 512 tokens)
                result = self.finbert_pipeline(text[:2000])[0]
                label = result['label'].lower()  # 'positive', 'negative', or 'neutral'
                score = result['score']          # Confidence probability [0, 1]

                # Map probability score to a -1 to +1 range for comparison metrics
                if label == 'positive':
                    mapped_score = score
                elif label == 'negative':
                    mapped_score = -score
                else:
                    mapped_score = 0.0
                    
                return label, mapped_score
            except Exception as e:
                logger.error(f"FinBERT inference error: {e}. Falling back to VADER.")

        # VADER Fallback
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
            
        return label, compound
