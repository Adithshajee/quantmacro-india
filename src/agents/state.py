from typing import TypedDict

class AgentState(TypedDict):
    query: str
    sector: str
    retrieved_chunks: list
    ml_signal: dict
    news_sentiment: float
    answer: str
    sources: list
    confidence: str
    error: str
