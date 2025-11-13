"""
Embedding functions for different providers
"""
from typing import Callable
try:
    from langchain.embeddings.base import Embeddings
except ImportError:
    from langchain_core.embeddings import Embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for sentence-transformers embeddings"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.embedding_model
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)


def get_embedding_function() -> Embeddings:
    """Get the appropriate embedding function based on configuration"""
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )
    elif settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbeddings()
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

