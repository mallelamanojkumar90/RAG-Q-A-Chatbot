"""
Embedding functions for different providers
"""
import re
from typing import Callable
try:
    from langchain.embeddings.base import Embeddings
except ImportError:
    from langchain_core.embeddings import Embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings


def _remove_surrogate_characters(text: str) -> str:
    """Remove surrogate characters (U+D800 to U+DFFF) that cannot be encoded in UTF-8"""
    if not isinstance(text, str):
        return text
    return re.sub(r'[\ud800-\udfff]', '', text)


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
        """Embed a list of documents"""
        # Validate and clean texts before embedding
        cleaned_texts = []
        for i, text in enumerate(texts):
            if text is None:
                print(f"Warning: Text at index {i} is None, skipping...")
                continue
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception:
                    print(f"Warning: Text at index {i} cannot be converted to string, skipping...")
                    continue
            # Remove null bytes, surrogate characters, and normalize
            text = text.replace('\x00', '')
            text = _remove_surrogate_characters(text)
            text = text.strip()
            if text:  # Only add non-empty texts
                cleaned_texts.append(text)
        
        if not cleaned_texts:
            return []
        
        try:
            return self.embeddings.embed_documents(cleaned_texts)
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            print(f"Number of texts: {len(cleaned_texts)}")
            if cleaned_texts:
                print(f"First text preview: {cleaned_texts[0][:100]}")
            raise
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text"""
        if text is None:
            raise ValueError("Query text cannot be None")
        if not isinstance(text, str):
            text = str(text)
        # Clean text - remove null bytes and surrogate characters
        text = text.replace('\x00', '')
        text = _remove_surrogate_characters(text)
        text = text.strip()
        if not text:
            raise ValueError("Query text cannot be empty")
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

