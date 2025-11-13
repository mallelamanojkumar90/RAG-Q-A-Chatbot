"""
Configuration management for RAG Question Generator
"""
import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # LLM Configuration
    llm_provider: Literal["openai", "gemini", "local"] = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Embedding Configuration
    embedding_provider: Literal["sentence_transformers", "openai"] = os.getenv(
        "EMBEDDING_PROVIDER", "sentence_transformers"
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Vector Database Configuration
    vector_db: Literal["chromadb", "pinecone"] = os.getenv("VECTOR_DB", "chromadb")
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "rag-questions")
    
    # Data Configuration
    data_folder: str = os.getenv("DATA_FOLDER", "./data")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Question Generation
    max_questions_per_request: int = int(os.getenv("MAX_QUESTIONS_PER_REQUEST", "100"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # History Configuration
    history_db_path: str = os.getenv("HISTORY_DB_PATH", "./question_history.db")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure data folder exists
Path(settings.data_folder).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_db_path).mkdir(parents=True, exist_ok=True)

