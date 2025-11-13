"""
Vector store implementations for ChromaDB and Pinecone
"""
from typing import List
try:
    from langchain.vectorstores import VectorStore
    from langchain.embeddings.base import Embeddings
except ImportError:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings

from langchain_chroma import Chroma
from langchain_community.vectorstores import Pinecone

from config import settings


def get_vector_store(embedding_function: Embeddings) -> VectorStore:
    """Get the appropriate vector store based on configuration"""
    if settings.vector_db == "chromadb":
        return Chroma(
            persist_directory=settings.chroma_db_path,
            embedding_function=embedding_function,
            collection_name="iit_jee_questions"
        )
    elif settings.vector_db == "pinecone":
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        
        import pinecone
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        
        return Pinecone.from_existing_index(
            index_name=settings.pinecone_index_name,
            embedding=embedding_function
        )
    else:
        raise ValueError(f"Unknown vector database: {settings.vector_db}")

