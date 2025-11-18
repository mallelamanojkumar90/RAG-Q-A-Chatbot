"""
Vector store implementations for ChromaDB and Pinecone
"""
from typing import List, Optional
try:
    from langchain.vectorstores import VectorStore
    from langchain.embeddings.base import Embeddings
except ImportError:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings

# Try to import ChromaDB (optional, only needed if using chromadb)
Chroma = None
try:
    from langchain_chroma import Chroma
except ImportError:
    pass

# Try to import from langchain-pinecone first, fallback to langchain-community
PineconeVectorStore = None
Pinecone = None
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    try:
        from langchain_community.vectorstores import Pinecone
    except ImportError:
        pass

from config import settings


class CustomPineconeVectorStore(VectorStore):
    """
    Custom Pinecone vector store implementation for new Pinecone client API (v7+)
    This is a workaround when langchain-pinecone is not available.
    """
    
    def __init__(self, index, embedding_function: Embeddings, text_key: str = "text"):
        """Initialize custom Pinecone vector store"""
        self.index = index
        self.embedding_function = embedding_function
        self.text_key = text_key
    
    def _clean_text(self, text):
        """Clean and validate text for embedding"""
        if text is None:
            return None
        
        # Ensure text is a string
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return None
        
        # Handle encoding issues - decode if bytes, re-encode to utf-8
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8', errors='replace')
            except Exception:
                return None
        
        # Remove surrogate characters (U+D800 to U+DFFF) - these cannot be encoded in UTF-8
        # Surrogates are invalid Unicode characters that can cause encoding errors
        import re
        text = re.sub(r'[\ud800-\udfff]', '', text)
        
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        # Return None if empty after cleaning
        if not text:
            return None
        
        # Ensure text is not too long (sentence-transformers has limits)
        # Most models handle up to 512 tokens, but we'll be conservative
        max_length = 10000  # characters
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def add_documents(self, documents, **kwargs):
        """Add documents to the Pinecone index"""
        from langchain_core.documents import Document
        import uuid
        import numpy as np
        
        # Filter and clean documents before embedding
        valid_documents = []
        valid_texts = []
        
        for i, doc in enumerate(documents):
            # Clean and validate text
            cleaned_text = self._clean_text(doc.page_content)
            if cleaned_text is None:
                print(f"Warning: Document {i} has invalid or empty text, skipping...")
                continue
            
            valid_documents.append(doc)
            valid_texts.append(cleaned_text)
        
        if not valid_texts:
            print("Warning: No valid documents to embed")
            return []
        
        # Generate embeddings in batches to handle errors better
        vectors = []
        for i, (doc, text) in enumerate(zip(valid_documents, valid_texts)):
            try:
                # Generate embedding for this document
                embedding = self.embedding_function.embed_documents([text])[0]
            except Exception as e:
                print(f"Warning: Failed to generate embedding for document {i}: {e}")
                print(f"  Text preview: {text[:100]}...")
                continue
            # Generate unique ID
            doc_id = doc.metadata.get("chunk_id", str(uuid.uuid4()))
            
            # Validate embedding
            if not embedding or len(embedding) == 0:
                print(f"Warning: Empty embedding for document {i}, skipping...")
                continue
            
            # Check for NaN or invalid values in embedding
            if isinstance(embedding, list):
                try:
                    embedding = [float(x) for x in embedding]
                    if any(not np.isfinite(x) for x in embedding):
                        print(f"Warning: Invalid values in embedding for document {i}, skipping...")
                        continue
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert embedding to floats for document {i}: {e}")
                    continue
            
            # Prepare metadata (Pinecone metadata has size limits)
            # Pinecone metadata values must be strings, numbers, or booleans
            # Total metadata size limit is ~40KB per vector
            # Truncate text if too long (keep first 10000 chars to be safe)
            text_content = text  # Use the cleaned text
            max_text_length = 10000
            if len(text_content) > max_text_length:
                text_content = text_content[:max_text_length] + "...[truncated]"
            
            metadata = {self.text_key: text_content}
            
            # Add other metadata fields, filtering out invalid values
            for key, value in doc.metadata.items():
                # Skip None values and empty strings
                if value is None or value == "":
                    continue
                
                # Convert to string, but keep original type for numbers and booleans
                if isinstance(value, bool):
                    metadata[str(key)] = value
                elif isinstance(value, (int, float)):
                    # Check for NaN or infinity
                    if isinstance(value, float) and not np.isfinite(value):
                        continue
                    metadata[str(key)] = value
                elif isinstance(value, str):
                    # Remove surrogate characters and truncate long strings (max 1000 chars per metadata field)
                    value = self._clean_text(value) or ""  # Use _clean_text to remove surrogates
                    if len(value) > 1000:
                        value = value[:1000] + "...[truncated]"
                    if value:  # Only add non-empty strings
                        metadata[str(key)] = value
                else:
                    # Convert other types to string, but truncate
                    str_value = str(value)
                    # Remove surrogate characters
                    str_value = self._clean_text(str_value) or ""
                    if len(str_value) > 1000:
                        str_value = str_value[:1000] + "...[truncated]"
                    if str_value:  # Only add non-empty strings
                        metadata[str(key)] = str_value
            
            vectors.append({
                "id": str(doc_id),  # Ensure ID is string
                "values": embedding,
                "metadata": metadata
            })
        
        if not vectors:
            print("Warning: No valid vectors to upsert")
            return []
        
        # Upsert in batches (Pinecone limit is 100 vectors per batch)
        batch_size = 100
        upserted_ids = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                # Upsert batch
                self.index.upsert(vectors=batch)
                upserted_ids.extend([v["id"] for v in batch])
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                # Try upserting one by one to identify problematic vectors
                for vector in batch:
                    try:
                        self.index.upsert(vectors=[vector])
                        upserted_ids.append(vector["id"])
                    except Exception as single_error:
                        print(f"Error upserting vector {vector['id']}: {single_error}")
                        print(f"  Vector dimension: {len(vector['values'])}")
                        print(f"  Metadata keys: {list(vector['metadata'].keys())}")
        
        return upserted_ids
    
    def similarity_search(self, query: str, k: int = 4, **kwargs):
        """Search for similar documents"""
        from langchain_core.documents import Document
        
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Convert results to Document objects
        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get(self.text_key, "")
            documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
    
    def as_retriever(self, **kwargs):
        """Create a retriever from this vector store"""
        from langchain_core.retrievers import BaseRetriever
        from typing import Any
        
        class PineconeRetriever(BaseRetriever):
            """Custom retriever for Pinecone vector store"""
            
            vector_store: Any  # Store reference to vector store
            search_kwargs: dict = {}  # Search parameters
            
            class Config:
                arbitrary_types_allowed = True
            
            def __init__(self, vector_store, **search_kwargs):
                # Initialize with proper Pydantic handling
                super().__init__(
                    vector_store=vector_store,
                    search_kwargs=search_kwargs or {}
                )
            
            def _get_relevant_documents(self, query: str):
                """Retrieve relevant documents from the vector store"""
                k = self.search_kwargs.get("k", 4)
                return self.vector_store.similarity_search(query, k=k)
        
        return PineconeRetriever(vector_store=self, **kwargs)
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """Create vector store from texts (required by VectorStore interface)"""
        # This is a class method, but we need instance to add documents
        # This method is typically used for initialization, but we handle it differently
        raise NotImplementedError(
            "Use get_vector_store() function instead of from_texts() for CustomPineconeVectorStore"
        )


def get_embedding_dimension(embedding_function: Embeddings) -> int:
    """
    Get the dimension of embeddings by testing with a sample text.
    Common embedding dimensions:
    - all-MiniLM-L6-v2: 384
    - OpenAI text-embedding-3-small: 1536
    - OpenAI text-embedding-3-large: 3072
    - OpenAI text-embedding-ada-002: 1536
    """
    try:
        # Test with a sample text to get the dimension
        test_text = "test"
        embedding = embedding_function.embed_query(test_text)
        return len(embedding)
    except Exception as e:
        print(f"Warning: Could not determine embedding dimension automatically: {e}")
        # Fallback to common dimensions based on model name
        if hasattr(settings, 'embedding_model'):
            model_name = settings.embedding_model.lower()
            if "minilm" in model_name or "all-minilm" in model_name:
                return 384
            elif "text-embedding-3-small" in model_name:
                return 1536
            elif "text-embedding-3-large" in model_name:
                return 3072
            elif "ada-002" in model_name or "text-embedding-ada-002" in model_name:
                return 1536
        # Default to 384 for sentence-transformers
        print("Using default dimension: 384")
        return 384


def ensure_pinecone_index_exists(
    pc_client,
    index_name: str,
    dimension: int,
    metric: str = "cosine"
) -> None:
    """
    Ensure Pinecone index exists, create it if it doesn't.
    Checks for dimension mismatch and handles it.
    Supports both serverless and pod-based deployments.
    """
    try:
        # Check if index exists
        existing_indexes = [index.name for index in pc_client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
            
            # Try to create serverless index first (modern default)
            try:
                from pinecone import ServerlessSpec
                
                # Parse region from environment setting (e.g., "us-east-1-aws" -> "us-east-1")
                region = settings.pinecone_environment
                cloud = "aws"  # default
                
                if "-aws" in region:
                    region = region.replace("-aws", "")
                    cloud = "aws"
                elif "-gcp" in region:
                    region = region.replace("-gcp", "")
                    cloud = "gcp"
                elif "-azure" in region:
                    region = region.replace("-azure", "")
                    cloud = "azure"
                
                pc_client.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                print(f"Successfully created serverless index '{index_name}'")
            except Exception as e:
                # Fallback to pod-based index if serverless fails
                print(f"Serverless index creation failed: {e}")
                print("Attempting to create pod-based index...")
                try:
                    from pinecone import PodSpec
                    pc_client.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=PodSpec(
                            environment=settings.pinecone_environment,
                            pod_type="p1.x1"  # Default pod type
                        )
                    )
                    print(f"Successfully created pod-based index '{index_name}'")
                except Exception as pod_error:
                    raise ValueError(
                        f"Failed to create Pinecone index '{index_name}': {pod_error}. "
                        f"Please check your Pinecone API key and environment settings."
                    )
        else:
            # Index exists - check dimension mismatch
            try:
                index_info = pc_client.describe_index(index_name)
                existing_dimension = index_info.dimension
                
                if existing_dimension != dimension:
                    raise ValueError(
                        f"Dimension mismatch: Index '{index_name}' has dimension {existing_dimension}, "
                        f"but embedding model produces dimension {dimension}. "
                        f"\n\nOptions to fix:"
                        f"\n1. Delete the existing index and recreate it (will lose all data):"
                        f"\n   - Go to Pinecone dashboard: https://app.pinecone.io"
                        f"\n   - Delete index '{index_name}'"
                        f"\n   - Run this script again to create with correct dimension"
                        f"\n\n2. Use a different embedding model that produces {existing_dimension}-dimensional vectors"
                        f"\n   - For 1024 dimensions, consider using a different model"
                        f"\n   - Update EMBEDDING_MODEL in .env file"
                        f"\n\n3. Use a different index name:"
                        f"\n   - Set PINECONE_INDEX_NAME in .env to a new name"
                    )
                else:
                    print(f"Pinecone index '{index_name}' already exists with matching dimension {dimension}")
            except ValueError:
                # Re-raise ValueError (dimension mismatch) - this is a critical error
                raise
            except Exception as e:
                # If we can't check dimension, just warn but continue
                print(f"Pinecone index '{index_name}' already exists (could not verify dimension: {e})")
                print(f"Warning: Make sure the index dimension matches {dimension}")
    except ValueError:
        # Re-raise ValueError (dimension mismatch) - don't wrap it
        raise
    except Exception as e:
        raise ValueError(f"Error checking/creating Pinecone index: {e}")


def get_vector_store(embedding_function: Embeddings) -> VectorStore:
    """Get the appropriate vector store based on configuration"""
    if settings.vector_db == "chromadb":
        if Chroma is None:
            raise ImportError(
                "ChromaDB vector store not available. "
                "Please install: pip install langchain-chroma chromadb"
            )
        return Chroma(
            persist_directory=settings.chroma_db_path,
            embedding_function=embedding_function,
            collection_name="iit_jee_questions"
        )
    elif settings.vector_db == "pinecone":
        # Check if Pinecone client is available (required)
        try:
            from pinecone import Pinecone as PineconeClient
        except ImportError as e:
            raise ImportError(
                f"Pinecone client not installed or conflict detected: {e}. "
                "Please install: pip install pinecone>=5.1.0 "
                "And remove old package: pip uninstall pinecone-client"
            )
        
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        
        # Initialize Pinecone client with modern API
        pc_client = PineconeClient(api_key=settings.pinecone_api_key)
        
        # Get embedding dimension
        dimension = get_embedding_dimension(embedding_function)
        print(f"Detected embedding dimension: {dimension}")
        
        # Ensure index exists
        ensure_pinecone_index_exists(
            pc_client=pc_client,
            index_name=settings.pinecone_index_name,
            dimension=dimension,
            metric="cosine"
        )
        
        # Create or connect to the vector store
        if PineconeVectorStore is not None:
            # Use langchain-pinecone (newer API)
            # langchain-pinecone reads PINECONE_API_KEY from environment or can use the client
            import os
            # Ensure API key is in environment for langchain-pinecone
            if 'PINECONE_API_KEY' not in os.environ:
                os.environ['PINECONE_API_KEY'] = settings.pinecone_api_key
            
            return PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=embedding_function
            )
        else:
            # Use custom Pinecone wrapper (works with new Pinecone client API)
            # This is a fallback when langchain-pinecone is not available
            print("Using custom Pinecone vector store wrapper (langchain-pinecone not available)")
            index = pc_client.Index(settings.pinecone_index_name)
            return CustomPineconeVectorStore(
                index=index,
                embedding_function=embedding_function,
                text_key="text"
            )
    else:
        raise ValueError(f"Unknown vector database: {settings.vector_db}")

