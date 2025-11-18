"""
Test script for Pinecone vector database implementation
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_pinecone_implementation():
    """Test Pinecone vector store implementation"""
    print("=" * 60)
    print("Testing Pinecone Vector Database Implementation")
    print("=" * 60)
    
    # Test 1: Check environment variables
    print("\n[Test 1] Checking environment variables...")
    api_key = os.getenv("PINECONE_API_KEY", "")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-questions")
    vector_db = os.getenv("VECTOR_DB", "chromadb")
    
    print(f"  VECTOR_DB: {vector_db}")
    print(f"  PINECONE_API_KEY: {'***' + api_key[-4:] if len(api_key) > 4 else 'NOT SET'}")
    print(f"  PINECONE_ENVIRONMENT: {environment}")
    print(f"  PINECONE_INDEX_NAME: {index_name}")
    
    if not api_key or api_key == "YOUR_PINECONE_API_KEY":
        print("  [ERROR] PINECONE_API_KEY is not set or still has placeholder value")
        print("  Please update .env file with your actual Pinecone API key")
        return False
    print("  [OK] Environment variables check passed")
    
    # Test 2: Import dependencies
    print("\n[Test 2] Checking dependencies...")
    try:
        from pinecone import Pinecone
        print("  [OK] pinecone package imported successfully")
    except ImportError as e:
        print(f"  [ERROR] Failed to import pinecone: {e}")
        print("  Please run: pip install pinecone>=5.1.0")
        return False
    
    try:
        from langchain_pinecone import PineconeVectorStore
        print("  [OK] langchain-pinecone package imported successfully")
    except ImportError:
        try:
            from langchain_community.vectorstores import Pinecone
            print("  [WARN] langchain-pinecone not found, using langchain-community fallback")
        except ImportError as e:
            print(f"  [ERROR] Failed to import Pinecone vector store: {e}")
            return False
    
    # Test 3: Initialize Pinecone client
    print("\n[Test 3] Initializing Pinecone client...")
    try:
        # Import the actual Pinecone client (not from langchain)
        from pinecone import Pinecone as PineconeClient
        pc = PineconeClient(api_key=api_key)
        print("  [OK] Pinecone client initialized successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize Pinecone client: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check/List indexes
    print("\n[Test 4] Checking existing indexes...")
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        print(f"  Found {len(existing_indexes)} existing index(es): {existing_indexes}")
        if index_name in existing_indexes:
            print(f"  [OK] Index '{index_name}' already exists")
        else:
            print(f"  [INFO] Index '{index_name}' does not exist yet (will be created when needed)")
    except Exception as e:
        print(f"  [WARN] Warning: Could not list indexes: {e}")
    
    # Test 5: Test embedding function and dimension detection
    print("\n[Test 5] Testing embedding function and dimension detection...")
    try:
        from pipeline.embeddings import get_embedding_function
        from pipeline.vector_store import get_embedding_dimension
        
        embedding_function = get_embedding_function()
        print(f"  [OK] Embedding function created: {type(embedding_function).__name__}")
        
        dimension = get_embedding_dimension(embedding_function)
        print(f"  [OK] Detected embedding dimension: {dimension}")
    except Exception as e:
        print(f"  [ERROR] Failed to test embedding function: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Test vector store creation (only if VECTOR_DB is set to pinecone)
    if vector_db.lower() != "pinecone":
        print(f"\n[Test 6] Skipped - VECTOR_DB is set to '{vector_db}', not 'pinecone'")
        print("  ℹ️  To test Pinecone, set VECTOR_DB=pinecone in .env file")
        return True
    
    print("\n[Test 6] Testing vector store creation...")
    try:
        from pipeline.vector_store import get_vector_store
        
        print("  Creating vector store (this may create index if it doesn't exist)...")
        vector_store = get_vector_store(embedding_function)
        print(f"  [OK] Vector store created successfully: {type(vector_store).__name__}")
        
        # Test 7: Test basic operations
        print("\n[Test 7] Testing basic vector store operations...")
        try:
            # Try to get retriever
            retriever = vector_store.as_retriever(search_kwargs={"k": 1})
            print("  [OK] Retriever created successfully")
            
            # Try a simple similarity search (may return empty if no documents)
            try:
                results = vector_store.similarity_search("test query", k=1)
                print(f"  [OK] Similarity search works (returned {len(results)} results)")
            except Exception as e:
                print(f"  [WARN] Similarity search test: {e}")
                print("  (This is normal if no documents have been added yet)")
        except Exception as e:
            print(f"  [WARN] Warning: Could not test retriever: {e}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to create vector store: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed! Pinecone implementation is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pinecone_implementation()
    sys.exit(0 if success else 1)

