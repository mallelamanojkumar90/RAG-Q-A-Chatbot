# RAG Q&A Chatbot - Pinecone Vector Database Implementation

## Background and Motivation

The user wants to implement Pinecone vector database support in the RAG Q&A Chatbot project. Currently, the project has:
- Basic Pinecone code in `pipeline/vector_store.py` but using outdated API
- Configuration settings for Pinecone in `config.py`
- The code assumes the Pinecone index already exists
- Uses old `pinecone.init()` API which is deprecated

**Goal**: Update and properly implement Pinecone vector database integration with:
- Modern Pinecone API (using `pinecone-client` v3+)
- Automatic index creation if it doesn't exist
- Proper dimension detection based on embedding model
- Support for both ChromaDB and Pinecone as vector stores

## Key Challenges and Analysis

1. **API Changes**: Pinecone has updated their API - need to use `pinecone.Pinecone()` instead of `pinecone.init()`
2. **Index Creation**: Need to handle index creation with proper dimensions based on embedding model
3. **Dimension Detection**: Different embedding models have different dimensions (e.g., all-MiniLM-L6-v2 = 384, OpenAI text-embedding-3-small = 1536)
4. **Environment Configuration**: Pinecone now uses different environment/region settings
5. **LangChain Integration**: Ensure compatibility with LangChain's Pinecone integration

## High-level Task Breakdown

### Task 1: Update Pinecone Client and Dependencies
- **Success Criteria**: 
  - Verify `pinecone-client` version in requirements.txt is compatible
  - Check if `langchain-pinecone` package is needed (newer LangChain versions)
- **Files to modify**: `requirements.txt`

### Task 2: Update vector_store.py with Modern Pinecone API
- **Success Criteria**:
  - Replace `pinecone.init()` with `pinecone.Pinecone()` client
  - Add function to get embedding dimension dynamically
  - Add function to create index if it doesn't exist
  - Update `Pinecone.from_existing_index()` to handle both existing and new indexes
  - Handle Pinecone serverless vs pod-based deployments
- **Files to modify**: `pipeline/vector_store.py`

### Task 3: Update Configuration for Pinecone
- **Success Criteria**:
  - Verify all Pinecone settings are properly configured
  - Add dimension configuration option (optional, auto-detect preferred)
  - Update environment variable documentation
- **Files to modify**: `config.py` (if needed)

### Task 4: Test Pinecone Integration
- **Success Criteria**:
  - Test index creation
  - Test document addition
  - Test similarity search
  - Verify it works with both embedding providers (sentence_transformers and OpenAI)
- **Files to test**: `pipeline/vector_store.py`, `pipeline/load_data.py`, `generator/question_generator.py`

## Project Status Board

- [x] Task 1: Update Pinecone Client and Dependencies
- [x] Task 2: Update vector_store.py with Modern Pinecone API
- [x] Task 3: Update Configuration for Pinecone
- [ ] Task 4: Test Pinecone Integration

## Current Status / Progress Tracking

**Status**: Implementation completed. All code changes have been made.

### Completed Tasks:

1. **Updated requirements.txt**:
   - Changed `pinecone-client>=3.1.0` to `pinecone>=5.1.0`
   - Added `langchain-pinecone>=0.1.0` for better LangChain integration

2. **Updated pipeline/vector_store.py**:
   - Replaced deprecated `pinecone.init()` with modern `Pinecone(api_key=...)` client
   - Added `get_embedding_dimension()` function to automatically detect embedding dimensions
   - Added `ensure_pinecone_index_exists()` function to create index if it doesn't exist
   - Supports both serverless (default) and pod-based Pinecone deployments
   - Added proper error handling and fallback mechanisms
   - Updated to use `langchain-pinecone` package with fallback to `langchain-community`

3. **Configuration**:
   - Verified `config.py` already has all necessary Pinecone settings
   - No changes needed to config.py

### Key Features Implemented:

- **Automatic Dimension Detection**: The code automatically detects embedding dimensions by testing the embedding function
- **Index Auto-Creation**: Creates Pinecone index if it doesn't exist with proper dimensions
- **Serverless Support**: Defaults to serverless indexes (modern Pinecone approach)
- **Pod-based Fallback**: Falls back to pod-based indexes if serverless creation fails
- **Region Parsing**: Automatically parses region from environment setting (e.g., "us-east-1-aws")
- **Backward Compatibility**: Falls back to `langchain-community` if `langchain-pinecone` is not available

## Executor's Feedback or Assistance Requests

**Implementation Complete**: All code changes have been implemented. The user should:

1. Install updated dependencies: `pip install -r requirements.txt`
2. Set environment variables:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_ENVIRONMENT`: Region (e.g., "us-east-1-aws" for serverless)
   - `PINECONE_INDEX_NAME`: Name for your index (default: "rag-questions")
   - `VECTOR_DB=pinecone`: To use Pinecone instead of ChromaDB

3. Test the implementation by running the pipeline or question generator

**Note**: The linter shows a warning about `langchain_pinecone` import, but this is expected until the package is installed. The code handles this gracefully with try/except blocks.

### Testing Results:

**Test Status**: Partially successful
- ✅ Pinecone client initialization: Working
- ✅ Index listing/checking: Working  
- ✅ Embedding dimension detection: Working (detected 384 for all-MiniLM-L6-v2)
- ⚠️ Vector store creation: Requires `langchain-pinecone` package
  - `langchain-community` Pinecone integration is deprecated and not compatible with new Pinecone client API (v7+)
  - `langchain-pinecone` has dependency constraints (requires Python < 3.13)
  - Current Python version: 3.14 (incompatible with langchain-pinecone)

**Recommendation**: 
- For Python 3.13+: Use ChromaDB as the vector store, or wait for langchain-pinecone to support Python 3.14
- For Python < 3.13: Install langchain-pinecone for best Pinecone integration
- The code gracefully falls back to ChromaDB if Pinecone is not available

### Additional Updates:

4. **Created .env file**:
   - Added Pinecone environment variables template
   - User needs to replace `YOUR_PINECONE_API_KEY` with their actual API key
   - Set `VECTOR_DB=pinecone` to enable Pinecone usage

## Lessons

- Pinecone API has changed from `pinecone.init()` to `pinecone.Pinecone()` client pattern
- Need to handle embedding dimensions dynamically based on the embedding model used
- LangChain's Pinecone integration may require `langchain-pinecone` package in newer versions
- **Surrogate characters (U+D800 to U+DFFF) cannot be encoded in UTF-8** - must be removed before encoding operations. Added comprehensive text cleaning in:
  - `pipeline/vector_store.py`: `_clean_text()` method removes surrogates before storing in Pinecone
  - `pipeline/embeddings.py`: Clean text before embedding operations
  - `pipeline/load_data.py`: Clean text when loading PDFs and extracting via OCR
  - All metadata string values are also cleaned to prevent encoding errors

