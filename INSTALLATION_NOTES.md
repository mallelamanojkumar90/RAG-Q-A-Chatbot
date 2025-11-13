# Installation Notes

## Python 3.14 Compatibility Issues

You're using Python 3.14, which is very new. Some packages may not have compatible wheels yet.

### Current Status

✅ **Installed:**
- LangChain and related packages
- FastAPI, Uvicorn
- PDF processing (pypdf)
- Watchdog
- Basic dependencies

⚠️ **Partially Installed:**
- ChromaDB (installed but missing some dependencies)
- Sentence Transformers (not yet installed - large download)
- Streamlit (not yet installed - dependency conflicts)

❌ **Missing Dependencies:**
- `onnxruntime` - Required by ChromaDB but no Python 3.14 wheel available yet
- `pulsar-client` - Required by ChromaDB 0.4.22 but not available for Python 3.14

### Workarounds

#### Option 1: Use Python 3.11 or 3.12 (Recommended)
Python 3.11 or 3.12 have better package compatibility:

```bash
# Install Python 3.11 or 3.12, then:
py -3.11 -m pip install -r requirements.txt
```

#### Option 2: Install Missing Packages Manually
Try installing onnxruntime from source or use a pre-release version:

```bash
py -m pip install onnxruntime --pre
```

#### Option 3: Use Alternative Vector Database
You can modify the code to use Pinecone (cloud-based) instead of ChromaDB, which doesn't require onnxruntime.

### Quick Test

To test if basic functionality works (without ChromaDB):

1. Comment out ChromaDB imports in `pipeline/vector_store.py`
2. Use a mock vector store for testing
3. Install sentence-transformers separately: `py -m pip install sentence-transformers`

### Next Steps

1. **For now**: The core LangChain functionality should work
2. **For ChromaDB**: Consider using Python 3.11/3.12 or wait for Python 3.14 wheels
3. **For production**: Use Python 3.11 or 3.12 for better stability

### Testing Without Full Installation

You can test the PDF loading and text splitting without the vector database:

```python
from pipeline.load_data import PDFProcessor
processor = PDFProcessor()
# This will fail at vector store creation, but you can test PDF loading separately
```

