# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

**On Windows:**
```bash
install_requirements.bat
# OR
py -m pip install -r requirements.txt
```

**On Linux/Mac:**
```bash
pip install -r requirements.txt
# OR
python -m pip install -r requirements.txt
```

### Step 2: Create Environment File

Create a `.env` file in the project root with the following content:

```env
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Configuration (use sentence_transformers for free local embeddings)
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database (use chromadb for local storage)
VECTOR_DB=chromadb
CHROMA_DB_PATH=./chroma_db

# Data Configuration
DATA_FOLDER=./data
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Question Generation
MAX_QUESTIONS_PER_REQUEST=100
TEMPERATURE=0.7
```

**Note**: For local testing without API keys, use:
- `EMBEDDING_PROVIDER=sentence_transformers` (free, runs locally)
- `LLM_PROVIDER=local` (requires Ollama setup) OR use OpenAI/Gemini with API keys

### Step 3: Add PDF Files

Place your IIT JEE question paper PDFs in the `data/` folder:

```bash
# Create data folder if it doesn't exist
mkdir data

# Copy your PDF files
cp /path/to/your/*.pdf data/
```

### Step 4: Process PDFs

**On Windows:**
```bash
# Process all PDFs once
run_pipeline.bat

# OR process and watch for new files
run_pipeline.bat --watch

# OR use Python launcher directly
py run_pipeline.py
py run_pipeline.py --watch
```

**On Linux/Mac:**
```bash
# Process all PDFs once
python run_pipeline.py

# OR process and watch for new files
python run_pipeline.py --watch
```

### Step 5: Start the API Server

**On Windows:**
```bash
run_api.bat
# OR
py run_api.py
```

**On Linux/Mac:**
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Step 6: Generate Questions

**Option A: Using the API directly**
```bash
curl "http://localhost:8000/generate?count=10"
```

**Option B: Using the Streamlit Frontend**

**On Windows:**
```bash
run_frontend.bat
# OR
py run_frontend.py
```

**On Linux/Mac:**
```bash
python run_frontend.py
```

Then open `http://localhost:8501` in your browser.

## 📝 Example Workflow

1. **Add PDFs**: Copy IIT JEE question papers to `data/` folder
2. **Process**: Run `python run_pipeline.py` to create embeddings
3. **Generate**: Use API or frontend to generate questions
4. **Watch Mode**: Use `--watch` flag to auto-process new PDFs

## 🔧 Configuration Options

### Using OpenAI (Recommended for best results)
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Using Google Gemini
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
```

### Using Local Models (Ollama)
```env
LLM_PROVIDER=local
# Requires Ollama installed and running
```

### Using Pinecone (Cloud Vector DB)
```env
VECTOR_DB=pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=rag-questions
```

## 🐛 Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not set"
- Create `.env` file with your API key
- Or use `sentence_transformers` for embeddings (free)

### "No PDFs found"
- Check that PDFs are in the `data/` folder
- Ensure files have `.pdf` extension

### "No questions generated"
- Make sure PDFs have been processed first
- Check that vector database has data
- Verify LLM API keys are correct

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize prompts in `generator/question_generator.py`
- Adjust chunk size in `.env` for better results
- Add more PDFs to increase question variety

