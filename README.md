# RAG-Based IIT JEE Question Generator Chatbot

A full-stack RAG (Retrieval-Augmented Generation) system that generates unique IIT JEE questions from PDF question papers.

## 🎯 Features

- **PDF Processing**: Automatically loads and processes PDFs from the `/data` folder
- **Auto-reprocessing**: Watches for new PDFs and reprocesses automatically
- **Vector Database**: Stores embeddings in ChromaDB (with Pinecone option)
- **Unique Questions**: Generates distinct, non-repetitive questions
- **Question Tracking**: Tracks previously generated questions using UUID-based history
- **FastAPI Backend**: RESTful API with `/generate` endpoint
- **Streamlit Frontend**: Simple web interface for question generation
- **Configurable**: Supports multiple LLM providers (OpenAI, Gemini, Local) and embedding models

## 📋 Prerequisites

- Python 3.8+
- pip or conda

## 🚀 Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
EMBEDDING_PROVIDER=sentence_transformers
VECTOR_DB=chromadb
```

## 📁 Project Structure

```
RAG Q&A Chatbot/
├── data/                   # Place your PDF files here
├── pipeline/
│   ├── load_data.py       # PDF processing and data pipeline
│   ├── embeddings.py      # Embedding function providers
│   └── vector_store.py    # Vector database implementations
├── generator/
│   ├── question_generator.py  # RAG-based question generation
│   ├── llm_provider.py        # LLM provider implementations
│   └── history.py             # Question history tracking
├── api/
│   └── main.py            # FastAPI backend
├── frontend/
│   └── streamlit_app.py   # Streamlit frontend
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Usage

### 1. Prepare Your Data

Place your IIT JEE question paper PDFs in the `data/` folder:
```bash
mkdir data
# Copy your PDF files to data/
```

### 2. Process PDFs

Run the data pipeline to process PDFs:

**On Windows:**
```bash
# Process all PDFs once
run_pipeline.bat
# OR
py run_pipeline.py

# Process and watch for new files
run_pipeline.bat --watch
# OR
py run_pipeline.py --watch
```

**On Linux/Mac:**
```bash
# Process all PDFs once
python run_pipeline.py

# Process and watch for new files
python run_pipeline.py --watch
```

This will:
- Load all PDFs from the `data/` folder
- Split them into chunks
- Generate embeddings
- Store in ChromaDB

### 3. Start the FastAPI Backend

**On Windows:**
```bash
run_api.bat
# OR
py run_api.py
# OR
py -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**On Linux/Mac:**
```bash
python run_api.py
# OR
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Use the API

**Generate questions**:
```bash
curl "http://localhost:8000/generate?count=50"
```

**Check health**:
```bash
curl "http://localhost:8000/health"
```

**Get history count**:
```bash
curl "http://localhost:8000/history/count"
```

### 5. Use the Streamlit Frontend

**On Windows:**
```bash
run_frontend.bat
# OR
py run_frontend.py
# OR
py -m streamlit run frontend/streamlit_app.py
```

**On Linux/Mac:**
```bash
python run_frontend.py
# OR
streamlit run frontend/streamlit_app.py
```

Open your browser to `http://localhost:8501`

## ⚙️ Configuration

Edit `.env` file to configure:

### LLM Providers
- `LLM_PROVIDER`: `openai`, `gemini`, or `local`
- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Google Gemini API key

### Embeddings
- `EMBEDDING_PROVIDER`: `sentence_transformers` or `openai`
- `EMBEDDING_MODEL`: Model name (for sentence-transformers)
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model name

### Vector Database
- `VECTOR_DB`: `chromadb` or `pinecone`
- `CHROMA_DB_PATH`: Path to ChromaDB storage
- `PINECONE_API_KEY`: Pinecone API key (if using Pinecone)

### Data Processing
- `DATA_FOLDER`: Path to PDF folder (default: `./data`)
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## 📊 API Endpoints

### `GET /`
Health check endpoint

### `GET /generate?count=50`
Generate unique questions
- **Parameters**:
  - `count` (int): Number of questions to generate (1-100)
- **Response**: JSON with generated questions

### `GET /history/count`
Get total number of questions in history

### `DELETE /history`
Clear question history (use with caution)

### `GET /health`
Health check with history count

## 🔍 How It Works

1. **Data Pipeline**:
   - Watches the `/data` folder for PDF files
   - Loads PDFs using PyPDFLoader
   - Splits text into chunks using RecursiveCharacterTextSplitter
   - Generates embeddings using configured provider
   - Stores in vector database (ChromaDB/Pinecone)

2. **Question Generation**:
   - Retrieves relevant context chunks using semantic search
   - Uses RAG pipeline with LLM to generate questions
   - Ensures uniqueness by checking question history
   - Returns structured questions with metadata

3. **History Tracking**:
   - Stores question IDs in SQLite database
   - Prevents duplicate question generation
   - Tracks generation metadata

## 🛠️ Troubleshooting

### PDFs not processing
- Check that PDFs are in the `data/` folder
- Ensure PDFs are not corrupted
- Check console for error messages

### API connection errors
- Make sure FastAPI server is running
- Check that port 8000 is not in use
- Verify API_URL in streamlit_app.py

### No questions generated
- Ensure PDFs have been processed
- Check that vector database has data
- Verify LLM API keys are set correctly

### Memory issues
- Reduce `CHUNK_SIZE` in config
- Process fewer PDFs at once
- Use smaller embedding models

## 📝 Notes

- First run will download embedding models (sentence-transformers)
- Question generation may take time depending on LLM provider
- ChromaDB data persists in `./chroma_db` folder
- Question history is stored in `./question_history.db`

## 🚢 Deployment

### Local Deployment
The application runs locally by default. All data is stored in local directories.

### Cloud Deployment
For production deployment:
1. Set up environment variables on your hosting platform
2. Use Pinecone for vector database (if needed)
3. Configure proper CORS settings
4. Set up proper authentication for API endpoints

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

