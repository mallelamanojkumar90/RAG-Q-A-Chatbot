"""
Data pipeline for loading PDFs, chunking, and storing embeddings
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from config import settings
from pipeline.embeddings import get_embedding_function
from pipeline.vector_store import get_vector_store


class PDFProcessor:
    """Process PDF files and store embeddings"""
    
    def __init__(self):
        self.embedding_function = get_embedding_function()
        self.vector_store = get_vector_store(self.embedding_function)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self.processed_files: Dict[str, str] = {}  # file_path -> hash
        
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_file_processed(self, file_path: str) -> bool:
        """Check if file has been processed"""
        if file_path not in self.processed_files:
            return False
        current_hash = self._get_file_hash(file_path)
        return self.processed_files[file_path] == current_hash
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF and return documents"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = file_path
                doc.metadata["filename"] = Path(file_path).name
            
            return documents
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
    
    def process_pdf(self, file_path: str) -> bool:
        """Process a single PDF file"""
        file_path = str(Path(file_path).absolute())
        
        # Check if already processed
        if self._is_file_processed(file_path):
            print(f"File {file_path} already processed, skipping...")
            return False
        
        print(f"Processing PDF: {file_path}")
        
        # Load PDF
        documents = self.load_pdf(file_path)
        if not documents:
            print(f"No documents loaded from {file_path}")
            return False
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Skip if no chunks (empty PDF or no extractable text)
        if not chunks:
            print(f"Warning: No text extracted from {file_path}, skipping...")
            return False
        
        # Store in vector database
        try:
            # Add unique IDs based on file and chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{Path(file_path).stem}_{i}"
            
            self.vector_store.add_documents(chunks)
            print(f"Stored {len(chunks)} chunks in vector database")
            
            # Mark as processed
            self.processed_files[file_path] = self._get_file_hash(file_path)
            return True
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False
    
    def process_all_pdfs(self, folder_path: str = None) -> int:
        """Process all PDFs in the data folder"""
        if folder_path is None:
            folder_path = settings.data_folder
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Data folder {folder_path} does not exist")
            return 0
        
        pdf_files = list(folder_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        processed_count = 0
        for pdf_file in pdf_files:
            if self.process_pdf(str(pdf_file)):
                processed_count += 1
        
        return processed_count


class DataFolderWatcher(FileSystemEventHandler):
    """Watch for changes in the data folder"""
    
    def __init__(self, processor: PDFProcessor):
        self.processor = processor
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"New PDF detected: {event.src_path}")
            self.processor.process_pdf(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"PDF modified: {event.src_path}")
            # Remove from processed files to force reprocessing
            if event.src_path in self.processor.processed_files:
                del self.processor.processed_files[event.src_path]
            self.processor.process_pdf(event.src_path)


def watch_data_folder(processor: PDFProcessor = None):
    """Start watching the data folder for changes"""
    if processor is None:
        processor = PDFProcessor()
    
    # Process existing files first
    print("Processing existing PDFs...")
    processor.process_all_pdfs()
    
    # Set up file watcher
    event_handler = DataFolderWatcher(processor)
    observer = Observer()
    observer.schedule(event_handler, settings.data_folder, recursive=False)
    observer.start()
    
    print(f"Watching {settings.data_folder} for new PDFs...")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Run the data pipeline
    processor = PDFProcessor()
    
    # Process all existing PDFs
    print("=" * 50)
    print("RAG Data Pipeline - Processing PDFs")
    print("=" * 50)
    processed = processor.process_all_pdfs()
    print(f"\nProcessed {processed} PDF files")
    
    # Optionally start watching for new files
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        watch_data_folder(processor)

