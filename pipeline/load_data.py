"""
Data pipeline for loading PDFs, chunking, and storing embeddings
"""
import os
import hashlib
import re
import json
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


def _clean_text_for_processing(text: str) -> str:
    """Clean text by removing surrogate characters and other problematic characters"""
    if not isinstance(text, str):
        return text
    # Remove surrogate characters (U+D800 to U+DFFF) - these cannot be encoded in UTF-8
    text = re.sub(r'[\ud800-\udfff]', '', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    return text

# OCR imports (optional)
OCR_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pass


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
        self.skipped_files: Dict[str, str] = {}  # file_path -> reason
        self.failed_files: Dict[str, str] = {}  # file_path -> error message
        self.cache_file = Path(settings.processed_files_cache)
        self._load_processed_files_cache()
    
    def _load_processed_files_cache(self):
        """Load processed files cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.processed_files = json.load(f)
                print(f"Loaded {len(self.processed_files)} processed files from cache")
            except Exception as e:
                print(f"Warning: Could not load processed files cache: {e}")
                self.processed_files = {}
        else:
            self.processed_files = {}
    
    def _save_processed_files_cache(self):
        """Save processed files cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save processed files cache: {e}")
        
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
    
    def _extract_text_with_ocr(self, file_path: str) -> List[Document]:
        """Extract text from image-based PDF using OCR"""
        if not OCR_AVAILABLE:
            print(f"  [WARNING] OCR libraries not available. Install: pip install pdf2image pytesseract")
            print(f"  [INFO] Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            return []
        
        try:
            # Check if Tesseract is available, try to set path if needed
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                # Try to use custom path from config if provided
                if settings.tesseract_cmd and os.path.exists(settings.tesseract_cmd):
                    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                    try:
                        pytesseract.get_tesseract_version()
                        print(f"  [INFO] Using Tesseract from config: {settings.tesseract_cmd}")
                    except Exception:
                        pass
                else:
                    # Try common Windows installation paths
                    common_paths = [
                        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                        r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
                    ]
                    tesseract_found = False
                    for path in common_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            try:
                                pytesseract.get_tesseract_version()
                                print(f"  [INFO] Found Tesseract at: {path}")
                                tesseract_found = True
                                break
                            except Exception:
                                continue
                    
                    if not tesseract_found:
                        print(f"  [ERROR] Tesseract OCR not found. Please install it:")
                        print(f"  [INFO] Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                        print(f"  [INFO] Or set TESSERACT_CMD in .env file with the full path to tesseract.exe")
                        print(f"  [INFO] Example: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
                        return []
            
            print(f"  [INFO] Attempting OCR extraction...")
            print(f"  [INFO] This may take a while depending on PDF size and number of pages...")
            
            # Configure Poppler path if needed
            poppler_path = None
            if settings.poppler_path and os.path.exists(settings.poppler_path):
                poppler_path = settings.poppler_path
                print(f"  [INFO] Using Poppler from config: {poppler_path}")
            else:
                # First, try to find poppler in PATH
                import shutil
                pdftoppm_exe = shutil.which("pdftoppm")
                if pdftoppm_exe:
                    poppler_path = os.path.dirname(pdftoppm_exe)
                    print(f"  [INFO] Found Poppler in PATH: {poppler_path}")
                else:
                    # Try to find poppler in common Windows locations
                    # Poppler is typically in a bin subdirectory
                    common_poppler_paths = [
                        r"C:\poppler\Library\bin",  # Common conda/package manager location
                        r"C:\poppler\bin",
                        r"C:\Program Files\poppler\bin",
                        r"C:\Program Files (x86)\poppler\bin",
                        r"C:\Users\{}\AppData\Local\Programs\poppler\bin".format(os.getenv('USERNAME', '')),
                        # Also check if poppler binaries are directly in these paths
                        r"C:\poppler",
                        r"C:\Program Files\poppler",
                        r"C:\Program Files (x86)\poppler",
                    ]
                    for path in common_poppler_paths:
                        if os.path.exists(path):
                            # Check if pdftoppm.exe exists (one of the poppler utilities)
                            pdftoppm_path = os.path.join(path, "pdftoppm.exe")
                            if os.path.exists(pdftoppm_path):
                                poppler_path = path
                                print(f"  [INFO] Found Poppler at: {poppler_path}")
                                break
            
            # Convert PDF pages to images
            try:
                if poppler_path:
                    images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
                else:
                    images = convert_from_path(file_path, dpi=300)  # Try with PATH
            except Exception as e:
                if "PDFInfoNotInstalledError" in str(type(e).__name__) or "poppler" in str(e).lower():
                    print(f"  [ERROR] Poppler not found. Please install Poppler and:")
                    print(f"  [INFO] 1. Add it to your system PATH, OR")
                    print(f"  [INFO] 2. Set POPPLER_PATH in .env file with the path to the poppler bin directory")
                    print(f"  [INFO] Example: POPPLER_PATH=C:\\poppler\\bin")
                    print(f"  [INFO] Download from: https://github.com/oschwartz10612/poppler-windows/releases")
                    return []
                raise
            
            if not images:
                print(f"  [WARNING] Could not convert PDF pages to images")
                return []
            
            print(f"  [INFO] Converted {len(images)} pages to images")
            print(f"  [INFO] Performing OCR on {len(images)} pages (this may take several minutes)...")
            
            documents = []
            for page_num, image in enumerate(images, start=1):
                try:
                    # Show progress for OCR
                    print(f"  [OCR] Processing page {page_num}/{len(images)}...", end="\r")
                    
                    # Perform OCR on the image
                    text = pytesseract.image_to_string(
                        image, 
                        lang=settings.ocr_language
                    )
                    
                    if text and text.strip():
                        # Clean OCR text to remove surrogate characters
                        cleaned_text = _clean_text_for_processing(text.strip())
                        if cleaned_text:
                            # Create document from OCR text
                            doc = Document(
                                page_content=cleaned_text,
                                metadata={
                                    "source": file_path,
                                    "filename": Path(file_path).name,
                                    "page": page_num,
                                    "extraction_method": "ocr"
                                }
                            )
                            documents.append(doc)
                            print(f"  [OCR] Page {page_num}/{len(images)}: Extracted {len(cleaned_text)} characters")
                    else:
                        print(f"  [WARNING] Page {page_num}/{len(images)}: No text extracted via OCR")
                except Exception as e:
                    print(f"  [WARNING] OCR failed for page {page_num}/{len(images)}: {e}")
                    continue
            
            if documents:
                print(f"  [SUCCESS] OCR extracted text from {len(documents)} pages")
            else:
                print(f"  [WARNING] OCR did not extract any text from the PDF")
            
            return documents
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"  [ERROR] OCR extraction failed ({error_type}): {error_msg}")
            if "tesseract" in error_msg.lower() or "not found" in error_msg.lower():
                print(f"  [INFO] Tesseract OCR may not be installed or not in PATH")
                print(f"  [INFO] Install Tesseract: https://github.com/tesseract-ocr/tesseract")
            return []
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF and return documents. Uses OCR if no text is extracted."""
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                print(f"  [ERROR] File does not exist: {file_path}")
                return []
            
            if not os.access(file_path, os.R_OK):
                print(f"  [ERROR] File is not readable: {file_path}")
                return []
            
            # Try standard PDF text extraction first
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Check if any text was extracted
            has_text = False
            if documents:
                for doc in documents:
                    if doc.page_content and doc.page_content.strip():
                        has_text = True
                        break
            
            # If no text extracted and OCR is enabled, try OCR
            if not has_text and settings.enable_ocr:
                print(f"  [INFO] No text extracted with standard method")
                print(f"  [INFO] Attempting OCR extraction...")
                print(f"  [WARNING] OCR can take several minutes depending on PDF size!")
                ocr_documents = self._extract_text_with_ocr(file_path)
                if ocr_documents:
                    documents = ocr_documents
                    has_text = True
                    print(f"  [SUCCESS] OCR extraction completed successfully")
                else:
                    print(f"  [WARNING] OCR extraction did not produce any text")
            
            if not documents or not has_text:
                if settings.enable_ocr:
                    print(f"  [WARNING] PDF loaded but no text extracted (tried standard extraction and OCR)")
                else:
                    print(f"  [WARNING] PDF loaded but no text extracted. Enable OCR in .env: ENABLE_OCR=true")
                return []
            
            # Add metadata to all documents and clean text content
            for doc in documents:
                # Clean page content to remove surrogate characters
                if doc.page_content:
                    doc.page_content = _clean_text_for_processing(doc.page_content)
                
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
                if "filename" not in doc.metadata:
                    doc.metadata["filename"] = Path(file_path).name
                
                # Clean metadata string values
                for key, value in doc.metadata.items():
                    if isinstance(value, str):
                        doc.metadata[key] = _clean_text_for_processing(value)
            
            return documents
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"  [ERROR] Failed to load PDF ({error_type}): {error_msg}")
            # Provide more specific error messages for common issues
            if "encrypted" in error_msg.lower() or "password" in error_msg.lower():
                print(f"  [INFO] This PDF may be password-protected or encrypted")
            elif "corrupt" in error_msg.lower() or "invalid" in error_msg.lower():
                print(f"  [INFO] This PDF may be corrupted or in an unsupported format")
            return []
    
    def process_pdf(self, file_path: str) -> bool:
        """Process a single PDF file"""
        file_path = str(Path(file_path).absolute())
        filename = Path(file_path).name
        
        # Check if already processed
        if self._is_file_processed(file_path):
            reason = "Already processed (file hash matches previous processing)"
            self.skipped_files[filename] = reason
            print(f"[SKIP] {filename}: {reason}")
            return False
        
        print(f"\n[PROCESSING] {filename}")
        print(f"  Checking file...")
        
        # Load PDF
        documents = self.load_pdf(file_path)
        if not documents:
            reason = "No documents loaded (PDF may be corrupted, encrypted, or empty)"
            self.skipped_files[filename] = reason
            print(f"[SKIP] {filename}: {reason}")
            return False
        
        # Check extraction method used
        extraction_method = documents[0].metadata.get("extraction_method", "standard")
        if extraction_method == "ocr":
            print(f"  [OCR] Loaded {len(documents)} pages (using OCR extraction)")
        else:
            print(f"  Loaded {len(documents)} pages (standard text extraction)")
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"  Split into {len(chunks)} chunks")
        
        # Skip if no chunks (empty PDF or no extractable text)
        if not chunks:
            reason = "No text extracted (PDF may be image-based or have no extractable text)"
            self.skipped_files[filename] = reason
            print(f"[SKIP] {filename}: {reason}")
            return False
        
        # Store in vector database
        try:
            # Add unique IDs based on file and chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = f"{Path(file_path).stem}_{i}"
            
            upserted_ids = self.vector_store.add_documents(chunks)
            if upserted_ids:
                print(f"  [SUCCESS] Stored {len(upserted_ids)} chunks in vector database")
            else:
                reason = "No chunks were successfully stored (all chunks may have failed validation)"
                self.failed_files[filename] = reason
                print(f"  [WARNING] {reason}")
                return False
            
            # Mark as processed
            self.processed_files[file_path] = self._get_file_hash(file_path)
            # Save cache after each successful processing
            self._save_processed_files_cache()
            return True
        except Exception as e:
            error_msg = str(e)
            self.failed_files[filename] = error_msg
            print(f"  [ERROR] Failed to store embeddings: {error_msg}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup_cache(self, folder_path: Path):
        """Remove entries from cache for files that no longer exist"""
        if not folder_path.exists():
            return
        
        existing_files = {str(p.absolute()) for p in folder_path.glob("*.pdf")}
        cached_files = set(self.processed_files.keys())
        deleted_files = cached_files - existing_files
        
        if deleted_files:
            for deleted_file in deleted_files:
                del self.processed_files[deleted_file]
            print(f"Cleaned up {len(deleted_files)} deleted files from cache")
            self._save_processed_files_cache()
    
    def process_all_pdfs(self, folder_path: str = None) -> int:
        """Process all PDFs in the data folder"""
        if folder_path is None:
            folder_path = settings.data_folder
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Data folder {folder_path} does not exist")
            return 0
        
        # Clean up cache for deleted files
        self._cleanup_cache(folder_path)
        
        pdf_files = list(folder_path.glob("*.pdf"))
        total_files = len(pdf_files)
        
        # Count how many are already processed
        already_processed = 0
        for pdf_file in pdf_files:
            if self._is_file_processed(str(pdf_file)):
                already_processed += 1
        
        new_or_changed = total_files - already_processed
        
        print(f"\n{'='*60}")
        print(f"Found {total_files} PDF files")
        print(f"  - Already processed: {already_processed}")
        print(f"  - New or changed: {new_or_changed}")
        print(f"{'='*60}\n")
        
        if new_or_changed == 0:
            print("All PDFs are already processed. No action needed.")
            return 0
        
        processed_count = 0
        current_file = 0
        for pdf_file in pdf_files:
            current_file += 1
            filename = pdf_file.name
            print(f"\n[{current_file}/{total_files}] Processing: {filename}")
            if self.process_pdf(str(pdf_file)):
                processed_count += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total PDFs found: {total_files}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped: {len(self.skipped_files)}")
        print(f"Failed: {len(self.failed_files)}")
        
        if self.skipped_files:
            print(f"\nSkipped files ({len(self.skipped_files)}):")
            for filename, reason in self.skipped_files.items():
                print(f"  - {filename}: {reason}")
        
        if self.failed_files:
            print(f"\nFailed files ({len(self.failed_files)}):")
            for filename, error in self.failed_files.items():
                print(f"  - {filename}: {error}")
        
        print(f"{'='*60}\n")
        
        # Save cache after processing all files
        self._save_processed_files_cache()
        
        return processed_count


class DataFolderWatcher(FileSystemEventHandler):
    """Watch for changes in the data folder"""
    
    def __init__(self, processor: PDFProcessor):
        self.processor = processor
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"New PDF detected: {event.src_path}")
            if self.processor.process_pdf(event.src_path):
                # Cache is saved automatically in process_pdf
                print(f"Successfully processed new file: {event.src_path}")
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and event.src_path.endswith('.pdf'):
            print(f"PDF modified: {event.src_path}")
            # Remove from processed files to force reprocessing
            file_path = str(Path(event.src_path).absolute())
            if file_path in self.processor.processed_files:
                del self.processor.processed_files[file_path]
                self.processor._save_processed_files_cache()
            if self.processor.process_pdf(event.src_path):
                print(f"Successfully reprocessed modified file: {event.src_path}")


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

