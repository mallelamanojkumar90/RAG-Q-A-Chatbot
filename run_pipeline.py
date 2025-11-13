"""
Helper script to run the data pipeline
"""
import sys
from pipeline.load_data import PDFProcessor, watch_data_folder

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        print("Starting data pipeline in watch mode...")
        watch_data_folder()
    else:
        print("Processing all PDFs in data folder...")
        processor = PDFProcessor()
        processed = processor.process_all_pdfs()
        print(f"\nProcessed {processed} PDF files successfully!")
        print("\nTo watch for new files, run: py -3.11 run_pipeline.py --watch")

