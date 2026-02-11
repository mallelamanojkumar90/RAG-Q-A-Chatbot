"""
Helper script to run the data pipeline
"""
import sys
from pipeline.load_data import PDFProcessor, watch_data_folder

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        print("=" * 60)
        print("Starting data pipeline in WATCH MODE...")
        print("=" * 60)
        print("The pipeline will:")
        print("  1. Process any new or changed PDFs")
        print("  2. Skip already processed PDFs (using cache)")
        print("  3. Automatically process new PDFs when added to the data folder")
        print("\nPress Ctrl+C to stop watching\n")
        watch_data_folder()
    else:
        print("=" * 60)
        print("RAG Data Pipeline - Processing PDFs")
        print("=" * 60)
        print("The pipeline will:")
        print("  - Skip already processed PDFs (using cache)")
        print("  - Only process new or changed PDFs")
        print("  - Save processing status for future runs")
        print("=" * 60 + "\n")
        processor = PDFProcessor()
        processed = processor.process_all_pdfs()
        if processed == 0:
            print("\nNo new PDFs to process. All files are already processed!")
            print("To reprocess all files, delete the cache file: processed_files.json")
        else:
            print(f"\nSuccessfully processed {processed} new/changed PDF file(s)!")
        print("\nTo watch for new files automatically, run: py -3.11 run_pipeline.py --watch")

