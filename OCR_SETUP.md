# OCR Setup Guide for Image-Based PDFs

This guide explains how to set up OCR (Optical Character Recognition) to extract text from image-based or scanned PDFs.

## What is OCR?

OCR allows the system to extract text from PDFs that contain images of text (scanned documents) rather than actual text layers. When a PDF has no extractable text, the system will automatically attempt OCR if enabled.

## Installation Steps

### Step 1: Install Python Packages

The required Python packages are already in `requirements.txt`. Install them:

```bash
pip install pdf2image pytesseract Pillow
```

### Step 2: Install Tesseract OCR

Tesseract is the OCR engine that does the actual text recognition. You need to install it separately:

#### Windows:

1. **Download Tesseract installer:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)

2. **Install Tesseract:**
   - Run the installer
   - **Important**: During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)
   - Make sure to check "Add to PATH" during installation, OR add it manually:
     - Add `C:\Program Files\Tesseract-OCR` to your system PATH environment variable

3. **Verify installation:**
   ```bash
   tesseract --version
   ```

#### Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### macOS:

```bash
brew install tesseract
```

### Step 3: Install Poppler (for pdf2image)

`pdf2image` requires Poppler to convert PDF pages to images:

#### Windows:

1. Download Poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the zip file
3. Add the `bin` folder to your system PATH (e.g., `C:\poppler\Library\bin`)

#### Linux (Ubuntu/Debian):

```bash
sudo apt-get install poppler-utils
```

#### macOS:

```bash
brew install poppler
```

### Step 4: Configure OCR Settings

In your `.env` file, you can configure OCR:

```env
# Enable/disable OCR (default: true)
ENABLE_OCR=true

# OCR language (default: eng for English)
# For multiple languages, use: eng+hin (English + Hindi)
OCR_LANGUAGE=eng
```

### Step 5: Test OCR

Run the pipeline and check if OCR works:

```bash
py run_pipeline.py
```

If OCR is working, you'll see messages like:
```
[INFO] No text extracted, attempting OCR...
[INFO] Attempting OCR extraction (this may take a while)...
[INFO] Converted 10 pages to images, performing OCR...
[OCR] Page 1: Extracted 1234 characters
[SUCCESS] OCR extracted text from 10 pages
```

## Supported Languages

Tesseract supports many languages. Common language codes:
- `eng` - English
- `hin` - Hindi
- `spa` - Spanish
- `fra` - French
- `deu` - German
- `chi_sim` - Chinese (Simplified)
- `jpn` - Japanese

For multiple languages, use `+` to combine: `eng+hin` (English + Hindi)

To install additional language packs:
- **Windows**: Download from Tesseract installer or https://github.com/tesseract-ocr/tessdata
- **Linux**: `sudo apt-get install tesseract-ocr-[lang]` (e.g., `tesseract-ocr-hin`)
- **macOS**: Language packs are usually included

## Troubleshooting

### Error: "TesseractNotFoundError" or "tesseract is not installed"

**Solution**: Make sure Tesseract is installed and in your PATH. On Windows, you may need to restart your terminal/IDE after adding to PATH.

### Error: "pdf2image.exceptions.PDFInfoNotInstalledError"

**Solution**: Install Poppler and add it to PATH (see Step 3 above).

### OCR is slow

**Solution**: OCR processing is slower than text extraction. For large PDFs, this is normal. You can:
- Reduce DPI in the code (currently 300) for faster processing (lower accuracy)
- Process PDFs in smaller batches

### OCR accuracy is poor

**Solutions**:
- Ensure PDF images are high quality
- Use higher DPI (currently 300, can increase to 400-600 for better accuracy)
- Pre-process images (not currently implemented, but can be added)

## Disabling OCR

If you don't need OCR, you can disable it in `.env`:

```env
ENABLE_OCR=false
```

This will skip OCR processing and only use standard text extraction.








