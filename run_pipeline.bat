@echo off
REM Helper script to run the data pipeline on Windows
if "%1"=="--watch" (
    echo Starting data pipeline in watch mode...
    py -3.11 run_pipeline.py --watch
) else (
    echo Processing all PDFs in data folder...
    py -3.11 run_pipeline.py
)

