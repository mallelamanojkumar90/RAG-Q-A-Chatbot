@echo off
REM Helper script to run the FastAPI server on Windows
echo Starting FastAPI server...
echo API will be available at http://localhost:8000
echo API docs at http://localhost:8000/docs
py -3.11 run_api.py

