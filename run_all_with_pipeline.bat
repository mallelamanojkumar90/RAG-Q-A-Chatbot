@echo off
REM Helper script to run Frontend, Backend API, and Data Pipeline on Windows
echo ========================================
echo Starting RAG Q&A Chatbot (Full Stack)
echo ========================================
echo.
echo Starting Data Pipeline (Watch Mode)...
start "Data Pipeline" cmd /k "echo Data Pipeline - Watching for new PDFs && echo. && py -3.11 run_pipeline.py --watch"
echo.
echo Waiting 2 seconds...
timeout /t 2 /nobreak >nul
echo.
echo Starting FastAPI Backend...
start "FastAPI Backend" cmd /k "echo FastAPI Backend - http://localhost:8000 && echo API Docs - http://localhost:8000/docs && echo. && py -3.11 run_api.py"
echo.
echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul
echo.
echo Starting Streamlit Frontend...
start "Streamlit Frontend" cmd /k "echo Streamlit Frontend - http://localhost:8501 && echo. && py -3.11 run_frontend.py"
echo.
echo ========================================
echo Full Stack Application Started!
echo ========================================
echo.
echo Data Pipeline: Watching data/ folder
echo Backend API:   http://localhost:8000
echo API Docs:      http://localhost:8000/docs
echo Frontend:      http://localhost:8501
echo.
echo Press any key to stop all services...
pause >nul
echo.
echo Stopping all services...
taskkill /FI "WindowTitle eq Data Pipeline*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq FastAPI Backend*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Streamlit Frontend*" /T /F >nul 2>&1
echo All services stopped.
