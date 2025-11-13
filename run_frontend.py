"""
Helper script to run the Streamlit frontend
"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting Streamlit frontend...")
    print("Frontend will be available at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"])

