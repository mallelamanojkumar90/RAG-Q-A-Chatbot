"""
Helper script to run the FastAPI server
"""
import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("API will be available at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

