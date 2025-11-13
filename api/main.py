"""
FastAPI backend for RAG Question Generator
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from generator.question_generator import QuestionGenerator
from generator.history import QuestionHistory

app = FastAPI(
    title="RAG Question Generator API",
    description="API for generating IIT JEE questions using RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generator (lazy loading)
_generator: Optional[QuestionGenerator] = None
_history: Optional[QuestionHistory] = None


def get_generator() -> QuestionGenerator:
    """Get or create question generator instance"""
    global _generator
    if _generator is None:
        _generator = QuestionGenerator()
    return _generator


def get_history() -> QuestionHistory:
    """Get or create history instance"""
    global _history
    if _history is None:
        _history = QuestionHistory()
    return _history


class QuestionResponse(BaseModel):
    """Response model for a single question"""
    id: str
    question: str
    options: Dict[str, str] = Field(default_factory=dict)
    correct_answer: str = ""
    explanation: str = ""
    subject: str = ""
    difficulty: str = ""
    topic: str = ""
    source_documents: List[Dict[str, str]] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    """Response model for question generation"""
    success: bool
    count: int
    questions: List[QuestionResponse]
    total_in_history: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    history_count: int


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    history = get_history()
    return HealthResponse(
        status="healthy",
        message="RAG Question Generator API is running",
        history_count=history.get_history_count()
    )


@app.get("/generate", response_model=GenerateResponse)
async def generate_questions(
    count: int = Query(default=10, ge=1, le=100, description="Number of questions to generate")
):
    """
    Generate unique questions
    
    - **count**: Number of questions to generate (1-100)
    """
    try:
        generator = get_generator()
        history = get_history()
        
        # Generate questions
        questions = generator.generate_questions(count=count)
        
        # Convert to response models
        question_responses = [
            QuestionResponse(**q) for q in questions
        ]
        
        return GenerateResponse(
            success=True,
            count=len(question_responses),
            questions=question_responses,
            total_in_history=history.get_history_count()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


@app.get("/history/count")
async def get_history_count():
    """Get total number of questions in history"""
    history = get_history()
    return {"count": history.get_history_count()}


@app.delete("/history")
async def clear_history():
    """Clear question history (use with caution)"""
    history = get_history()
    history.clear_history()
    return {"message": "History cleared successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await root()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

