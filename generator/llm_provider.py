"""
LLM provider implementations for different services
"""
from typing import List, Dict, Any
try:
    from langchain.llms.base import LLM
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
except ImportError:
    from langchain_core.language_models.llms import BaseLLM as LLM
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Lazy imports to prevent startup crashes
ChatOpenAI = None
ChatGoogleGenerativeAI = None

from config import settings


def get_llm():
    """Get the appropriate LLM based on configuration"""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain-openai not installed. Please install: pip install langchain-openai")
            
        return ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    elif settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
            
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("langchain-google-genai not installed. Please install: pip install langchain-google-genai")
            
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=settings.temperature,
            google_api_key=settings.gemini_api_key
        )
    elif settings.llm_provider == "local":
        # For local models, you can use Ollama or other local providers
        # This is a placeholder - you'll need to configure based on your local setup
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model="llama2",
                temperature=settings.temperature
            )
        except ImportError:
            raise ValueError("Ollama not installed. Install with: pip install langchain-community")
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

