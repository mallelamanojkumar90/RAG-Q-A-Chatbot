"""
Question generator using RAG pipeline
"""
import json
import uuid
from typing import List, Dict, Any
try:
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
except ImportError:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document

from config import settings
from generator.llm_provider import get_llm
from pipeline.embeddings import get_embedding_function
from pipeline.vector_store import get_vector_store
from generator.history import QuestionHistory


class QuestionGenerator:
    """Generate questions using RAG"""
    
    def __init__(self):
        self.llm = get_llm()
        self.embedding_function = get_embedding_function()
        self.vector_store = get_vector_store(self.embedding_function)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        )
        self.history = QuestionHistory()
        
        # Question generation prompt
        self.prompt_template = PromptTemplate(
            input_variables=["context", "topic"],
            template="""You are an expert IIT JEE question generator. Based on the following context from IIT JEE question papers, generate a high-quality question.

Context:
{context}

Instructions:
1. Generate a single, well-structured IIT JEE level question
2. The question should be clear, precise, and test conceptual understanding
3. Include multiple choice options (A, B, C, D) if applicable
4. Make the question challenging but fair
5. Ensure the question is based on the provided context

Generate the question in the following JSON format:
{{
    "question": "The question text here",
    "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation of the answer",
    "subject": "Physics/Chemistry/Mathematics",
    "difficulty": "Easy/Medium/Hard",
    "topic": "Specific topic name"
}}

Question:"""
        )
    
    def _generate_single_question(self, context_chunks: List[Document]) -> Dict[str, Any]:
        """Generate a single question from context chunks"""
        # Combine context
        context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])
        
        # Create prompt
        prompt = self.prompt_template.format(
            context=context_text,
            topic="IIT JEE"
        )
        
        # Generate question
        response = self.llm.invoke(prompt)
        
        # Parse response
        try:
            # Extract JSON from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            question_data = json.loads(response_text)
            
            # Add metadata
            question_data["id"] = str(uuid.uuid4())
            question_data["source_documents"] = [
                {
                    "source": chunk.metadata.get("source", ""),
                    "filename": chunk.metadata.get("filename", ""),
                    "chunk_id": chunk.metadata.get("chunk_id", "")
                }
                for chunk in context_chunks
            ]
            
            return question_data
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response_text}")
            # Return a fallback question structure
            return {
                "id": str(uuid.uuid4()),
                "question": response_text[:500],  # Use first 500 chars as question
                "options": {},
                "correct_answer": "",
                "explanation": "",
                "subject": "Unknown",
                "difficulty": "Medium",
                "topic": "Unknown",
                "error": "Failed to parse response"
            }
    
    def generate_questions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate multiple unique questions"""
        if count > settings.max_questions_per_request:
            count = settings.max_questions_per_request
        
        questions = []
        attempts = 0
        max_attempts = count * 3  # Allow up to 3x attempts to find unique questions
        
        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            
            # Retrieve random context (by querying with a random topic or using similarity search)
            # For variety, we can use different query strategies
            query = self._get_random_query()
            context_chunks = self.retriever.get_relevant_documents(query)
            
            if not context_chunks:
                print("No relevant context found, skipping...")
                continue
            
            # Generate question
            question = self._generate_single_question(context_chunks)
            question_id = question.get("id")
            
            # Check if question already exists
            if question_id and not self.history.question_exists(question_id):
                questions.append(question)
                self.history.add_question(question_id, question)
            else:
                print(f"Duplicate question detected, skipping...")
        
        return questions
    
    def _get_random_query(self) -> str:
        """Generate a random query to retrieve diverse context"""
        import random
        queries = [
            "physics problem",
            "chemistry reaction",
            "mathematics calculation",
            "conceptual question",
            "numerical problem",
            "theoretical concept",
            "problem solving",
            "derivation",
            "formula application"
        ]
        return random.choice(queries)

