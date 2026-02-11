"""
Question generator using RAG pipeline
"""
import json
import uuid
import re
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


def _fix_json_escape_sequences(json_str: str) -> str:
    r"""
    Fix invalid escape sequences in JSON strings, particularly LaTeX expressions.
    JSON only allows: \\, \", \/, \b, \f, \n, \r, \t, \uXXXX
    LaTeX expressions like \(, \), \[, \] need to be escaped as \\(, \\), etc.
    
    This function fixes backslashes that are not part of valid JSON escape sequences.
    """
    # Iterate through the string and fix invalid escape sequences
    result = []
    i = 0
    while i < len(json_str):
        if json_str[i] == '\\':
            # Check if this is a valid escape sequence
            if i + 1 < len(json_str):
                next_char = json_str[i + 1]
                # Valid escape sequences: ", \, /, b, f, n, r, t, or u (for \uXXXX)
                if next_char in '"\\/bfnrtu':
                    # Special handling for \uXXXX pattern
                    if next_char == 'u' and i + 5 < len(json_str):
                        # Check if followed by 4 hex digits
                        hex_part = json_str[i+2:i+6]
                        if all(c in '0123456789abcdefABCDEF' for c in hex_part):
                            # Valid \uXXXX escape
                            result.append(json_str[i:i+6])
                            i += 6
                            continue
                        else:
                            # Invalid \u - escape the backslash
                            result.append('\\\\')
                            i += 1
                            continue
                    else:
                        # Valid escape sequence (", \, /, b, f, n, r, t)
                        result.append(json_str[i:i+2])
                        i += 2
                        continue
                else:
                    # Invalid escape sequence - escape the backslash
                    # e.g., \( becomes \\, then we'll append ( on next iteration
                    result.append('\\\\')
                    i += 1
                    continue
            else:
                # Backslash at end of string - escape it
                result.append('\\\\')
                i += 1
                continue
        else:
            # Regular character
            result.append(json_str[i])
            i += 1
    
    return ''.join(result)


def _clean_latex_delimiters(text: str) -> str:
    """
    Remove LaTeX inline math delimiters and convert LaTeX math expressions to readable text.
    Converts \(...\) to (...) and \[...\] to [...]
    Also converts common LaTeX math commands to readable format.
    """
    if not isinstance(text, str):
        return text
    
    # Fix broken LaTeX commands (missing backslashes)
    # Fix common typos like "rac" instead of "\frac", "left" instead of "\left", etc.
    text = re.sub(r'\brac\{', r'\\frac{', text)  # Fix "rac{" → "\frac{"
    text = re.sub(r'\bleft\(', r'\\left(', text)  # Fix "left(" → "\left("
    text = re.sub(r'(\S)\s+ight\)', r'\1\\right)', text)  # Fix "x ight)" → "x\right)"
    text = re.sub(r'^ight\)', r'\\right)', text)  # Fix "ight)" at start → "\right)"
    text = re.sub(r'\bright\)', r'\\right)', text)  # Fix "right)" → "\right)"
    text = re.sub(r'\bcdot\b', r'\\cdot', text)  # Fix "cdot" → "\cdot"
    
    # Remove inline math delimiters \( and \)
    text = text.replace(r'\(', '(').replace(r'\)', ')')
    
    # Remove display math delimiters \[ and \]
    text = text.replace(r'\[', '[').replace(r'\]', ']')
    
    # Convert \left( and \right) to regular parentheses
    text = text.replace(r'\left(', '(').replace(r'\right)', ')')
    text = text.replace(r'\left[', '[').replace(r'\right]', ']')
    text = text.replace(r'\left{', '{').replace(r'\right}', '}')
    
    # Convert \cdot to · (middle dot) or * for multiplication
    text = text.replace(r'\cdot', '·')
    
    # Convert common LaTeX symbols to readable text FIRST
    # This ensures symbols inside fractions are converted before fraction processing
    text = text.replace(r'\pi', 'π')
    text = text.replace(r'\alpha', 'α')
    text = text.replace(r'\beta', 'β')
    text = text.replace(r'\gamma', 'γ')
    text = text.replace(r'\delta', 'δ')
    text = text.replace(r'\theta', 'θ')
    text = text.replace(r'\lambda', 'λ')
    text = text.replace(r'\mu', 'μ')
    text = text.replace(r'\sigma', 'σ')
    text = text.replace(r'\phi', 'φ')
    text = text.replace(r'\omega', 'ω')
    text = text.replace(r'\Delta', 'Δ')
    text = text.replace(r'\Omega', 'Ω')
    
    # Convert \frac{a}{b} to (a/b)
    # Handle nested braces by processing from innermost to outermost
    def find_and_replace_frac(text_str):
        """Find and replace \frac commands, handling nested braces"""
        result = []
        i = 0
        while i < len(text_str):
            if text_str[i:i+5] == r'\frac':
                i += 5
                # Skip whitespace
                while i < len(text_str) and text_str[i] in ' \t':
                    i += 1
                # Find opening brace for numerator
                if i < len(text_str) and text_str[i] == '{':
                    # Find matching closing brace for numerator
                    brace_count = 1
                    num_start = i + 1
                    i += 1
                    while i < len(text_str) and brace_count > 0:
                        if text_str[i] == '{':
                            brace_count += 1
                        elif text_str[i] == '}':
                            brace_count -= 1
                        i += 1
                    numerator = text_str[num_start:i-1]
                    
                    # Skip whitespace
                    while i < len(text_str) and text_str[i] in ' \t':
                        i += 1
                    # Find opening brace for denominator
                    if i < len(text_str) and text_str[i] == '{':
                        brace_count = 1
                        denom_start = i + 1
                        i += 1
                        while i < len(text_str) and brace_count > 0:
                            if text_str[i] == '{':
                                brace_count += 1
                            elif text_str[i] == '}':
                                brace_count -= 1
                            i += 1
                        denominator = text_str[denom_start:i-1]
                        # Convert to readable format
                        result.append(f'({numerator}/{denominator})')
                        continue
            result.append(text_str[i])
            i += 1
        return ''.join(result)
    
    # Process fractions (may need multiple passes for nested fractions)
    max_passes = 5
    for _ in range(max_passes):
        if r'\frac' not in text:
            break
        new_text = find_and_replace_frac(text)
        if new_text == text:
            break
        text = new_text
    
    # Convert \sqrt{x} to √(x)
    def replace_sqrt(match):
        content = match.group(1)
        return f'√({content})'
    text = re.sub(r'\\sqrt\{([^}]+)\}', replace_sqrt, text)
    
    # Convert \sqrt[n]{x} to nth root
    def replace_nth_root(match):
        n = match.group(1)
        content = match.group(2)
        if n == '2':
            return f'√({content})'
        return f'({n}th root of {content})'
    text = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', replace_nth_root, text)
    
    # Convert superscripts x^{n} to x^n (simple case)
    def replace_superscript(match):
        base = match.group(1)
        exp = match.group(2)
        # Remove braces if present
        exp = exp.replace('{', '').replace('}', '')
        return f'{base}^{exp}'
    text = re.sub(r'([a-zA-Z0-9\)\]\)]+)\^\{([^}]+)\}', replace_superscript, text)
    text = re.sub(r'([a-zA-Z0-9\)\]\)]+)\^([0-9]+)', r'\1^\2', text)
    
    # Convert subscripts x_{n} to x_n
    def replace_subscript(match):
        base = match.group(1)
        sub = match.group(2)
        # Remove braces if present
        sub = sub.replace('{', '').replace('}', '')
        return f'{base}_{sub}'
    text = re.sub(r'([a-zA-Z0-9\)\]\)]+)_\{([^}]+)\}', replace_subscript, text)
    text = re.sub(r'([a-zA-Z0-9\)\]\)]+)_([0-9]+)', r'\1_\2', text)
    
    # Remove other LaTeX spacing commands
    text = text.replace(r'\,', ' ')  # thin space
    text = text.replace(r'\;', ' ')  # medium space
    text = text.replace(r'\:', ' ')  # medium space
    text = text.replace(r'\!', '')   # negative thin space
    
    # Remove remaining LaTeX commands that might be single characters
    # This handles cases like \sin, \cos, \log, etc.
    text = re.sub(r'\\(sin|cos|tan|sec|csc|cot|log|ln|exp|min|max|lim|sum|prod|int)\b', r'\1', text)
    
    # Clean up any remaining single backslash commands (but preserve escaped ones)
    # This is a fallback for any LaTeX commands we might have missed
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    
    # Fix superscripts in units: m/s(^2) → m/s², m/s(^3) → m/s³, etc.
    def fix_unit_superscripts(text_str):
        """Convert unit superscripts like m/s(^2) to m/s²"""
        # Pattern: unit/(^n) or unit(^n) where unit is like m, kg, etc.
        # Convert (^2) to ², (^3) to ³, etc.
        superscript_map = {
            '^2': '²', '^3': '³', '^4': '⁴', '^5': '⁵',
            '^6': '⁶', '^7': '⁷', '^8': '⁸', '^9': '⁹',
            '^1': '¹', '^0': '⁰'
        }
        
        # Match patterns like m/s(^2), kg(^3), etc.
        def replace_superscript(match):
            unit = match.group(1)
            exp = match.group(2)
            superscript = superscript_map.get(exp, f'^{exp[1:]}')  # Remove the ^ if not in map
            return f'{unit}{superscript}'
        
        # Pattern: word characters, optionally /word, followed by (^n)
        text_str = re.sub(r'([a-zA-Z/]+)\((\^[0-9]+)\)', replace_superscript, text_str)
        return text_str
    
    text = fix_unit_superscripts(text)
    
    # Remove unnecessary parentheses around single variables or simple expressions
    # Pattern: space + ( + variable/expression + ) + space or punctuation
    # Examples: " ( x ) " → " x ", " ( f(x) ) " → " f(x) ", " (t = 2) " → " t = 2 "
    def remove_unnecessary_parens(text_str):
        """Remove parentheses around single variables or simple expressions"""
        # Pattern to match: space + ( + content + ) + space/punctuation/end
        # Content should be a single variable, function call, or simple expression
        patterns = [
            # Single variable: ( t ), ( x ), etc.
            (r' \( ([a-zA-Z]) \) ', r' \1 '),
            (r' \( ([a-zA-Z]) \)([.,;:!?])', r' \1\2'),
            (r'^\( ([a-zA-Z]) \) ', r'\1 '),
            (r' \( ([a-zA-Z]) \)$', r' \1'),
            
            # Variable assignments: ( t = 2 ), ( x = a ), etc.
            (r' \( ([a-zA-Z]\s*=\s*[^)]+) \) ', r' \1 '),
            (r' \( ([a-zA-Z]\s*=\s*[^)]+) \)([.,;:!?])', r' \1\2'),
            (r'^\( ([a-zA-Z]\s*=\s*[^)]+) \) ', r'\1 '),
            (r' \( ([a-zA-Z]\s*=\s*[^)]+) \)$', r' \1'),
            
            # Function calls: ( f(x) ), ( g(x) ), ( x(t) ), etc.
            (r' \( ([a-zA-Z]\([^)]+\)) \) ', r' \1 '),
            (r' \( ([a-zA-Z]\([^)]+\)) \)([.,;:!?])', r' \1\2'),
            (r'^\( ([a-zA-Z]\([^)]+\)) \) ', r'\1 '),
            (r' \( ([a-zA-Z]\([^)]+\)) \)$', r' \1'),
            
            # Function definitions: ( x(t) = ... ), ( f(x) = ... ), etc.
            # Match until we find a closing paren followed by space or punctuation
            (r' \( ([a-zA-Z]\([^)]+\)\s*=\s*[^)]+?) \) ', r' \1 '),
            (r' \( ([a-zA-Z]\([^)]+\)\s*=\s*[^)]+?) \)([.,;:!?])', r' \1\2'),
            (r'^\( ([a-zA-Z]\([^)]+\)\s*=\s*[^)]+?) \) ', r'\1 '),
            (r' \( ([a-zA-Z]\([^)]+\)\s*=\s*[^)]+?) \)$', r' \1'),
            
            # Simple expressions in parentheses at start/end of sentences
            (r'^\( ([^()]+) \) ', r'\1 '),
            (r' \( ([^()]+) \)$', r' \1'),
        ]
        
        for pattern, replacement in patterns:
            text_str = re.sub(pattern, replacement, text_str)
        
        return text_str
    
    # Apply parentheses removal (multiple passes for nested cases)
    for _ in range(3):
        new_text = remove_unnecessary_parens(text)
        if new_text == text:
            break
        text = new_text
    
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text


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
            
            # Fix invalid escape sequences (e.g., LaTeX expressions with unescaped backslashes)
            response_text = _fix_json_escape_sequences(response_text)
            
            question_data = json.loads(response_text)
            
            # Clean LaTeX delimiters from text fields
            if "question" in question_data:
                question_data["question"] = _clean_latex_delimiters(question_data["question"])
            if "explanation" in question_data:
                question_data["explanation"] = _clean_latex_delimiters(question_data["explanation"])
            if "options" in question_data and isinstance(question_data["options"], dict):
                # Clean each option value
                for key, value in question_data["options"].items():
                    if isinstance(value, str):
                        question_data["options"][key] = _clean_latex_delimiters(value)
            
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

