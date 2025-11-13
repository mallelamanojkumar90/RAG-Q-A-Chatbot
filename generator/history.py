"""
Question history tracker to avoid duplicates
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Set
from datetime import datetime

from config import settings


class QuestionHistory:
    """Track generated questions to avoid duplicates"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = settings.history_db_path
        
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for question history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS question_history (
                id TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                question_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON question_history(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def question_exists(self, question_id: str) -> bool:
        """Check if a question ID already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM question_history WHERE id = ?", (question_id,))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
    
    def add_question(self, question_id: str, question_data: Dict[str, Any]):
        """Add a question to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        question_text = question_data.get("question", "")
        question_json = json.dumps(question_data)
        
        cursor.execute("""
            INSERT OR REPLACE INTO question_history 
            (id, question_text, question_data) 
            VALUES (?, ?, ?)
        """, (question_id, question_text, question_json))
        
        conn.commit()
        conn.close()
    
    def get_all_question_ids(self) -> Set[str]:
        """Get all question IDs from history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM question_history")
        ids = {row[0] for row in cursor.fetchall()}
        
        conn.close()
        return ids
    
    def get_history_count(self) -> int:
        """Get total number of questions in history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM question_history")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def clear_history(self):
        """Clear all question history (use with caution)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM question_history")
        
        conn.commit()
        conn.close()

