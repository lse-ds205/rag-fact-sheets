from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from group4py.src.database import Connection
from group4py.src.chunk_embed import Embedding
from group4py.src.helpers import Logger

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from constants.regex import (
    REGEX_WORD_PLACEHOLDER_1, REGEX_WORD_PLACEHOLDER_2, REGEX_WORD_PLACEHOLDER_3,
    REGEX_SENTENCE_PLACEHOLDER_1, REGEX_SENTENCE_PLACEHOLDER_2, REGEX_SENTENCE_PLACEHOLDER_3
)
from helpers import Logger, Test, TaskInfo

class VectorComparison:
    """
    Vector comparison class.
    """
    def __init__(self, connection: Connection = None):
        """
        Initialize the vector comparison engine.
        
        Args:
            connection: Database connection. If None, creates a new one.
        """
        self.connection = connection if connection is not None else Connection()

    def get_vector_similarity(self, chunk: str):
        """
        Get vector similarity for a given chunk.
        """
        # Placeholder for vector similarity calculation
        pass

    @Logger.debug_log()
    @staticmethod
    def filter_by_similarity_threshold(chunks: List[Dict[str, Any]], 
                                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter chunks by minimum similarity threshold.
        
        Args:
            chunks: List of chunk dictionaries with similarity scores
            threshold: Minimum similarity score to keep
            
        Returns:
            Filtered list of chunks
        """
        try:
            filtered_chunks = [
                chunk for chunk in chunks 
                if chunk.get('similarity_score', 0) >= threshold
            ]
            
            logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} above threshold {threshold}")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error filtering chunks by threshold: {e}")
            return chunks

    @Logger.debug_log()
    @staticmethod
    def filter_by_cosine_distance_threshold(chunks: List[Dict[str, Any]], 
                                          max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter chunks by maximum cosine distance threshold (lower distance = higher similarity).
        
        Args:
            chunks: List of chunk dictionaries with cosine distances
            max_distance: Maximum cosine distance to keep (0 = identical, 2 = opposite)
            
        Returns:
            Filtered list of chunks
        """
        try:
            filtered_chunks = [
                chunk for chunk in chunks 
                if chunk.get('cosine_distance', 2.0) <= max_distance
            ]
            
            logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} within distance {max_distance}")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error filtering chunks by distance threshold: {e}")
            return chunks

    @Logger.debug_log()
    @staticmethod
    def create_vector_indices():
        """
        Create pgvector indices for transformer and word2vec embeddings.
        This should be called once after embeddings are stored in the database.
        
        Returns:
            bool: True if indices were created successfully, False otherwise
        """
        try:
            import os
            from sqlalchemy import create_engine, text
            
            # Get database URL from environment
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                logger.error("No DATABASE_URL found in environment")
                return False
            
            logger.info("Creating pgvector indices for embeddings...")
            
            engine = create_engine(db_url)
            
            with engine.connect() as conn:
                # Create the pgvector extension if it doesn't exist
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("Ensured pgvector extension exists")
                
                # Convert transformer_embedding column to vector type
                conn.execute(text("""
                    ALTER TABLE doc_chunks 
                    ALTER COLUMN transformer_embedding TYPE vector(768)
                    USING transformer_embedding::vector(768);
                """))
                logger.info("Converted transformer_embedding column to vector(768) type")
                
                # Convert word2vec_embedding column to vector type
                conn.execute(text("""
                    ALTER TABLE doc_chunks 
                    ALTER COLUMN word2vec_embedding TYPE vector(300)
                    USING word2vec_embedding::vector(300);
                """))
                logger.info("Converted word2vec_embedding column to vector(300) type")
                
                # Create an index for transformer_embedding
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS transformer_embedding_idx 
                    ON doc_chunks 
                    USING ivfflat (transformer_embedding vector_cosine_ops)
                    WITH (lists = 100);
                """))
                logger.info("Created transformer_embedding index")
                
                # Create an index for word2vec_embedding
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS word2vec_embedding_idx 
                    ON doc_chunks 
                    USING ivfflat (word2vec_embedding vector_cosine_ops)
                    WITH (lists = 100);
                """))
                logger.info("Created word2vec_embedding index")
                
                # Commit changes
                conn.commit()
                
            logger.info("Successfully created all pgvector indices")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector indices: {e}")
            return False
     
class RegexComparison:
    """
    Regex comparison class.
    """
    def __init__(self):
        pass
    
    @Logger.debug_log()
    def evaluate_regex_1(self, chunk: str):
        regex_1 = REGEX_WORD_PLACEHOLDER_1
        pass

    @Logger.debug_log()
    def evaluate_regex_2(self, chunk: str):
        regex_2 = REGEX_WORD_PLACEHOLDER_2
        pass

    @Logger.debug_log()
    def evaluate_regex_3(self, chunk: str):
        regex_3 = REGEX_WORD_PLACEHOLDER_3
        pass

class SomeOtherComparison:
    """
    Some other comparison class.
    """
    def __init__(self):
        pass

class Evaluator:
    """
    Evaluator class. General methodology:
    """
    def __init__(self):
        pass

    @Logger.debug_log()
    def evaluate_total_score():
        """
        Some function(s) to evaluate the prompt.
        """
        score_1 = Evaluator._Evaluator.evaluate_function_1()
        score_2 = Evaluator._Evaluator.evaluate_function_2()
        total_score = score_1 + score_2
        pass

    @Logger.debug_log()
    def some_other_evaluation():
        pass

    class _Evaluator:

        @Logger.debug_log()
        @staticmethod
        def evaluate_function_1():
            """
            Some function(s) to evaluate the prompt.
            """
            pass

        @Logger.debug_log()
        @staticmethod
        def evaluate_function_2():
            """
            Some function(s) to evaluate the prompt.
            """
            pass