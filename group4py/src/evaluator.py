from pathlib import Path
import sys
import os
import re
import logging
import traceback
import numpy as np
import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import text, create_engine
from difflib import SequenceMatcher

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from database import Connection
from helpers.internal import Logger
from database import Connection
from schema import DatabaseConfig
from embed.hoprag import HopRAGGraphProcessor

logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'hex'):
            return obj.hex
        return json.JSONEncoder.default(self, obj)

class Evaluator:
    """Base class defining the interface for chunk evaluation strategies."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        logger.debug(f"Initialized {self.name} evaluator")
    
    def evaluate(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Evaluates chunks against a query. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalizes a score to be within the specified range."""
        return max(min_val, min(max_val, score))


class VectorComparison(Evaluator):
    """
    Vector-based similarity comparison using cosine similarity.
    """
    
    def __init__(self, connection=None):
        """
        Initialize VectorComparison with optional database connection.
        
        Args:
            connection: Optional database Connection instance
        """
        super().__init__()
        if connection is not None:
            self.connection = connection
        else:
            # Create a connection with default config
            config = DatabaseConfig.from_env()
            self.connection = Connection(config)
            self.connection.connect()
        
        # Test the vector functionality
        self._test_vector_functionality()

    def _test_vector_functionality(self):
        """Test the pgvector functionality to verify it's working properly"""
        try:
            engine = self.connection.get_engine()
            with engine.connect() as conn:
                # First check if pgvector extension exists
                result = conn.execute(text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"))
                extension_exists = result.scalar()
                if not extension_exists:
                    logger.error("pgvector extension not installed in database")
                    return False
                
                # Test a simple vector operation
                test_query = text("""
                    SELECT 1 - ('[1,2,3]'::vector <=> '[1,2,4]'::vector) as similarity
                """)
                result = conn.execute(test_query)
                similarity = result.scalar()
                logger.info(f"pgvector test result: similarity = {similarity}")
                
                # Check for NULL embeddings without using array_length
                count_query = text("""
                    SELECT COUNT(*) FROM doc_chunks 
                    WHERE transformer_embedding IS NULL
                """)
                null_count = conn.execute(count_query).scalar() or 0
                if null_count > 0:
                    logger.warning(f"Found {null_count} chunks with NULL embeddings")
                
                # Check dimensions using pgvector's dimension functions or column definitions
                try:
                    # Use the column definition to get dimensions
                    meta_query = text("""
                        SELECT 
                            a.attname AS column_name,
                            format_type(a.atttypid, a.atttypmod) AS data_type
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE c.relname = 'doc_chunks'
                        AND n.nspname = 'public'
                        AND a.attname IN ('transformer_embedding', 'word2vec_embedding', 'hoprag_embedding')
                    """)
                    
                    meta_result = conn.execute(meta_query)
                    column_info = {row[0]: row[1] for row in meta_result}
                    
                    dimensions = {}
                    for col, type_info in column_info.items():
                        # Extract dimension from type like "vector(768)"
                        if 'vector' in type_info:
                            dim_match = re.search(r'vector\((\d+)\)', type_info)
                            if dim_match:
                                dimensions[col] = int(dim_match.group(1))
                    
                    if dimensions:
                        logger.info(f"Vector dimensions from schema: {dimensions}")
                    else:
                        # Fallback: Try to get a sample vector and count elements
                        for col in ['transformer_embedding', 'word2vec_embedding', 'hoprag_embedding']:
                            try:
                                # Get vector as text and count commas to determine dimension
                                sample_query = text(f"""
                                    SELECT '{{'::text || {col}::text || '}}'::text
                                    FROM doc_chunks
                                    WHERE {col} IS NOT NULL
                                    LIMIT 1
                                """)
                                
                                with conn.begin():  # Ensure fresh transaction
                                    sample_result = conn.execute(sample_query)
                                    sample_row = sample_result.fetchone()
                                    
                                    if sample_row and sample_row[0]:
                                        vector_text = sample_row[0]
                                        # Count commas and add 1 to get dimension
                                        elements = vector_text.strip('{}').split(',')
                                        dimensions[col] = len(elements)
                            except Exception as ex:
                                logger.debug(f"Could not determine dimension for {col}: {ex}")
                        
                        if dimensions:
                            logger.info(f"Vector dimensions from samples: {dimensions}")
                        else:
                            logger.warning("Could not determine vector dimensions")
                
                except Exception as e:
                    logger.warning(f"Could not check vector dimensions: {e}")
                
                return True
        except Exception as e:
            logger.error(f"Vector functionality test failed: {e}")
            logger.error(traceback.format_exc())
            return False

    @Logger.debug_log()
    def get_vector_similarity(self, embedded_prompt: List[float], embedding_type: str = 'transformer', 
                            top_k: int = 10, min_similarity: float = 0.1, country: str = None, 
                            n_per_doc: int = None) -> List[Dict[str, Any]]:
        """
        Get vector similarity for a given embedded prompt.
        
        Args:
            embedded_prompt: The embedded query vector
            embedding_type: Type of embedding to use ('transformer' or 'word2vec')
            top_k: Number of top similar chunks to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            country: Optional country filter
            n_per_doc: Optional limit of chunks per document
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        try:
            # Create database connection using the Connection class
            session = self.connection.get_session()
            
            try:
                # Determine which embedding column to use and convert prompt to vector string
                if embedding_type == 'transformer':
                    embedding_column = 'transformer_embedding'
                    vector_dim = 768
                elif embedding_type == 'word2vec':
                    embedding_column = 'word2vec_embedding'
                    vector_dim = 300
                else:
                    logger.error(f"[VECTOR_SIMILARITY] Invalid embedding_type: {embedding_type}")
                    return []
                
                # Validate embedded_prompt dimensions
                if not embedded_prompt or len(embedded_prompt) != vector_dim:
                    logger.error(f"[VECTOR_SIMILARITY] Invalid embedding dimensions. Expected {vector_dim}, got {len(embedded_prompt) if embedded_prompt else 0}")
                    return []
                
                # Convert embedding to PostgreSQL vector literal format
                vector_literal = '[' + ','.join(map(str, embedded_prompt)) + ']'
                
                # IMPROVED: Log sample of the vector to check format
                logger.debug(f"Vector literal sample (first 20 elements): {vector_literal[:50]}...")
                
                # Use a simpler initial query to test vector functionality
                test_query = text(f"""
                    SELECT 
                        id, 
                        1 - ({embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score
                    FROM doc_chunks
                    WHERE {embedding_column} IS NOT NULL
                    LIMIT 5
                """)
                
                test_result = session.execute(test_query)
                test_rows = test_result.fetchall()
                
                # Log the test results
                if test_rows:
                    logger.info(f"Vector test query returned {len(test_rows)} rows")
                    for i, row in enumerate(test_rows):
                        logger.info(f"  Row {i+1}: id={row[0]}, similarity={row[1]}")
                else:
                    logger.warning("Vector test query returned no results")
                
                # Now proceed with the full query as before, with better error handling
                # ... [rest of the existing query construction code] ...
                
                # Add diagnostic query to check embedding format
                diagnostic_query = text(f"""
                    SELECT 
                        pg_typeof({embedding_column}) as embedding_type,
                        array_length({embedding_column}, 1) as embedding_dim,
                        {embedding_column}[1:3] as embedding_sample
                    FROM doc_chunks
                    WHERE {embedding_column} IS NOT NULL
                    LIMIT 1
                """)
                
                diagnostic_result = session.execute(diagnostic_query)
                diagnostic_row = diagnostic_result.fetchone()
                
                if diagnostic_row:
                    logger.info(f"Embedding diagnostic: type={diagnostic_row[0]}, dim={diagnostic_row[1]}, sample={diagnostic_row[2]}")
                else:
                    logger.warning("No embeddings found in the database")
                
                # ... [continue with the original function code] ...
                
                # Rest of the function remains the same
                
                # If no results were returned, try a basic query to see if data exists
                if not chunks:
                    basic_query = text("""
                        SELECT COUNT(*) FROM doc_chunks
                        WHERE transformer_embedding IS NOT NULL
                    """)
                    count = session.execute(basic_query).scalar() or 0
                    logger.info(f"Database has {count} chunks with embeddings")
                
                return chunks
                
            finally:
                # Close the session
                session.close()
                
        except Exception as e:
            traceback_string = traceback.format_exc()
            logger.error(f"[VECTOR_SIMILARITY] Error in chunk retrieval: {e}\nTraceback: {traceback_string}")
            return []

    @Logger.debug_log()
    def get_similarity_by_chunk_content(self, chunk_content: str, query_embedding: List[float], 
                                      embedding_type: str = 'transformer') -> float:
        """
        Calculate similarity between a chunk's content and a query embedding by looking up
        the chunk's stored embedding in the database.
        
        Args:
            chunk_content: The text content of the chunk
            query_embedding: The embedded query vector
            embedding_type: Type of embedding to use ('transformer' or 'word2vec')
            
        Returns:
            Similarity score (0.0 to 1.0), or 0.0 if chunk not found or error
        """
        try:
            # Get a session using the Connection class
            session = self.connection.get_session()
            
            try:
                # Determine which embedding column to use
                if embedding_type == 'transformer':
                    embedding_column = 'transformer_embedding'
                    vector_dim = 768
                elif embedding_type == 'word2vec':
                    embedding_column = 'word2vec_embedding'
                    vector_dim = 300
                else:
                    logger.error(f"[CHUNK_SIMILARITY] Invalid embedding_type: {embedding_type}")
                    return 0.0
                
                # Validate query_embedding dimensions
                if not query_embedding or len(query_embedding) != vector_dim:
                    logger.error(f"[CHUNK_SIMILARITY] Invalid query embedding dimensions. Expected {vector_dim}, got {len(query_embedding) if query_embedding else 0}")
                    return 0.0                # Convert query embedding to PostgreSQL vector literal format
                vector_literal = '[' + ','.join(map(str, query_embedding)) + ']'
                  # Query to find the chunk by content and calculate similarity
                query = text(f"""
                    SELECT 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score
                    FROM doc_chunks c
                    WHERE c.content = :chunk_content
                      AND c.{embedding_column} IS NOT NULL
                    LIMIT 1
                """)
                
                result = session.execute(query, {
                    'chunk_content': chunk_content
                })
                
                row = result.fetchone()
                if row:
                    similarity_score = float(row[0])
                    logger.debug(f"[CHUNK_SIMILARITY] Found similarity score: {similarity_score:.4f}")
                    return similarity_score
                else:
                    logger.warning(f"[CHUNK_SIMILARITY] Chunk content not found in database")
                    return 0.0
                    
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"[CHUNK_SIMILARITY] Error calculating similarity: {e}")
            return 0.0

    @Logger.debug_log()
    def batch_similarity_calculation(self, chunk_ids: List[str], query_embedding: List[float], 
                                   embedding_type: str = 'transformer') -> Dict[str, float]:
        """
        Calculate similarity scores for multiple chunks efficiently in a single query.
        Now handles chunks without embeddings by assigning them a score of 0.0.
        """
        try:
            if not chunk_ids:
                return {}
                
            session = self.connection.get_session()
            
            try:
                # Determine which embedding column and dimension to use
                if embedding_type == 'transformer':
                    embedding_column = 'transformer_embedding'
                    vector_dim = 768
                elif embedding_type == 'word2vec':
                    embedding_column = 'word2vec_embedding'
                    vector_dim = 300
                elif embedding_type == 'hoprag':
                    embedding_column = 'hoprag_embedding'
                    # Get hoprag dimension from first row since it might vary
                    dim_query = text("""
                        SELECT format_type(a.atttypid, a.atttypmod) AS data_type
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE c.relname = 'doc_chunks'
                        AND n.nspname = 'public'
                        AND a.attname = 'hoprag_embedding'
                    """)
                    dim_result = session.execute(dim_query).fetchone()
                    if dim_result and dim_result[0]:
                        dim_match = re.search(r'vector\((\d+)\)', dim_result[0])
                        if dim_match:
                            vector_dim = int(dim_match.group(1))
                        else:
                            vector_dim = 1024  # Default fallback dimension
                    else:
                        vector_dim = 1024  # Default fallback dimension
                else:
                    logger.error(f"[BATCH_SIMILARITY] Invalid embedding_type: {embedding_type}")
                    return {}
                
                # Validate query_embedding dimensions
                if not query_embedding or len(query_embedding) != vector_dim:
                    logger.error(f"[BATCH_SIMILARITY] Invalid query embedding dimensions. Expected {vector_dim}, got {len(query_embedding) if query_embedding else 0}")
                    return {}
                
                # Convert query embedding to PostgreSQL vector literal format
                vector_literal = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Convert chunk_ids to a comma-separated string for SQL IN clause
                quoted_ids = ", ".join(f"'{id}'" for id in chunk_ids)
                
                # Modified query to handle NULL embeddings
                query = text(f"""
                    SELECT 
                        c.id::text,
                        CASE 
                            WHEN c.{embedding_column} IS NULL THEN 0.0
                            ELSE 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim}))
                        END AS similarity_score
                    FROM doc_chunks c
                    WHERE c.id::text IN ({quoted_ids})
                """)
                
                result = session.execute(query)
                
                # Build result dictionary
                similarities = {}
                for row in result:
                    chunk_id = row[0]
                    similarity_score = float(row[1])
                    similarities[chunk_id] = similarity_score
                
                # For any chunk_ids that weren't found in the database, assign 0.0
                for chunk_id in chunk_ids:
                    if chunk_id not in similarities:
                        similarities[chunk_id] = 0.0
                
                logger.info(f"[BATCH_SIMILARITY] Calculated similarity for {len(similarities)}/{len(chunk_ids)} chunks")
                return similarities
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"[BATCH_SIMILARITY] Error in batch similarity calculation: {e}")
            logger.error(traceback.format_exc())
            # Return 0.0 for all chunks if there's an error
            return {chunk_id: 0.0 for chunk_id in chunk_ids}

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

class RegexComparison(Evaluator):
    """
    Regex-based comparison for keyword and pattern matching.
    """
    
    def __init__(self):
        super().__init__()
        # Climate-specific keywords and patterns
        self.climate_keywords = {
            'emissions': ['emission', 'emissions', 'ghg', 'greenhouse gas', 'co2', 'carbon dioxide', 'methane', 'nitrous oxide'],
            'targets': ['target', 'goal', 'objective', 'commitment', 'reduction', 'increase', 'percentage', '%'],
            'energy': ['renewable', 'solar', 'wind', 'hydro', 'nuclear', 'fossil fuel', 'coal', 'oil', 'gas'],
            'adaptation': ['adaptation', 'resilience', 'climate change', 'vulnerability', 'impact'],
            'mitigation': ['mitigation', 'reduction', 'abatement', 'sequestration', 'offset'],
            'finance': ['finance', 'funding', 'investment', 'cost', 'budget', 'billion', 'million'],
            'policy': ['policy', 'regulation', 'law', 'framework', 'strategy', 'plan', 'program']
        }
        
        # Numerical patterns
        self.number_patterns = {
            'percentage': r'\d+(?:\.\d+)?%',
            'year': r'20\d{2}',
            'emission_amount': r'\d+(?:\.\d+)?\s*(?:Mt|Gt|kt|t)?\s*CO2(?:e|eq)?',
            'monetary': r'\$?\d+(?:\.\d+)?\s*(?:billion|million|trillion)'
        }

    def evaluate_chunk_score(self, chunk_content: str, query: str) -> Dict[str, Any]:
        """
        Evaluate a chunk using regex patterns and keyword matching.
        
        Args:
            chunk_content: Content of the chunk to evaluate
            query: Query string to match against
            
        Returns:
            Dictionary with evaluation results
        """
        chunk_lower = chunk_content.lower()
        query_lower = query.lower()
        
        # Count keyword matches by category
        category_matches = {}
        total_matches = 0
        
        for category, keywords in self.climate_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            category_matches[category] = matches
            total_matches += matches
        
        # Check for numerical patterns
        numerical_matches = {}
        for pattern_name, pattern in self.number_patterns.items():
            matches = len(re.findall(pattern, chunk_content, re.IGNORECASE))
            numerical_matches[pattern_name] = matches
            total_matches += matches
        
        # Query-specific keyword matching
        query_words = re.findall(r'\w+', query_lower)
        query_matches = sum(1 for word in query_words if word in chunk_lower and len(word) > 2)
        
        # Calculate regex score (0-1)
        base_score = min(total_matches / 10.0, 1.0)  # Normalize to max 1.0
        query_boost = min(query_matches / max(len(query_words), 1), 1.0)
        
        regex_score = 0.7 * base_score + 0.3 * query_boost
        
        return {
            'regex_score': regex_score,
            'keyword_matches': total_matches,
            'category_matches': category_matches,
            'numerical_matches': numerical_matches,
            'query_word_matches': query_matches,
            'query_types': list(category_matches.keys())
        }

    def get_keyword_highlights(self, chunk_content: str, query: str) -> List[str]:
        """
        Get highlighted keywords found in the chunk.
        
        Args:
            chunk_content: Content to search in
            query: Query to find keywords from
            
        Returns:
            List of highlighted keyword phrases
        """
        highlights = []
        chunk_lower = chunk_content.lower()
        
        # Find query words
        query_words = re.findall(r'\w+', query.lower())
        for word in query_words:
            if len(word) > 2 and word in chunk_lower:
                highlights.append(word)
        
        # Find climate keywords
        for category, keywords in self.climate_keywords.items():
            for keyword in keywords:
                if keyword in chunk_lower:
                    highlights.append(f"{keyword} ({category})")
        
        return highlights[:10]  # Limit to top 10 highlights

class FuzzyRegexComparison(Evaluator):
    """
    Fuzzy regex comparison class that implements advanced text matching techniques
    including fuzzy matching, n-gram analysis, and contextual pattern recognition.
    """
    
    def __init__(self, fuzzy_threshold: float = 0.6, ngram_size: int = 3):
        """
        Initialize FuzzyRegexComparison with configurable parameters.
        
        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0.0 to 1.0)
            ngram_size: Size of n-grams for text analysis
        """
        super().__init__()
        self.fuzzy_threshold = fuzzy_threshold
        self.ngram_size = ngram_size
        
        # Climate-specific patterns for contextual matching
        self.climate_patterns = {
            'mitigation': [
                r'\b(?:reduc\w+|lower\w+|decreas\w+|cut\w*)\s+(?:emission\w*|ghg|co2|carbon)',
                r'\b(?:renewable|clean|green)\s+energy',
                r'\bcarbon\s+(?:neutral|negative|capture)',
                r'\b(?:energy\s+efficiency|efficiency\s+improvements?)'
            ],
            'adaptation': [
                r'\b(?:adapt\w+|resilien\w+|vulnerab\w+)\s+(?:to|against)?\s*(?:climate|weather)',
                r'\b(?:disaster|risk)\s+(?:management|reduction|preparedness)',
                r'\b(?:infrastructure|coastal|agricultural)\s+adaptation',
                r'\b(?:early\s+warning|climate\s+monitoring)'
            ],
            'finance': [
                r'\$\d+(?:[\d,]*)?(?:\s*(?:million|billion|trillion))?',
                r'\b(?:fund\w+|invest\w+|financ\w+|budget\w+)\s+(?:for|towards|in)?\s*(?:climate|green)',
                r'\b(?:green\s+bonds?|climate\s+finance|carbon\s+tax)',
                r'\b(?:development\s+bank|multilateral\s+fund)'
            ],
            'targets': [
                r'\b(?:by|until|before)\s+(?:20\d{2}|2030|2050)',
                r'\b\d+%?\s+(?:reduction|increase|improvement)',
                r'\b(?:net\s+zero|carbon\s+neutral|emissions?\s+free)',
                r'\b(?:baseline|reference)\s+year'
            ]
        }

    @Logger.debug_log()
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching by normalizing whitespace,
        removing special characters, and converting to lowercase.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text string
        """
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove excessive punctuation but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\%\$]', ' ', text)
        
        # Normalize common abbreviations and terms
        replacements = {
            r'\bghg\b': 'greenhouse gas',
            r'\bco2\b': 'carbon dioxide',
            r'\bch4\b': 'methane',
            r'\bn2o\b': 'nitrous oxide',
            r'\bndc\b': 'nationally determined contribution',
            r'\bunfccc\b': 'united nations framework convention climate change'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @Logger.debug_log()
    def generate_ngrams(self, text: str, n: int = None) -> List[str]:
        """
        Generate n-grams from text for pattern matching.
        
        Args:
            text: Input text
            n: N-gram size (uses instance default if None)
            
        Returns:
            List of n-gram strings
        """
        n = n or self.ngram_size
        words = text.split()
        
        if len(words) < n:
            return [text]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    @Logger.debug_log()
    def fuzzy_pattern_match(self, text: str, patterns: List[str]) -> Dict[str, Any]:
        """
        Perform fuzzy matching against a list of regex patterns.
        
        Args:
            text: Text to search in
            patterns: List of regex patterns to match against
            
        Returns:
            Dictionary with match results and scores
        """
        matches = []
        total_score = 0.0
        
        preprocessed_text = self.preprocess_text(text)
        
        for pattern in patterns:
            try:
                # Direct regex match
                direct_matches = re.findall(pattern, preprocessed_text, re.IGNORECASE)
                if direct_matches:
                    matches.extend(direct_matches)
                    total_score += len(direct_matches) * 1.0
                    continue
                
                # Fuzzy matching for patterns that didn't match directly
                pattern_words = re.sub(r'[^\w\s]', ' ', pattern).split()
                text_ngrams = self.generate_ngrams(preprocessed_text)
                
                best_similarity = 0.0
                best_match = None
                
                for ngram in text_ngrams:
                    similarity = SequenceMatcher(None, ' '.join(pattern_words), ngram).ratio()
                    if similarity > best_similarity and similarity >= self.fuzzy_threshold:
                        best_similarity = similarity
                        best_match = ngram
                
                if best_match:
                    matches.append(best_match)
                    total_score += best_similarity
                
            except re.error as e:
                self.logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        return {
            'matches': matches,
            'match_count': len(matches),
            'total_score': total_score,
            'average_score': total_score / len(patterns) if patterns else 0.0
        }
    
    @Logger.debug_log()
    def contextual_similarity_score(self, chunk_content: str, query: str) -> Dict[str, Any]:
        """
        Calculate contextual similarity between chunk content and query using
        multiple regex techniques and fuzzy matching.
        
        Args:
            chunk_content: The text content of the chunk
            query: The user's query
            
        Returns:
            Dictionary containing detailed similarity analysis
        """
        try:
            results = {
                'fuzzy_score': 0.0,
                'contextual_matches': {},
                'query_coverage': 0.0,
                'semantic_patterns': [],
                'total_matches': 0
            }
            
            # Preprocess both texts
            processed_chunk = self.preprocess_text(chunk_content)
            processed_query = self.preprocess_text(query)
            
            # Extract key terms from query for targeted matching
            query_terms = [word for word in processed_query.split() if len(word) > 2]
            
            # Score based on direct term overlap
            term_matches = 0
            for term in query_terms:
                if re.search(rf'\b{re.escape(term)}\b', processed_chunk):
                    term_matches += 1
            
            query_coverage = term_matches / len(query_terms) if query_terms else 0.0
            results['query_coverage'] = query_coverage
            
            # Contextual pattern matching for different climate categories
            total_contextual_score = 0.0
            for category, patterns in results['contextual_matches'].items():
                match_result = self.fuzzy_pattern_match(processed_chunk, patterns)
                results['contextual_matches'][category] = match_result
                total_contextual_score += match_result['total_score']
                results['total_matches'] += match_result['match_count']
            
            # Calculate overall fuzzy score
            base_score = query_coverage * 0.4  # 40% weight for direct term matches
            contextual_score = min(total_contextual_score / 10.0, 0.6)  # 60% weight for contextual matches, capped
            
            results['fuzzy_score'] = base_score + contextual_score
            
            # Identify semantic patterns found
            for category, match_data in results['contextual_matches'].items():
                if match_data['match_count'] > 0:
                    results['semantic_patterns'].append(category)
            
            self.logger.debug(f"[FUZZY_REGEX] Query: '{query[:50]}...' | Score: {results['fuzzy_score']:.3f} | Coverage: {query_coverage:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"[FUZZY_REGEX] Error in contextual similarity calculation: {e}")
            return {
                'fuzzy_score': 0.0,
                'contextual_matches': {},
                'query_coverage': 0.0,
                'semantic_patterns': [],
                'total_matches': 0,
                'error': str(e)
            }
    
    @Logger.debug_log()
    def evaluate_chunk_relevance(self, chunk_content: str, query: str, 
                               boost_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of chunk relevance using multiple fuzzy regex techniques.
        
        Args:
            chunk_content: The text content of the chunk
            query: The user's query
            boost_factors: Optional dictionary of category boost factors
            
        Returns:
            Dictionary with comprehensive relevance analysis
        """
        boost_factors = boost_factors or {
            'mitigation': 1.0,
            'adaptation': 1.0,
            'finance': 1.2,  # Slightly boost finance-related content
            'targets': 1.1   # Slightly boost target-related content
        }
        
        try:
            # Simple fuzzy matching implementation
            chunk_lower = chunk_content.lower()
            query_lower = query.lower()
            
            # Query coverage calculation
            query_words = [word for word in query_lower.split() if len(word) > 2]
            word_matches = sum(1 for word in query_words if word in chunk_lower)
            query_coverage = word_matches / len(query_words) if query_words else 0.0
            
            # Pattern matching for climate categories
            pattern_scores = {}
            total_pattern_matches = 0
            semantic_patterns = []
            
            for category, patterns in self.climate_patterns.items():
                category_matches = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, chunk_content, re.IGNORECASE))
                    category_matches += matches
                
                if category_matches > 0:
                    semantic_patterns.append(category)
                    pattern_scores[category] = category_matches
            
            # Calculate base score
            base_score = 0.4 * query_coverage + 0.6 * min(total_pattern_matches / 5.0, 1.0)
            
            # Apply boost factors
            boosted_score = base_score
            applied_boosts = []
            
            for pattern in semantic_patterns:
                if pattern in boost_factors:
                    boost = boost_factors[pattern]
                    boosted_score *= boost
                    applied_boosts.append(f"{pattern}: {boost}x")
            
            final_score = min(boosted_score, 1.0)
            
            return {
                'final_score': final_score,
                'base_fuzzy_score': base_score,
                'query_coverage': query_coverage,
                'contextual_matches': pattern_scores,
                'semantic_patterns': semantic_patterns,
                'total_matches': total_pattern_matches,
                'applied_boosts': applied_boosts,
                'boost_multiplier': boosted_score / base_score if base_score > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Error in fuzzy evaluation: {e}")
            return {
                'final_score': 0.0,
                'base_fuzzy_score': 0.0,
                'query_coverage': 0.0,
                'contextual_matches': {},
                'semantic_patterns': [],
                'total_matches': 0,
                'applied_boosts': [],
                'boost_multiplier': 1.0,
                'error': str(e)
            }

class GraphHopRetriever(Evaluator):
    """
    Wrapper for the HopRAGGraphProcessor that provides a compatible interface
    for multi-hop reasoning through graph traversal and relationship-based navigation.
    
    This class leverages the existing HopRAGGraphProcessor implementation instead of
    duplicating functionality.
    """
    
    def __init__(self, connection=None):
        super().__init__()
        
        # Always create a config object, regardless of connection
        config = DatabaseConfig.from_env()
        
        if connection is not None:
            self.connection = connection
        else:
            # Create a connection with default config
            self.connection = Connection(config)
            self.connection.connect()
        
        # Create HopRAGGraphProcessor instance with the config
        self.processor = HopRAGGraphProcessor(config)
        
        # Initialize the processor
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if one doesn't exist
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.processor.initialize())
        logger.info(f"Initialized {self.name} using HopRAGGraphProcessor")
    
    def retrieve_with_hop_reasoning(self, query: str, top_k: int = 20, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Main retrieval method using multi-hop reasoning.
        
        Args:
            query: Query text
            top_k: Number of results to return
            country: Optional country filter
            
        Returns:
            List of retrieved chunks with scores and metadata
        """
        try:
            logger.info(f"Starting hop retrieval for query: '{query[:50]}...' (country: {country})")
            
            # Check if there are any relationships in the database
            engine = self.connection.get_engine()
            with engine.connect() as conn:
                # Count total relationships
                rel_count = conn.execute(text("SELECT COUNT(*) FROM logical_relationships")).scalar() or 0
                logger.info(f"Total relationships in database: {rel_count}")
                
                # If country filter is applied, check relationships for that country
                if country:
                    country_query = text("""
                        SELECT COUNT(*) 
                        FROM logical_relationships lr
                        JOIN doc_chunks src ON lr.source_chunk_id = src.id
                        JOIN doc_chunks tgt ON lr.target_chunk_id = tgt.id
                        WHERE 
                            src.chunk_data->>'country' = :country
                            AND tgt.chunk_data->>'country' = :country
                    """)
                    country_rel_count = conn.execute(country_query, {"country": country}).scalar() or 0
                    logger.info(f"Relationships for country '{country}': {country_rel_count}")
                    
                    if country_rel_count == 0:
                        logger.warning(f"No relationships found for country '{country}'. Graph traversal will yield no results.")
            
            # Use the processor to get results
            loop = asyncio.get_event_loop()
            logger.info(f"Executing processor.get_top_ranked_nodes with max_hops={2}")
            raw_results = loop.run_until_complete(self.processor.get_top_ranked_nodes(query, max_hops=2))
            
            logger.info(f"Raw processor results: {raw_results.get('total_nodes', 0)} total nodes found")
            
            # Format results to match the expected output format
            results = []
            for node in raw_results.get('top_nodes', [])[:top_k]:
                result = {
                    'id': node['chunk_id'],
                    'content': node['content'],
                    'doc_id': 'N/A',  # HopRAGGraphProcessor doesn't return doc_id in top_nodes
                    'hop_distance': node.get('hops', 0),
                    'centrality_score': node['centrality_scores'].get('pagerank', 0),
                    'semantic_score': node.get('bfs_confidence', 0),
                    'combined_score': node['combined_score'],
                    'method': 'graph_hop'
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} chunks using hop reasoning")
            return results
            
        except Exception as e:
            logger.error(f"Error in hop reasoning retrieval: {e}")
            traceback_string = traceback.format_exc()
            logger.error(traceback_string)
            return []
    
    def save_to_json(self, results: List[Dict[str, Any]], query: str, 
                    country: Optional[str] = None, question_number: Optional[int] = None) -> str:
        """
        Save retrieval results to JSON file in the data/retrieve_hop folder.
        
        Args:
            results: List of retrieval results
            query: Original query
            country: Optional country filter
            question_number: Optional question number
            
        Returns:
            Path to saved file
        """
        
        # Create output directory
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "retrieve_hop"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_str = f"_q{question_number}" if question_number is not None else ""
        country_str = f"_{country}" if country else ""
        filename = f"hop_results{country_str}{question_str}_{timestamp}.json"
        
        # Create output data structure
        output_data = {
            "metadata": {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "retrieval_method": "graph_hop",
                "country": country,
                "question_number": question_number,
                "hop_parameters": {
                    "max_hops": 2,
                    "min_confidence": 0.6
                }
            },
            "results": results
        }
        
        # Write to file
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, cls=UUIDEncoder)
        
        logger.info(f"Saved hop results to {output_path}")
        return str(output_path)