from pathlib import Path
import sys
import os
import re
import logging
import traceback
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text, func, create_engine
import uuid
from difflib import SequenceMatcher

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from database import Connection
from embedding import TransformerEmbedding, CombinedEmbedding
from helpers.internal import Logger
from database import Connection
from schema import DatabaseConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Base class for all evaluation methods.
    Provides common functionality for different evaluation strategies.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        logger.debug(f"Initialized {self.name} evaluator")
    
    def evaluate(self, chunks: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Base evaluation method to be overridden by subclasses.
        
        Args:
            chunks: List of chunks to evaluate
            query: Query string to evaluate against
            
        Returns:
            List of evaluated chunks with scores
        """
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize a score to be within the specified range.
        
        Args:
            score: Score to normalize
            min_val: Minimum value of normalized range
            max_val: Maximum value of normalized range
            
        Returns:
            Normalized score
        """
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
            connection = Connection()
            
            # Get a session using the Connection class
            session = connection.get_session()
            
            try:                # Determine which embedding column to use and convert prompt to vector string
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
                vector_literal = '[' + ','.join(map(str, embedded_prompt)) + ']'                # Build query based on parameters
                if n_per_doc is not None:
                    # Query to get top N chunks per document with optional country filter
                    country_filter = "AND LOWER(d.country) = LOWER(:country)" if country else ""
                    
                    query = text(f"""
                        WITH similarity_results AS (
                            SELECT 
                                c.id,
                                c.doc_id,
                                c.content,
                                c.chunk_index,
                                c.paragraph,
                                c.language,
                                c.chunk_data,
                                d.country,
                                1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score,
                                ROW_NUMBER() OVER (
                                    PARTITION BY c.doc_id 
                                    ORDER BY c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})
                                ) as rank
                            FROM doc_chunks c
                            JOIN documents d ON c.doc_id = d.doc_id
                            WHERE c.{embedding_column} IS NOT NULL
                              AND 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) >= :min_similarity
                              {country_filter}
                        )
                        SELECT 
                            id,
                            doc_id,
                            content,
                            chunk_index,
                            paragraph,
                            language,
                            chunk_data,
                            country,
                            similarity_score
                        FROM similarity_results                        WHERE rank <= :n_per_doc
                        ORDER BY similarity_score DESC
                    """)
                    
                    query_params = {
                        'n_per_doc': n_per_doc,
                        'min_similarity': min_similarity
                    }
                    if country:
                        query_params['country'] = country
                        
                else:
                    # Standard query for top K chunks with optional country filter
                    if country:
                        query = text(f"""
                            SELECT 
                                c.id,
                                c.doc_id,
                                c.content,
                                c.chunk_index,
                                c.paragraph,
                                c.language,
                                c.chunk_data,
                                d.country,
                                1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score
                            FROM doc_chunks c
                            JOIN documents d ON c.doc_id = d.doc_id
                            WHERE c.{embedding_column} IS NOT NULL
                              AND LOWER(d.country) = LOWER(:country)
                              AND 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) >= :min_similarity
                            ORDER BY 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) DESC
                            LIMIT :top_k
                        """)
                        
                        query_params = {
                            'country': country,
                            'top_k': top_k,
                            'min_similarity': min_similarity
                        }
                    else:
                        query = text(f"""
                            SELECT 
                                c.id,
                                c.doc_id,
                                c.content,
                                c.chunk_index,
                                c.paragraph,
                                c.language,
                                c.chunk_data,
                                NULL as country,
                                1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score
                            FROM doc_chunks c
                            WHERE c.{embedding_column} IS NOT NULL
                              AND 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) >= :min_similarity
                            ORDER BY 1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) DESC
                            LIMIT :top_k
                        """)
                        
                        query_params = {
                            'top_k': top_k,
                            'min_similarity': min_similarity
                        }
                
                # Execute query using the session from Connection class
                result = session.execute(query, query_params)
                
                # Convert results to list of dictionaries
                chunks = []
                for row in result:
                    chunk_data = {
                        'id': row[0],
                        'doc_id': row[1],
                        'content': row[2],
                        'chunk_index': row[3],
                        'paragraph': row[4],
                        'language': row[5],
                        'chunk_data': row[6],
                        'country': row[7],
                        'similarity_score': float(row[8])
                    }
                    chunks.append(chunk_data)
                
                if country:
                    logger.info(f"[VECTOR_SIMILARITY] Successfully retrieved {len(chunks)} chunks for country '{country}'")
                else:
                    logger.info(f"[VECTOR_SIMILARITY] Successfully retrieved {len(chunks)} chunks")
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
    def batch_similarity_calculation(self, chunk_ids: List[int], query_embedding: List[float], 
                                   embedding_type: str = 'transformer') -> Dict[int, float]:
        """
        Calculate similarity scores for multiple chunks efficiently in a single query.
        
        Args:
            chunk_ids: List of chunk IDs to calculate similarity for
            query_embedding: The embedded query vector
            embedding_type: Type of embedding to use ('transformer' or 'word2vec')
            
        Returns:
            Dictionary mapping chunk_id to similarity_score
        """
        try:
            if not chunk_ids:
                return {}
                
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
                    logger.error(f"[BATCH_SIMILARITY] Invalid embedding_type: {embedding_type}")
                    return {}
                
                # Validate query_embedding dimensions
                if not query_embedding or len(query_embedding) != vector_dim:
                    logger.error(f"[BATCH_SIMILARITY] Invalid query embedding dimensions. Expected {vector_dim}, got {len(query_embedding) if query_embedding else 0}")
                    return {}                # Convert query embedding to PostgreSQL vector literal format
                vector_literal = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Convert chunk_ids to tuple for SQL IN clause
                chunk_ids_tuple = tuple(chunk_ids)
                  # Query to calculate similarity for all specified chunks
                query = text(f"""
                    SELECT 
                        c.id,
                        1 - (c.{embedding_column}::vector <=> '{vector_literal}'::vector({vector_dim})) AS similarity_score
                    FROM doc_chunks c
                    WHERE c.id = ANY(:chunk_ids)
                      AND c.{embedding_column} IS NOT NULL
                """)                
                result = session.execute(query, {
                    'chunk_ids': chunk_ids
                })
                
                # Build result dictionary
                similarities = {}
                for row in result:
                    chunk_id = row[0]
                    similarity_score = float(row[1])
                    similarities[chunk_id] = similarity_score
                
                logger.info(f"[BATCH_SIMILARITY] Calculated similarity for {len(similarities)} chunks")
                return similarities
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"[BATCH_SIMILARITY] Error in batch similarity calculation: {e}")
            return {}

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
                    total_pattern_matches += matches
                
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