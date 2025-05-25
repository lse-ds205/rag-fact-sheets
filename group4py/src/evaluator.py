from pathlib import Path
import sys
import os
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from sqlalchemy import create_engine, text
from group4py.src.database import Connection
from group4py.src.chunk_embed import Embedding
from group4py.src.helpers import Logger
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from constants.regex import (
    ndc_keyword_mapper, NDCKeywordMapper, QueryType, KeywordGroup,
    REGEX_NDC_TARGETS, REGEX_EMISSIONS, REGEX_RENEWABLE_ENERGY, 
    REGEX_CLIMATE_FINANCE, REGEX_NUMERICAL_TARGETS, REGEX_POLICY_MEASURES
)
from helpers import Logger, Test, TaskInfo

class VectorComparison:
    """
    Vector comparison class for performing similarity searches using pgvector embeddings.
    """
    def __init__(self, connection: Connection = None):
        """
        Initialize the vector comparison engine.
        
        Args:
            connection: Database connection. If None, creates a new one.
        """
        self.connection = connection if connection is not None else Connection()
        self.embedding_engine = Embedding()
        self.logger = logging.getLogger(__name__)

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

class RegexComparison:
    """
    Regex comparison class for evaluating chunks based on keyword matches.
    """
    def __init__(self, keyword_mapper: NDCKeywordMapper = None):
        """
        Initialize RegexComparison with keyword mapper
        
        Args:
            keyword_mapper: Optional NDCKeywordMapper instance. Uses global instance if None.
        """
        self.keyword_mapper = keyword_mapper or ndc_keyword_mapper
        self.logger = logging.getLogger(__name__)
    
    @Logger.debug_log()
    def evaluate_chunk_score(self, chunk_content: str, query: str) -> Dict[str, Any]:
        """
        Evaluate how well a chunk matches the query based on keyword patterns.
        
        Args:
            chunk_content: The text content of the chunk
            query: The user's query
            
        Returns:
            Dict containing:
                - regex_score: Overall regex matching score (0.0 to 1.0)
                - keyword_matches: Number of total keyword matches
                - match_details: Detailed breakdown by query type
                - query_types: Detected query types for the query
        """
        try:
            # Get comprehensive evaluation from the keyword mapper
            evaluation = self.keyword_mapper.evaluate_chunk_keywords(chunk_content, query)
            
            # Normalize the score to 0-1 range (assuming max reasonable score is 20)
            max_reasonable_score = 20.0
            normalized_score = min(evaluation['total_score'] / max_reasonable_score, 1.0)
            
            result = {
                'regex_score': normalized_score,
                'keyword_matches': evaluation['total_matches'],
                'match_details': evaluation['match_breakdown'],
                'query_types': evaluation['detected_types'],
                'raw_score': evaluation['total_score']
            }
            
            self.logger.debug(f"[REGEX_EVAL] Query: '{query[:50]}...' | Score: {normalized_score:.3f} | Matches: {evaluation['total_matches']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[REGEX_EVAL] Error evaluating chunk: {e}")
            return {
                'regex_score': 0.0,
                'keyword_matches': 0,
                'match_details': {},
                'query_types': [],
                'raw_score': 0.0,
                'error': str(e)
            }
    
    @Logger.debug_log()
    def get_keyword_highlights(self, chunk_content: str, query: str) -> Dict[str, Any]:
        """
        Extract highlighted keywords from chunk based on query patterns.
        
        Args:
            chunk_content: The text content of the chunk
            query: The user's query
            
        Returns:
            Dictionary with highlighted keywords and their positions
        """
        try:
            # Detect query types
            detected_types = self.keyword_mapper.detect_query_type(query)
            highlights = {}
            
            # For each detected query type, find pattern matches
            for query_type, confidence in detected_types:
                if query_type in self.keyword_mapper.compiled_patterns:
                    patterns = self.keyword_mapper.compiled_patterns[query_type]
                    
                    # Find matches for each category
                    primary_matches = self._find_matches(patterns['primary'], chunk_content)
                    secondary_matches = self._find_matches(patterns['secondary'], chunk_content)
                    contextual_matches = self._find_matches(patterns['contextual'], chunk_content)
                    
                    # Add to highlights if we found matches
                    if primary_matches or secondary_matches or contextual_matches:
                        highlights[query_type.value] = {
                            'primary': primary_matches,
                            'secondary': secondary_matches,
                            'contextual': contextual_matches
                        }
            
            return highlights
        
        except Exception as e:
            self.logger.error(f"[KEYWORD_HIGHLIGHTS] Error getting highlights: {e}")
            return {}
    
    def _find_matches(self, pattern, text):
        """Helper method to find all matches with their positions"""
        matches = []
        for match in pattern.finditer(text):
            matches.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return matches


class FuzzyRegexComparison:
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
        self.fuzzy_threshold = fuzzy_threshold
        self.ngram_size = ngram_size
        self.logger = logging.getLogger(__name__)
        
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
        
        similarity_result = self.contextual_similarity_score(chunk_content, query)
        
        # Apply boost factors based on detected patterns
        boosted_score = similarity_result['fuzzy_score']
        applied_boosts = []
        
        for pattern in similarity_result['semantic_patterns']:
            if pattern in boost_factors:
                boost = boost_factors[pattern]
                boosted_score *= boost
                applied_boosts.append(f"{pattern}: {boost}x")
        
        # Cap the final score at 1.0
        final_score = min(boosted_score, 1.0)
        
        return {
            'final_score': final_score,
            'base_fuzzy_score': similarity_result['fuzzy_score'],
            'query_coverage': similarity_result['query_coverage'],
            'contextual_matches': similarity_result['contextual_matches'],
            'semantic_patterns': similarity_result['semantic_patterns'],
            'total_matches': similarity_result['total_matches'],
            'applied_boosts': applied_boosts,
            'boost_multiplier': boosted_score / similarity_result['fuzzy_score'] if similarity_result['fuzzy_score'] > 0 else 1.0
        }

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