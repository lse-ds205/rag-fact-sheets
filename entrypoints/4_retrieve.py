import sys
from pathlib import Path
import logging
import traceback
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import json
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from database import Connection
from helpers.internal import Logger
from evaluator import VectorComparison, RegexComparison, FuzzyRegexComparison
from embedding import TransformerEmbedding
from constants.prompts import (
    QUESTION_PROMPT_1, QUESTION_PROMPT_2, QUESTION_PROMPT_3, QUESTION_PROMPT_4,
    QUESTION_PROMPT_5, QUESTION_PROMPT_6, QUESTION_PROMPT_7, QUESTION_PROMPT_8
)
from database import Connection
from schema import DatabaseConfig

logger = logging.getLogger(__name__)
load_dotenv()

QUESTION_PROMPTS = {
    1: QUESTION_PROMPT_1,
    2: QUESTION_PROMPT_2,
    3: QUESTION_PROMPT_3,
    4: QUESTION_PROMPT_4,
    5: QUESTION_PROMPT_5,
    6: QUESTION_PROMPT_6,
    7: QUESTION_PROMPT_7,
    8: QUESTION_PROMPT_8
}


def embed_prompt(prompt):
    """
    Embed a prompt using the embedding model.
    
    Args:
        prompt: The text string to embed
        
    Returns:
        list: Vector embedding of the prompt (list of floats)
    """
    try:  
        embedding_model = TransformerEmbedding()
        embedding_model.load_models()
        embedded_prompt = embedding_model.embed_transformer(prompt)
        return embedded_prompt
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"Error embedding prompt: {traceback_string}")
        raise e

def evaluate_chunks(prompt: str, chunks: List[Dict[str, Any]], embedded_prompt: List[float] = None) -> List[Dict[str, Any]]:
    """
    Evaluate chunks using multiple comparison methods: vector similarity, regex patterns, and fuzzy matching.
    
    Args:
        prompt: The original user query
        chunks: List of chunk dictionaries to evaluate
        embedded_prompt: Optional embedded prompt vector for additional similarity calculations
        
    Returns:
        List of chunks with evaluation scores and metadata added
    """
    if not chunks:
        return []

    try:
        # Step 1: Calculate vector similarities for all chunks
        
        # Create database config and connection for VectorComparison
        config = DatabaseConfig.from_env()
        db_connection = Connection(config)
        if not db_connection.connect():
            # Fall back to chunks without similarity scores
            for chunk in chunks:
                chunk['similarity_score'] = 0.0
            chunks_with_similarity = chunks
        else:
            # Initialize VectorComparison with proper database connection
            vector_comp = VectorComparison(connection=db_connection)
            
            # Get chunk IDs for batch similarity calculation
            chunk_ids = [chunk['id'] for chunk in chunks]
            
            # Calculate similarities in batch
            similarities = vector_comp.batch_similarity_calculation(
                chunk_ids=chunk_ids,
                query_embedding=embedded_prompt,
                embedding_type='transformer'
            )
            
            # Add similarity scores to chunks
            chunks_with_similarity = []
            for chunk in chunks:
                chunk_id = chunk['id']
                similarity_score = similarities.get(chunk_id, 0.0)
                
                # Add similarity score to chunk
                chunk['similarity_score'] = similarity_score
                chunks_with_similarity.append(chunk)
        
        # Initialize comparison engines
        regex_comparison = RegexComparison()
        fuzzy_regex_comparison = FuzzyRegexComparison()
        
        evaluated_chunks = []
        
        for i, chunk in enumerate(chunks_with_similarity):
            chunk_content = chunk.get('content', '')
            if not chunk_content:
                evaluated_chunks.append(chunk)
                continue
            
            # Start with the chunk as-is
            evaluated_chunk = chunk.copy()
            
            # 2. Apply regex evaluation (keyword-based)
            try:
                regex_eval = regex_comparison.evaluate_chunk_score(chunk_content, prompt)
                evaluated_chunk['regex_evaluation'] = regex_eval
                evaluated_chunk['regex_score'] = regex_eval['regex_score']
                evaluated_chunk['keyword_matches'] = regex_eval['keyword_matches']
                evaluated_chunk['query_types'] = regex_eval.get('query_types', [])
                
                # Get keyword highlights for debugging/explanation
                highlights = regex_comparison.get_keyword_highlights(chunk_content, prompt)
                if highlights:
                    evaluated_chunk['keyword_highlights'] = highlights
                
            except Exception as e:
                evaluated_chunk['regex_score'] = 0.0
                evaluated_chunk['keyword_matches'] = 0
                evaluated_chunk['regex_evaluation'] = {'error': str(e)}
            
            # 3. Apply fuzzy regex evaluation (context-based)
            try:
                fuzzy_eval = fuzzy_regex_comparison.evaluate_chunk_relevance(chunk_content, prompt)
                evaluated_chunk['fuzzy_evaluation'] = fuzzy_eval
                evaluated_chunk['fuzzy_score'] = fuzzy_eval['final_score']
                evaluated_chunk['semantic_patterns'] = fuzzy_eval.get('semantic_patterns', [])
                
            except Exception as e:
                evaluated_chunk['fuzzy_score'] = 0.0
                evaluated_chunk['fuzzy_evaluation'] = {'error': str(e)}
            
            # 4. Calculate combined score using all available methods
            similarity_score = evaluated_chunk.get('similarity_score', 0.0)
            regex_score = evaluated_chunk.get('regex_score', 0.0)
            fuzzy_score = evaluated_chunk.get('fuzzy_score', 0.0)
            
            # Configurable weights for different scoring methods
            vector_weight = 0.55  # Vector similarity has highest weight
            regex_weight = 0.25   # Direct keyword matches
            fuzzy_weight = 0.20   # Fuzzy contextual matching
            
            # Combine scores
            combined_score = (
                vector_weight * similarity_score +
                regex_weight * regex_score +
                fuzzy_weight * fuzzy_score
            )
            
            evaluated_chunk['combined_score'] = combined_score
            
            # Add information about how the score was calculated
            evaluated_chunk['score_weights'] = {
                'vector_weight': vector_weight,
                'regex_weight': regex_weight,
                'fuzzy_weight': fuzzy_weight
            }
            
            # Add to our collection of evaluated chunks
            evaluated_chunks.append(evaluated_chunk)
        
        # Sort by combined score (highest first)
        evaluated_chunks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return evaluated_chunks
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        # Return original chunks if evaluation fails
        return chunks

def retrieve_chunks(embedded_prompt, prompt, embedding_type='transformer', top_k=20, 
                   ensure_indices=True, country=None, n_per_doc=None, min_similarity=0.0):
    """
    Retrieve and evaluate the most similar chunks from the database using comprehensive evaluation.
    First evaluates ALL chunks, then returns top ones based on weighted average score.
    
    Args:
        embedded_prompt: The embedded query vector (list of floats)
        prompt: Original text prompt for evaluation
        embedding_type: Type of embedding to use ('transformer' or 'word2vec')
        top_k: Number of top similar chunks to retrieve
        ensure_indices: Whether to ensure vector indices exist before querying
        country: Optional country name to filter documents by
        n_per_doc: Optional number of chunks to return per document (overrides top_k)
        min_similarity: Minimum similarity threshold (0-1)
        
    Returns:
        List of dictionaries containing evaluated chunks with comprehensive scores
    """
    try:
        # Ensure vector indices exist if requested
        if ensure_indices:
            if not VectorComparison.create_vector_indices():
                pass  # Continue anyway
        
        # Step 1: Get ALL chunks from database for comprehensive evaluation
        config = DatabaseConfig.from_env()
        db_connection = Connection(config)
        
        if not db_connection.connect():
            return []
        
        # Get all chunks for the specified country or all chunks
        all_chunks = db_connection.get_all_chunks_for_evaluation(country=country)
        
        if not all_chunks:
            return []
        
        # Step 2: Run comprehensive evaluation on ALL chunks (includes vector similarity calculation)
        evaluated_chunks = evaluate_chunks(
            prompt=prompt, 
            chunks=all_chunks,
            embedded_prompt=embedded_prompt
        )
        
        if not evaluated_chunks:
            return []
        
        # Step 4: Apply minimum similarity filter after evaluation
        if min_similarity > 0.0:
            pre_filter_count = len(evaluated_chunks)
            evaluated_chunks = [
                chunk for chunk in evaluated_chunks 
                if chunk.get('combined_score', 0) >= min_similarity
            ]
        
        # Step 5: Handle n_per_doc constraint if specified
        if n_per_doc is not None:
            doc_chunk_counts = {}
            final_chunks = []
            
            for chunk in evaluated_chunks:
                doc_id = chunk.get('doc_id', 'unknown')
                if doc_chunk_counts.get(doc_id, 0) < n_per_doc:
                    final_chunks.append(chunk)
                    doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
                
                # Stop if we have enough chunks
                if len(final_chunks) >= top_k:
                    break
            
            evaluated_chunks = final_chunks
        
        # Step 6: Limit to requested top_k
        if len(evaluated_chunks) > top_k:
            evaluated_chunks = evaluated_chunks[:top_k]
        
        return evaluated_chunks
            
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"Error retrieving chunks: {traceback_string}")
        return []


@Logger.log(log_file=project_root / "logs/retrieve.log", log_level="DEBUG")
def run_script(question_number: int = None, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main function to run the retrieval script.
    Args:
        question_number (int): The question number (1-8) to use from predefined prompts.
        country (Optional[str]): Optional country name to filter documents by.
    
    Returns:
        List[Dict[str, Any]]: List of retrieved chunks with metadata.
    
    """
    try:
        # Select the question prompt based on the question number
        if question_number is None or question_number not in QUESTION_PROMPTS:
            question_number = 1  # Default to question 1 if invalid or not provided
        
        # Get the prompt text from the corresponding QUESTION_PROMPT
        prompt = QUESTION_PROMPTS[question_number]
        
        # Define an example country for use, only if country is specified
        if country is None:
            # Don't default to a specific country - retrieve from all countries
            country = None

        # Step 1: Embed the prompt
        embedded_prompt = embed_prompt(prompt)
        
        # Step 2: Retrieve and evaluate chunks in a single function call
        evaluated_chunks = retrieve_chunks(
            embedded_prompt=embedded_prompt,
            prompt=prompt,
            embedding_type='transformer',  # Better semantic understanding
            top_k=20,                      # Retrieve top 20 chunks
            ensure_indices=True,           # Ensure indices are created
            n_per_doc=None,                # Get diversity across documents
            country=country,               # Filter by country (None = all countries)
            min_similarity=0.3             # Lower threshold to see more results
        )
        
        if evaluated_chunks:
            # Create a metadata object to include with the evaluated chunks
            metadata = {
                "question_number": question_number,
                "query_text": prompt,
                "country": country,
                "timestamp": datetime.now().isoformat(),
                "chunk_count": len(evaluated_chunks)
            }
            
            # Create final output with query information and evaluated chunks
            final_output = {
                "metadata": metadata,
                "evaluated_chunks": evaluated_chunks
            }
            
            # Save only the evaluated chunks with score breakdowns
            evaluated_output_path = project_root / "data/evaluated_chunks.json"
            with open(evaluated_output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            
            return evaluated_chunks
        else:
            return []

    except Exception as e:
        traceback_string = traceback.format_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the retrieval script with a specified question number.')
    parser.add_argument('--question', type=int, choices=range(1, 9), 
                        help='Question number (1-8) to select a predefined prompt.')
    parser.add_argument('--country', type=str, help='Country to filter documents by (optional).')
    args = parser.parse_args()
    run_script(question_number=args.question, country=args.country)

    # Usage examples:
    # python 4_retrieve.py --question 1 --country "Japan"
    # python 4_retrieve.py --question 4  # Uses default country (Afghanistan)