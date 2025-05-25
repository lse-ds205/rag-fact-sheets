import sys
import os
from pathlib import Path
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from sqlalchemy import text, func
import json

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.database import Connection
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.evaluator import Evaluator, VectorComparison, RegexComparison, FuzzyRegexComparison
from group4py.src.chunk_embed import Embedding
from group4py.src.query import Booster
from group4py.src.constants.prompts import (
    QUESTION_PROMPT_1, QUESTION_PROMPT_2, QUESTION_PROMPT_3, QUESTION_PROMPT_4,
    QUESTION_PROMPT_5, QUESTION_PROMPT_6, QUESTION_PROMPT_7, QUESTION_PROMPT_8
)

logger = logging.getLogger(__name__)

# Dictionary mapping question numbers to their respective prompts
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
    logger.info(f"[4_RETRIEVE] Prompt given: {prompt}. Embedding prompt...")
    try:
        
        # # First boost the prompt to enhance search performance
        # booster = Booster()
        # boosted_prompt = booster.boost_function(prompt)
        # logger.debug(f"[4_RETRIEVE] Prompt boosted: {boosted_prompt[:50]}...")
        
        # Initialize the embedding model, load models
        embedding_model = Embedding()
        embedding_model.load_models()

        # Embed the prompt using transformer model
        embedded_prompt = embedding_model.embed_transformer(prompt)
        
        logger.info(f"[4_RETRIEVE] Prompt successfully embedded to vector of dimension {len(embedded_prompt)}")
        return embedded_prompt
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in prompt embedding: {e}\nTraceback: {traceback_string}")
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
        logger.info("[4_RETRIEVE] No chunks to evaluate")
        return []
    
    logger.info(f"[4_RETRIEVE] Evaluating {len(chunks)} chunks using multiple methods for query: '{prompt}'")
    
    try:
        # Initialize comparison engines
        vector_comparison = VectorComparison()
        regex_comparison = RegexComparison()
        fuzzy_regex_comparison = FuzzyRegexComparison()
        
        evaluated_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_content = chunk.get('content', '')
            if not chunk_content:
                logger.warning(f"[4_RETRIEVE] Chunk {i} has no content, skipping evaluation")
                evaluated_chunks.append(chunk)
                continue
            
            # Start with the chunk as-is
            evaluated_chunk = chunk.copy()
            
            # 1. Apply regex evaluation (keyword-based)
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
                logger.error(f"[4_RETRIEVE] Error in regex evaluation for chunk {i}: {e}")
                evaluated_chunk['regex_score'] = 0.0
                evaluated_chunk['keyword_matches'] = 0
                evaluated_chunk['regex_evaluation'] = {'error': str(e)}
            
            # 2. Apply fuzzy regex evaluation (context-based)
            try:
                fuzzy_eval = fuzzy_regex_comparison.evaluate_chunk_relevance(chunk_content, prompt)
                evaluated_chunk['fuzzy_evaluation'] = fuzzy_eval
                evaluated_chunk['fuzzy_score'] = fuzzy_eval['final_score']
                evaluated_chunk['semantic_patterns'] = fuzzy_eval.get('semantic_patterns', [])
                
            except Exception as e:
                logger.error(f"[4_RETRIEVE] Error in fuzzy evaluation for chunk {i}: {e}")
                evaluated_chunk['fuzzy_score'] = 0.0
                evaluated_chunk['fuzzy_evaluation'] = {'error': str(e)}
            
            # 3. Add additional vector similarity if embedded_prompt is provided
            additional_vector_score = None
            if embedded_prompt and chunk.get('id'):
                try:
                    # Get additional vector similarity for specific chunk
                    chunk_id = chunk['id']
                    chunk_ids = [chunk_id]
                    similarities = vector_comparison.batch_similarity_calculation(
                        chunk_ids=chunk_ids,
                        query_embedding=embedded_prompt,
                        embedding_type='transformer'
                    )
                    
                    if similarities and chunk_id in similarities:
                        additional_vector_score = similarities[chunk_id]
                        evaluated_chunk['additional_vector_score'] = additional_vector_score
                except Exception as e:
                    logger.error(f"[4_RETRIEVE] Error in additional vector calculation for chunk {i}: {e}")
            
            # 4. Calculate combined score using all available methods
            original_similarity = evaluated_chunk.get('similarity_score', 0.0)
            regex_score = evaluated_chunk.get('regex_score', 0.0)
            fuzzy_score = evaluated_chunk.get('fuzzy_score', 0.0)
            
            # Configurable weights for different scoring methods
            vector_weight = 0.55  # Vector similarity has highest weight
            regex_weight = 0.25   # Direct keyword matches
            fuzzy_weight = 0.20   # Fuzzy contextual matching
            
            # Combine scores
            combined_score = (
                vector_weight * original_similarity +
                regex_weight * regex_score +
                fuzzy_weight * fuzzy_score
            )
            
            # If we have additional vector score (from batch similarity), blend it in
            if additional_vector_score is not None:
                # Blend the two vector scores (original and additional)
                blended_vector_score = (original_similarity + additional_vector_score) / 2
                combined_score = (
                    vector_weight * blended_vector_score +
                    regex_weight * regex_score +
                    fuzzy_weight * fuzzy_score
                )
                evaluated_chunk['blended_vector_score'] = blended_vector_score
            
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
        
        # Log evaluation statistics
        regex_scores = [c.get('regex_score', 0) for c in evaluated_chunks]
        fuzzy_scores = [c.get('fuzzy_score', 0) for c in evaluated_chunks]
        keyword_counts = [c.get('keyword_matches', 0) for c in evaluated_chunks]
        
        if regex_scores:
            avg_regex_score = sum(regex_scores) / len(regex_scores)
            avg_fuzzy_score = sum(fuzzy_scores) / len(fuzzy_scores)
            avg_keyword_matches = sum(keyword_counts) / len(keyword_counts)
            max_regex_score = max(regex_scores)
            max_fuzzy_score = max(fuzzy_scores)
            
            logger.info(f"[4_RETRIEVE] Evaluation complete:")
            logger.info(f"  - Average regex score: {avg_regex_score:.3f}")
            logger.info(f"  - Average fuzzy score: {avg_fuzzy_score:.3f}")
            logger.info(f"  - Average keyword matches: {avg_keyword_matches:.1f}")
            logger.info(f"  - Max regex score: {max_regex_score:.3f} | Max fuzzy score: {max_fuzzy_score:.3f}")
            
            # Show top 5 evaluated chunks
            logger.debug("[4_RETRIEVE] Top evaluated chunks:")
            for i, chunk in enumerate(evaluated_chunks[:5], 1):
                combined = chunk.get('combined_score', 0)
                similarity = chunk.get('similarity_score', 0)
                regex = chunk.get('regex_score', 0)
                fuzzy = chunk.get('fuzzy_score', 0)
                matches = chunk.get('keyword_matches', 0)
                semantic = ", ".join(chunk.get('semantic_patterns', []))
                logger.debug(f"  #{i}: Combined={combined:.3f} (Vector={similarity:.3f}, Regex={regex:.3f}, Fuzzy={fuzzy:.3f})")
                logger.debug(f"     Keywords: {matches} matches | Semantic: {semantic}")
        
        return evaluated_chunks
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in chunk evaluation: {e}\nTraceback: {traceback_string}")
        # Return original chunks if evaluation fails
        return chunks


def retrieve_chunks(embedded_prompt, prompt, embedding_type='transformer', top_k=20, 
                   ensure_indices=True, country=None, n_per_doc=None, min_similarity=0.0):
    """
    Retrieve and evaluate the most similar chunks from the database.
    
    Args:
        embedded_prompt: The embedded query vector (list of floats)
        prompt: Original text prompt for evaluation
        embedding_type: Type of embedding to use ('transformer' or 'word2vec')
        top_k: Number of top similar chunks to retrieve (ignored if n_per_doc is specified)
        ensure_indices: Whether to ensure vector indices exist before querying
        country: Optional country name to filter documents by
        n_per_doc: Optional number of chunks to return per document (overrides top_k)
        min_similarity: Minimum similarity threshold (0-1)
        
    Returns:
        List of dictionaries containing evaluated chunks with scores
    """
    try:
        if country:
            logger.info(f"[4_RETRIEVE] Retrieving chunks for country '{country}' using {embedding_type} embeddings")
        else:
            logger.info(f"[4_RETRIEVE] Retrieving top {top_k} chunks using {embedding_type} embeddings")
        
        # Ensure vector indices exist if requested
        if ensure_indices:
            logger.info("[4_RETRIEVE] Ensuring vector indices exist...")
            if not VectorComparison.create_vector_indices():
                logger.warning("[4_RETRIEVE] Failed to create vector indices, continuing anyway...")
        
        # Use VectorComparison class to retrieve similar chunks
        vector_comp = VectorComparison()
        chunks = vector_comp.get_vector_similarity(
            embedded_prompt=embedded_prompt,
            embedding_type=embedding_type,
            top_k=top_k,
            min_similarity=min_similarity,
            country=country,
            n_per_doc=n_per_doc
        )
        
        if country:
            logger.info(f"[4_RETRIEVE] Successfully retrieved {len(chunks)} chunks for country '{country}'")
        else:
            logger.info(f"[4_RETRIEVE] Successfully retrieved {len(chunks)} chunks")
        
        # Evaluate chunks using regex, fuzzy regex, and additional vector comparisons
        if chunks:
            evaluated_chunks = evaluate_chunks(
                prompt=prompt, 
                chunks=chunks,
                embedded_prompt=embedded_prompt  # Pass in the embedded prompt for additional similarity checks
            )
            logger.info(f"[4_RETRIEVE] Chunks evaluated. Using {len(evaluated_chunks)} evaluated chunks.")
            return evaluated_chunks
        else:
            logger.warning(f"[4_RETRIEVE] No chunks retrieved to evaluate.")
            return []
            
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in chunk retrieval: {e}\nTraceback: {traceback_string}")
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
        logger.warning(f"\n\n[4_RETRIEVE] Running script...")
        
        # Select the question prompt based on the question number
        if question_number is None or question_number not in QUESTION_PROMPTS:
            question_number = 1  # Default to question 1 if invalid or not provided
            logger.info(f"[4_RETRIEVE] Using default question number: {question_number}")
        else:
            logger.info(f"[4_RETRIEVE] Using question number: {question_number}")
        
        # Get the prompt text from the corresponding QUESTION_PROMPT
        prompt = QUESTION_PROMPTS[question_number]
        logger.info(f"[4_RETRIEVE] Selected prompt: {prompt[:100]}...")
        
        # Define an example country for use, only if country is specified
        if country is None:
            country = "Afghanistan"
            logger.info(f"[4_RETRIEVE] Using example country: {country}")
        else:
            logger.info(f"[4_RETRIEVE] Using provided country: {country}")

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
            country=country,               # Filter by country
            min_similarity=0.5             # Filter out chunks with low similarity
        )
        logger.info(f"[4_RETRIEVE] Retrieved and evaluated {len(evaluated_chunks)} chunks")
        
        if evaluated_chunks:
            # Create a metadata object to include with the evaluated chunks
            metadata = {
                "question_number": question_number,
                "query_text": prompt,
                "country": country,
                "timestamp": Logger.get_timestamp(),
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
            logger.debug(f"[4_RETRIEVE] Saved {len(evaluated_chunks)} evaluated chunks with query metadata to {evaluated_output_path}")
            
            return evaluated_chunks
        else:
            logger.warning(f"[4_RETRIEVE] No chunks retrieved.")
            return []

    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 4_retrieve.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the retrieval script with a specified question number.')
    parser.add_argument('--question', type=int, choices=range(1, 9), 
                        help='Question number (1-8) to select a predefined prompt.')
    parser.add_argument('--country', type=str, help='Country to filter documents by (optional).')
    args = parser.parse_args()
    run_script(question_number=args.question, country=args.country)

    # Usage examples:
    # python 4_retrieve.py --question 1 --country "Japan"
    # python 4_retrieve.py --question 4  # Uses default country (Afghanistan)