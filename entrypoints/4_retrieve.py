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
from group4py.src.evaluator import Evaluator
from group4py.src.evaluator import VectorComparison, RegexComparison, SomeOtherComparison
from group4py.src.chunk_embed import Embedding
from group4py.src.query import Booster

logger = logging.getLogger(__name__)


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
    

def retrieve_chunks(embedded_prompt, embedding_type='transformer', top_k=20, ensure_indices=True, 
                   country=None, n_per_doc=None, min_similarity=0.0):
    """
    Retrieve the top K most similar chunks from the database using pgvector similarity search.
    
    Args:
        embedded_prompt: The embedded query vector (list of floats)
        embedding_type: Type of embedding to use ('transformer' or 'word2vec')
        top_k: Number of top similar chunks to retrieve (ignored if n_per_doc is specified)
        ensure_indices: Whether to ensure vector indices exist before querying
        country: Optional country name to filter documents by
        n_per_doc: Optional number of chunks to return per document (overrides top_k)
        min_similarity: Minimum similarity threshold (0-1)
        
    Returns:
        List of dictionaries containing chunk data and similarity scores
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
        
        # Create database connection using the Connection class
        connection = Connection()
        
        # Get a session using the Connection class
        session = connection.get_session()
        
        try:
            # Determine which embedding column to use and convert prompt to vector string
            if embedding_type == 'transformer':
                embedding_column = 'transformer_embedding'
                vector_dim = 768
            elif embedding_type == 'word2vec':
                embedding_column = 'word2vec_embedding'
                vector_dim = 300
            else:
                logger.error(f"[4_RETRIEVE] Invalid embedding_type: {embedding_type}")
                return []
            
            # Validate embedded_prompt dimensions
            if not embedded_prompt or len(embedded_prompt) != vector_dim:
                logger.error(f"[4_RETRIEVE] Invalid embedding dimensions. Expected {vector_dim}, got {len(embedded_prompt) if embedded_prompt else 0}")
                return []
            
            # Convert embedding to pgvector format
            vector_str = '[' + ','.join(map(str, embedded_prompt)) + ']'
            
            # Build query based on parameters
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
                            c.chunk_metadata,
                            d.country,
                            1 - (c.{embedding_column} <=> :query_vector) AS similarity_score,
                            ROW_NUMBER() OVER (
                                PARTITION BY c.doc_id 
                                ORDER BY c.{embedding_column} <=> :query_vector
                            ) as rank
                        FROM doc_chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        WHERE c.{embedding_column} IS NOT NULL
                          AND 1 - (c.{embedding_column} <=> :query_vector) >= :min_similarity
                          {country_filter}
                    )
                    SELECT 
                        id,
                        doc_id,
                        content,
                        chunk_index,
                        paragraph,
                        language,
                        chunk_metadata,
                        country,
                        similarity_score
                    FROM similarity_results
                    WHERE rank <= :n_per_doc
                    ORDER BY similarity_score DESC
                """)
                
                query_params = {
                    'query_vector': vector_str,
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
                            c.chunk_metadata,
                            d.country,
                            1 - (c.{embedding_column} <=> :query_vector) AS similarity_score
                        FROM doc_chunks c
                        JOIN documents d ON c.doc_id = d.doc_id
                        WHERE c.{embedding_column} IS NOT NULL
                          AND LOWER(d.country) = LOWER(:country)
                          AND 1 - (c.{embedding_column} <=> :query_vector) >= :min_similarity
                        ORDER BY 1 - (c.{embedding_column} <=> :query_vector) DESC
                        LIMIT :top_k
                    """)
                    
                    query_params = {
                        'query_vector': vector_str,
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
                            c.chunk_metadata,
                            NULL as country,
                            1 - (c.{embedding_column} <=> :query_vector) AS similarity_score
                        FROM doc_chunks c
                        WHERE c.{embedding_column} IS NOT NULL
                          AND 1 - (c.{embedding_column} <=> :query_vector) >= :min_similarity
                        ORDER BY 1 - (c.{embedding_column} <=> :query_vector) DESC
                        LIMIT :top_k
                    """)
                    
                    query_params = {
                        'query_vector': vector_str,
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
                    'chunk_metadata': row[6],
                    'country': row[7],
                    'similarity_score': float(row[8])
                }
                chunks.append(chunk_data)
            
            if country:
                logger.info(f"[4_RETRIEVE] Successfully retrieved {len(chunks)} chunks for country '{country}'")
            else:
                logger.info(f"[4_RETRIEVE] Successfully retrieved {len(chunks)} chunks")
            return chunks
            
        finally:
            # Close the session
            session.close()
            
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in chunk retrieval: {e}\nTraceback: {traceback_string}")
        return []

@Test.dummy_chunk()
def evaluate_chunks(prompt, chunks):
    try:
        vector_comparison = VectorComparison()
        regex_comparison = RegexComparison()
        some_other_comparison = SomeOtherComparison()
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in chunk evaluation: {e}\nTraceback: {traceback_string}")
        raise e
    
    try:
        evaluated_chunks = Evaluator(), vector_comparison, regex_comparison, some_other_comparison
        return evaluated_chunks
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in chunk evaluation: {e}\nTraceback: {traceback_string}")
        raise e


@Logger.log(log_file=project_root / "logs/retrieve.log", log_level="DEBUG")
def run_script(prompt: str = None, country: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main function to run the retrieval script.
    Args:
        prompt (str): The query prompt to use for retrieval.
        country (Optional[str]): Optional country name to filter documents by.
    
    Returns:
        List[Dict[str, Any]]: List of retrieved chunks with metadata.
    
    """
    try:
        logger.warning(f"\n\n[4_RETRIEVE] Running script...")
        
        # Use example prompt if none provided
        if prompt is None:
            prompt = "What are the main policies for emissions reduction?"
            logger.info(f"[4_RETRIEVE] Using example prompt: {prompt}")
        else:
            logger.info(f"[4_RETRIEVE] Using provided prompt: {prompt}")
        
        # Define an example country for use, only if country is specified
        if country is None:
            country = "Afghanistan"
            logger.info(f"[4_RETRIEVE] Using example country: {country}")
        else:
            logger.info(f"[4_RETRIEVE] Using provided country: {country}")

        # Step 1: Embed the prompt
        embedded_prompt = embed_prompt(prompt)
        
        # Step 2: Retrieve chunks
        chunks = retrieve_chunks(
            embedded_prompt=embedded_prompt,
            embedding_type='transformer',  # Better semantic understanding
            top_k=20,                      # Retrieve top 20 chunks
            ensure_indices=True,           # Ensure indices are created
            n_per_doc=None,                # Get diversity across documents
            country=country,               # Filter by country
            min_similarity=0.5             # Filter out chunks with low similarity
        )
        
        logger.info(f"[4_RETRIEVE] Retrieved {len(chunks)} chunks")
        output_path = project_root / "data/retrieved_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.debug(f"[4_RETRIEVE] Dumped {len(chunks)} chunks to {output_path}")
        return chunks
    
        # Uncomment the following lines to evaluate chunks
        # Step 3: Evaluate chunks
        # if chunks:
        #     evaluated_chunks = evaluate_chunks(prompt, chunks)
        #     logger.warning(f"[4_RETRIEVE] Chunks evaluated. Narrowed down to {len(evaluated_chunks) if evaluated_chunks else 0} chunks.")
        #     return evaluated_chunks
        # else:
        #     logger.warning(f"[4_RETRIEVE] No chunks retrieved.")
        #     return []

    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 4_retrieve.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        raise e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the query script with a specified prompt.')
    parser.add_argument('--prompt', type=str, help='Prompt to execute the script for.')
    parser.add_argument('--country', type=str, help='Country to filter documents by (optional).')
    args = parser.parse_args()
    run_script(prompt=args.prompt, country=args.country)

    # When argparse is added, can no longer press the 'run button' in VSCode/Cursor.
    # Instead, need to python xx.py --argument
    # This is to, potentially, allow an alternative way of bridging, 
    #   of the end-to-end communication between interface.py/Github Actions and these sub-entrypoints
    #   because xx.py --argument could work better than calling the function directly
    # E.g. use in terminal: python 4_retrieve.py --prompt "I am a prompt"