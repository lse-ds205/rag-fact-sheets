import sys
from pathlib import Path
import logging
import traceback
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from sqlalchemy import text
import json
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from database import Connection
from helpers.internal import Logger
from evaluator import VectorComparison, RegexComparison, FuzzyRegexComparison, GraphHopRetriever
from embedding import TransformerEmbedding
from constants.prompts import (
    QUESTION_PROMPT_1, QUESTION_PROMPT_2, QUESTION_PROMPT_3, QUESTION_PROMPT_4,
    QUESTION_PROMPT_5, QUESTION_PROMPT_6, QUESTION_PROMPT_7, QUESTION_PROMPT_8,
    HOP_KEYWORDS, GENERAL_NDC_KEYWORDS
)
from database import Connection
from schema import DatabaseConfig, UUIDEncoder

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
        logger.error(f"[4_RETRIEVE] Error embedding prompt: {e}")
        logger.error(f"[4_RETRIEVE] Traceback: {traceback_string}")
        return None

def evaluate_chunks(prompt: str, chunks: List[Dict[str, Any]], embedded_prompt: List[float] = None) -> List[Dict[str, Any]]:
    """
    Evaluate chunks using multiple comparison methods: vector similarity, regex patterns, and fuzzy matching.
    Now handles chunks without embeddings gracefully.
    """
    if not chunks:
        return []

    try:
        # Create database config and connection for VectorComparison
        config = DatabaseConfig.from_env()
        db_connection = Connection(config)
        
        chunks_with_similarity = []
        
        if not db_connection.connect() or embedded_prompt is None:
            # Fall back to chunks without similarity scores
            for chunk in chunks:
                chunk['similarity_score'] = 0.0
            chunks_with_similarity = chunks
        else:
            # Initialize VectorComparison with proper database connection
            vector_comp = VectorComparison(connection=db_connection)
            
            # Get chunk IDs for batch similarity calculation
            chunk_ids = [chunk['id'] for chunk in chunks]
            
            # Calculate similarities in batch (now handles NULL embeddings)
            similarities = vector_comp.batch_similarity_calculation(
                chunk_ids=chunk_ids,
                query_embedding=embedded_prompt,
                embedding_type='transformer'
            )
            
            # Add similarity scores to chunks
            for chunk in chunks:
                chunk_id = chunk['id']
                similarity_score = similarities.get(chunk_id, 0.0)
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
            vector_weight = 0.40  # Vector similarity has highest weight
            regex_weight = 0.35   # Direct keyword matches
            fuzzy_weight = 0.25   # Fuzzy contextual matching
            
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
        # Return original chunks with 0.0 similarity scores if evaluation fails
        for chunk in chunks:
            if 'similarity_score' not in chunk:
                chunk['similarity_score'] = 0.0
        return chunks

def retrieve_chunks(embedded_prompt, prompt, top_k=20, 
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
        
        # Get all chunks from database (without filtering by country in SQL)
        session = db_connection.get_session()
        try:
            # Get all chunks without country filtering in SQL
            query = text("SELECT * FROM doc_chunks")
            result = session.execute(query)
            all_chunks = []
            
            for row in result:
                # Ensure all potential UUID values are converted to strings
                chunk = {
                    'id': str(row.id),
                    'content': row.content,
                    'doc_id': str(row.doc_id),  # Convert doc_id to string in case it's a UUID
                    'chunk_data': row.chunk_data
                }
                
                # Filter by country in memory using the chunk_data JSON
                if country is not None:
                    chunk_country = chunk.get('chunk_data', {}).get('country', '')
                    if chunk_country.lower() != country.lower():
                        continue
                
                all_chunks.append(chunk)
            
            session.close()
        except Exception as e:
            print(f"[4_RETRIEVE] Error retrieving chunks: {e}")
            if session:
                session.close()
            return []
        
        # Add debugging to see what countries we actually retrieved
        if all_chunks:
            # Extract country from chunk_data JSON
            countries_found = set()
            for chunk in all_chunks:
                chunk_country = chunk.get('chunk_data', {}).get('country', 'Unknown')
                countries_found.add(chunk_country)
            
            print(f"[4_RETRIEVE] DEBUG: Found chunks from {len(countries_found)} countries: {sorted(countries_found)}")
            
            # Show count per country
            country_counts = {}
            for chunk in all_chunks:
                chunk_country = chunk.get('chunk_data', {}).get('country', 'Unknown')
                country_counts[chunk_country] = country_counts.get(chunk_country, 0) + 1
            
            for country_name, count in sorted(country_counts.items()):
                print(f"[4_RETRIEVE] DEBUG: {country_name}: {count} chunks")
        else:
            print("[4_RETRIEVE] DEBUG: No chunks retrieved from database")
            
            # Check if there are any chunks in the database at all
            try:
                session = db_connection.get_session()
                total_count = session.execute(text("SELECT COUNT(*) FROM doc_chunks")).scalar()
                print(f"[4_RETRIEVE] DEBUG: Database has {total_count} total chunks")
                
                # Get list of all countries in database using JSON extraction
                countries_in_db = session.execute(text(
                    "SELECT DISTINCT chunk_data->>'country' FROM doc_chunks WHERE chunk_data->>'country' IS NOT NULL ORDER BY chunk_data->>'country'"
                )).fetchall()
                countries_list = [row[0] for row in countries_in_db if row[0]]
                print(f"[4_RETRIEVE] DEBUG: Countries in database: {countries_list}")
                session.close()
            except Exception as e:
                print(f"[4_RETRIEVE] DEBUG: Error checking database: {e}")

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
        return []


def retrieve_chunks_with_hop(prompt: str, top_k: int = 20, country: Optional[str] = None, 
                           question_number: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve chunks using graph-based multi-hop reasoning through relationship-based navigation.
    
    Args:
        prompt: Query text to search for
        top_k: Number of top chunks to retrieve (default: 20)
        country: Optional country filter
        question_number: Optional question number for keyword enhancement
        
    Returns:
        List of dictionaries containing chunks with scores and metadata
    """
    try:
        print(f"[HOP_RETRIEVE] Starting hop retrieval for query: '{prompt[:50]}...'")
        
        # Enhance query with relevant keywords from HOP_KEYWORDS
        enhanced_query = prompt
        
        if question_number is not None and question_number in HOP_KEYWORDS:
            # Get keywords for this specific question number
            keywords = HOP_KEYWORDS[question_number]
            # Take a subset of keywords to avoid overly long queries
            selected_keywords = " ".join(keywords[:8])
            enhanced_query = f"{selected_keywords}"
            print(f"[HOP_RETRIEVE] Enhanced query with keywords for question {question_number}")
            print(f"[HOP_RETRIEVE] Using keywords: {selected_keywords}")
        else:
            # Use general NDC keywords if no specific question number
            selected_keywords = " ".join(GENERAL_NDC_KEYWORDS[:5])
            enhanced_query = f"{selected_keywords}"
            print(f"[HOP_RETRIEVE] Using general keywords: {selected_keywords}")
        
        # Create database connection
        config = DatabaseConfig.from_env()
        db_connection = Connection(config)
        if not db_connection.connect():
            print("[HOP_RETRIEVE] Failed to connect to database")
            return []
        
        # Check if logical_relationships table has data
        engine = db_connection.get_engine()
        with engine.connect() as conn:
            # Check total relationship count
            rel_count = conn.execute(text("SELECT COUNT(*) FROM logical_relationships")).scalar() or 0
            print(f"[HOP_RETRIEVE] Found {rel_count} total relationships in database")
            
            # Check chunk count for the specified country
            if country:
                country_chunks = conn.execute(
                    text("SELECT COUNT(*) FROM doc_chunks WHERE chunk_data->>'country' = :country"),
                    {"country": country}
                ).scalar() or 0
                print(f"[HOP_RETRIEVE] Found {country_chunks} chunks for country '{country}'")
        
        # Initialize the GraphHopRetriever
        hop_retriever = GraphHopRetriever(connection=db_connection)
        
        # Retrieve chunks using graph-based hop reasoning with enhanced query
        print(f"[HOP_RETRIEVE] Executing hop reasoning retrieval with enhanced query")
        results = hop_retriever.retrieve_with_hop_reasoning(
            query=enhanced_query,
            top_k=top_k,
            country=country
        )
        
        if not results:
            print("[HOP_RETRIEVE] No results returned from hop reasoning")
        else:
            print(f"[HOP_RETRIEVE] Retrieved {len(results)} chunks using hop reasoning")
        
        # Don't save individual files - we'll save one consolidated file per country
        return results
        
    except Exception as e:
        print(f"[HOP_RETRIEVE] Error in graph hop retrieval: {e}")
        traceback_string = traceback.format_exc()
        print(f"[HOP_RETRIEVE] Traceback: {traceback_string}")
        return []

@Logger.log(log_file=project_root / "logs/retrieve.log", log_level="INFO")
def run_script(question_number: int = None, country: Optional[str] = None, 
              use_hop_retrieval: bool = False) -> List[Dict[str, Any]]:
    """
    Main function to run the retrieval script.
    Args:
        question_number (int): The question number (1-8) to use from predefined prompts. If None, runs all questions.
        country (Optional[str]): Optional country name to filter documents by.
        use_hop_retrieval (bool): Whether to use graph-based hop retrieval in addition to vector retrieval.
    
    Returns:
        List[Dict[str, Any]]: List of retrieved chunks with metadata.
    
    """
    try:
        # Create retrieve directories if they don't exist
        retrieve_dir = project_root / "data" / "retrieve"
        retrieve_dir.mkdir(parents=True, exist_ok=True)
        
        retrieve_hop_dir = project_root / "data" / "retrieve_hop"
        retrieve_hop_dir.mkdir(parents=True, exist_ok=True)
        
        all_evaluated_chunks = []
        
        # Determine which questions to run
        if question_number is not None and question_number in QUESTION_PROMPTS:
            questions_to_run = [question_number]
        else:
            questions_to_run = list(range(1, 9))  # Run all questions 1-8
        
        print(f"[4_RETRIEVE] Running questions: {questions_to_run}")
        
        # If no specific country is provided, get all countries from database
        if country is None:
            config = DatabaseConfig.from_env()
            db_connection = Connection(config)
            if db_connection.connect():
                session = db_connection.get_session()
                try:
                    # Extract countries from chunk_data JSON
                    countries_result = session.execute(text(
                        "SELECT DISTINCT chunk_data->>'country' FROM doc_chunks WHERE chunk_data->>'country' IS NOT NULL ORDER BY chunk_data->>'country'"
                    )).fetchall()
                    countries_to_process = [row[0] for row in countries_result if row[0]]
                    session.close()
                except Exception as e:
                    print(f"[4_RETRIEVE] Error getting countries: {e}")
                    countries_to_process = []
                    if session:
                        session.close()
            else:
                countries_to_process = []
        else:
            countries_to_process = [country]
        
        print(f"[4_RETRIEVE] Processing countries: {countries_to_process}")
        
        # Process each country
        for country_name in countries_to_process:
            print(f"[4_RETRIEVE] Processing country: {country_name}")
            
            # Initialize country data structures (separate for standard and hop)
            standard_country_data = {
                "metadata": {
                    "country": country_name,
                    "timestamp": datetime.now().isoformat(),
                    "total_questions": len(questions_to_run),
                    "retrieval_method": "vector_similarity"
                },
                "questions": {}
            }
            
            # Only create hop data if hop retrieval is requested
            hop_country_data = None
            if use_hop_retrieval:
                hop_country_data = {
                    "metadata": {
                        "country": country_name,
                        "timestamp": datetime.now().isoformat(),
                        "total_questions": len(questions_to_run),
                        "retrieval_method": "graph_hop"
                    },
                    "questions": {}
                }
            
            # Process each question for this country
            for q_num in questions_to_run:
                print(f"[4_RETRIEVE] Processing question {q_num} for {country_name}")
                
                # Get the prompt text from the corresponding QUESTION_PROMPT
                prompt = QUESTION_PROMPTS[q_num]
                
                # Always do standard vector-based retrieval for all questions
                # Step 1: Embed the prompt
                embedded_prompt = embed_prompt(prompt)
                
                # Step 2: Retrieve and evaluate chunks for this specific country and question
                standard_chunks = retrieve_chunks(
                    embedded_prompt=embedded_prompt,
                    prompt=prompt,
                    top_k=20,
                    ensure_indices=True,
                    n_per_doc=None,
                    country=country_name,
                    min_similarity=0.2
                )
                
                # Store standard retrieval data
                standard_question_data = {
                    "question_number": q_num,
                    "question": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_count": len(standard_chunks),
                    "top_k_chunks": standard_chunks
                }
                
                standard_country_data["questions"][f"question_{q_num}"] = standard_question_data
                
                # Add to overall collection
                all_evaluated_chunks.extend(standard_chunks)
                
                print(f"[4_RETRIEVE] Question {q_num} for {country_name}: {len(standard_chunks)} chunks retrieved (vector)")
                
                # If hop retrieval is requested, also perform hop retrieval
                if use_hop_retrieval:
                    print(f"[4_RETRIEVE] Using graph hop retrieval for question {q_num}")
                    hop_chunks = retrieve_chunks_with_hop(
                        prompt=prompt,
                        top_k=20,
                        country=country_name,
                        question_number=q_num  # Pass question number to hop retrieval
                    )
                    
                    # Store hop retrieval data
                    hop_question_data = {
                        "question_number": q_num,
                        "question": prompt,
                        "timestamp": datetime.now().isoformat(),
                        "chunk_count": len(hop_chunks),
                        "top_k_chunks": hop_chunks
                    }
                    
                    hop_country_data["questions"][f"question_{q_num}"] = hop_question_data
                    
                    # Add hop chunks to overall collection as well
                    all_evaluated_chunks.extend(hop_chunks)
                    
                    print(f"[4_RETRIEVE] Question {q_num} for {country_name}: {len(hop_chunks)} chunks retrieved (hop)")
            
            # Save country file with all questions
            # Clean country name for filename
            clean_country_name = "".join(c for c in country_name if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_country_name = clean_country_name.replace(' ', '_')
            
            # Create the filenames
            retrieve_filename = f"{clean_country_name}.json"
            
            # Always save standard vector results
            standard_output_path = retrieve_dir / retrieve_filename
            print(f"[4_RETRIEVE] Saving vector retrieval results to: {standard_output_path}")
            
            with open(standard_output_path, "w", encoding="utf-8") as f:
                # Use the custom encoder to handle UUID objects
                json.dump(standard_country_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)
            
            standard_total_chunks = sum(len(q_data["top_k_chunks"]) for q_data in standard_country_data["questions"].values())
            print(f"[4_RETRIEVE] Saved {len(questions_to_run)} questions with {standard_total_chunks} total chunks for {country_name} (vector)")
            
            # If hop retrieval was used, also save hop results
            if use_hop_retrieval and hop_country_data:
                hop_output_path = retrieve_hop_dir / retrieve_filename
                print(f"[4_RETRIEVE] Saving hop retrieval results to: {hop_output_path}")
                
                with open(hop_output_path, "w", encoding="utf-8") as f:
                    # Use the custom encoder to handle UUID objects
                    json.dump(hop_country_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)
                
                hop_total_chunks = sum(len(q_data["top_k_chunks"]) for q_data in hop_country_data["questions"].values())
                print(f"[4_RETRIEVE] Saved {len(questions_to_run)} questions with {hop_total_chunks} total chunks for {country_name} (hop)")
        
        print(f"[4_RETRIEVE] Completed processing {len(countries_to_process)} countries with {len(questions_to_run)} questions each")
        return all_evaluated_chunks

    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_RETRIEVE] Error in run_script: {e}")
        logger.error(f"[4_RETRIEVE] Traceback: {traceback_string}")
        return []

if __name__ == "__main__":
    """
    Usage examples:
    python 4_retrieve.py --question 1 --country "Japan"         > Creates only standard vector retrieval JSONs
    python 4_retrieve.py --question 4                           > Creates standard JSONs for all countries
    python 4_retrieve.py --question 1 --country "Japan" --hop   > Creates both vector and hop retrieval JSONs
    """

    parser = argparse.ArgumentParser(description='Run the retrieval script with a specified question number.')
    parser.add_argument('--question', type=int, choices=range(1, 9), 
                        help='Question number (1-8) to select a predefined prompt.')
    parser.add_argument('--country', type=str, help='Country to filter documents by (optional).')
    parser.add_argument('--hop', action='store_true', 
                       help='Use graph-based hop retrieval in addition to vector retrieval.')
    args = parser.parse_args()
    run_script(question_number=args.question, country=args.country, use_hop_retrieval=args.hop)