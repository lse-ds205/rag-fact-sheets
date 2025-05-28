import sys
from pathlib import Path
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple  
import argparse
import json
from dotenv import load_dotenv
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.query import LLMClient, ResponseProcessor, ChunkFormatter, ConfidenceClassification

logger = logging.getLogger(__name__)
load_dotenv()

def setup_llm(supports_guided_json: bool = True) -> LLMClient:
    """
    Setup and initialize the LLM client.
    
    Args:
        supports_guided_json: Whether the LLM API supports guided JSON responses
        
    Returns:
        Initialized LLMClient instance
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Setting up LLM client (guided_json: {supports_guided_json})...")
        
        llm_client = LLMClient(supports_guided_json=supports_guided_json)
        
        if not llm_client.client:
            raise Exception("Failed to initialize LLM client - check API credentials")
            
        logger.info("[5_LLM_RESPONSE] LLM client setup completed successfully")
        return llm_client
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in setup_llm: {e}\nTraceback: {traceback_string}")
        raise e

def load_chunks_and_prompt_from_file(filepath: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load chunks and prompt from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing chunks and prompt
        
    Returns:
        Tuple of (chunks_list, prompt_string)
        
    Raises:
        ValueError: If prompt is not found in the JSON structure
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Loading chunks and prompt from file: {filepath}")
        
        # Convert to Path object for better handling
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {filepath}")
            
        if not file_path.suffix.lower() == '.json':
            raise ValueError(f"File must be a JSON file, got: {file_path.suffix}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract prompt and chunks based on JSON structure
        prompt = None
        chunks = None
        
        if isinstance(data, dict):
            # First check for the metadata structure from 4_retrieve.py
            if 'metadata' in data and isinstance(data['metadata'], dict):
                metadata = data['metadata']
                prompt = metadata.get('query_text') or metadata.get('prompt') or metadata.get('query')
            
            # If no prompt found in metadata, look for direct keys
            if not prompt:
                prompt_keys = ['prompt', 'query', 'question', 'user_prompt', 'user_query']
                for key in prompt_keys:
                    if key in data and data[key]:
                        prompt = data[key]
                        break
            
            # Look for chunks - check for evaluated_chunks first (from 4_retrieve.py)
            if 'evaluated_chunks' in data and isinstance(data['evaluated_chunks'], list):
                chunks = data['evaluated_chunks']
            else:
                # Look for other chunk keys
                chunk_keys = ['chunks', 'retrieved_chunks', 'context_chunks', 'documents']
                for key in chunk_keys:
                    if key in data and isinstance(data[key], list):
                        chunks = data[key]
                        break
            
            # If chunks not found in nested structure, check if the entire dict is chunk data
            if chunks is None and 'content' in data:
                # Single chunk as dict
                chunks = [data]
        elif isinstance(data, list):
            # Direct list of chunks - prompt should be in metadata or first chunk
            chunks = data
            # Try to find prompt in the first chunk's metadata
            if chunks and isinstance(chunks[0], dict):
                chunk_meta = chunks[0].get('metadata', {}) or chunks[0].get('chunk_metadata', {})
                if isinstance(chunk_meta, dict):
                    prompt = chunk_meta.get('prompt') or chunk_meta.get('query') or chunk_meta.get('query_text')
        else:
            raise ValueError(f"Invalid JSON structure in file: {filepath}")
        
        # Validate that we found both prompt and chunks
        if not prompt:
            # Additional debugging for the specific structure
            logger.error(f"[5_LLM_RESPONSE] JSON structure debug: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            if isinstance(data, dict) and 'metadata' in data:
                logger.error(f"[5_LLM_RESPONSE] Metadata keys: {list(data['metadata'].keys())}")
            raise ValueError(f"No prompt found in JSON file. Expected keys: ['query_text' in metadata, 'prompt', 'query', 'question', 'user_prompt', 'user_query']")
        if not chunks:
            raise ValueError(f"No chunks found in JSON file. Expected keys: ['evaluated_chunks', 'chunks', 'retrieved_chunks', 'context_chunks', 'documents']")
        if not isinstance(chunks, list):
            raise ValueError(f"Chunks must be a list, got: {type(chunks)}")
        
        logger.info(f"[5_LLM_RESPONSE] Successfully loaded {len(chunks)} chunks and prompt from file")
        logger.info(f"[5_LLM_RESPONSE] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        return chunks, prompt
        
    except FileNotFoundError as e:
        logger.error(f"[5_LLM_RESPONSE] File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[5_LLM_RESPONSE] Invalid JSON in file {filepath}: {e}")
        raise
    except ValueError as e:
        logger.error(f"[5_LLM_RESPONSE] Data structure error: {e}")
        raise
    except Exception as e:
        logger.error(f"[5_LLM_RESPONSE] Error loading chunks and prompt from file {filepath}: {e}")
        raise

def extract_main_question(query_text: str) -> str:
    """
    Extract the main question from a detailed query that may contain bullet points.
    
    Args:
        query_text: Full query text that may contain detailed instructions
        
    Returns:
        Clean main question without bullet points or detailed instructions
    """
    # Split by common separators that indicate additional instructions
    separators = ['\n\nPlease extract:', '\n\nPlease identify:', '\n\nPlease analyze:', '\n\nPlease determine:']
    
    main_question = query_text
    for separator in separators:
        if separator in query_text:
            main_question = query_text.split(separator)[0].strip()
            break
    
    # Clean up any remaining formatting
    main_question = main_question.strip().rstrip('?') + '?'
    
    return main_question

def get_llm_response(llm_client: LLMClient, question: str, chunks: List[Dict[str, Any]]) -> Any:
    """
    Get structured response from LLM using the provided chunks and question.
    
    Args:
        llm_client: Initialized LLM client
        question: User's question/prompt (can be detailed)
        chunks: List of chunk dictionaries to use as context
        
    Returns:
        Structured LLM response (LLMResponseModel or None on error)
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Getting LLM response for question: {question[:100]}...")
        logger.info(f"[5_LLM_RESPONSE] Using {len(chunks)} chunks as context")
        
        # Format chunks for LLM
        chunk_formatter = ChunkFormatter()
        formatted_context = chunk_formatter.format_chunks_for_context(chunks)
        
        # Create prompt using the full detailed question
        full_prompt = llm_client.create_llm_prompt(question, formatted_context)
        
        # Get structured LLM response
        structured_response = llm_client.call_llm(full_prompt)
        
        if not structured_response:
            logger.error("[5_LLM_RESPONSE] No response from LLM")
            return None
        
        logger.info("[5_LLM_RESPONSE] Successfully received structured LLM response")
        return structured_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in get_llm_response: {e}\nTraceback: {traceback_string}")
        return None

def process_response(llm_response: Any, original_chunks: List[Dict[str, Any]], question: str, main_question: str = None) -> Dict[str, Any]:
    """
    Process and validate the LLM response into final JSON format.

    Args:
        llm_response: Structured response from LLM (LLMResponseModel or None)
        original_chunks: Original chunk data for validation
        question: Original detailed question for error context
        main_question: Short version of the question for JSON storage
        
    Returns:
        Final JSON-serializable response dictionary
    """
    try:
        logger.info("[5_LLM_RESPONSE] Processing LLM response...")
        
        # Create response processor
        response_processor = ResponseProcessor()
        
        # Process the response with original chunks for citation validation
        final_response = response_processor.process_llm_response(
            llm_response=llm_response,
            original_chunks=original_chunks
        )
        
        # Override the question field with the short version if provided
        if main_question and 'question' in final_response:
            final_response['question'] = main_question
        
        logger.info("[5_LLM_RESPONSE] Successfully processed LLM response")
        return final_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in process_response: {e}\nTraceback: {traceback_string}")
        
        # Return error response with main question if available
        return {
            "answer": f"Error processing response: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "question": main_question or question,
            "error": True,
            "error_details": str(e)
        }

def process_country_file(file_path: Path) -> Dict[str, Any]:
    """
    Process a single country retrieve file and generate LLM responses.
    
    Args:
        file_path: Path to the country retrieve file
        
    Returns:
        Dictionary containing the processed responses
    """
    try:
        # Extract country name from filename
        country_name = file_path.stem.split('_')[0]
        logger.info(f"Processing country: {country_name}")
        
        # Load the country retrieve data
        with open(file_path, 'r', encoding='utf-8') as f:
            retrieve_data = json.load(f)
        
        # Initialize LLM client and confidence classifier
        llm_client = LLMClient()
        confidence_classifier = ConfidenceClassification()
        
        # Process each question in the retrieve data
        results = {}
        for question_id, question_data in retrieve_data.get('questions', {}).items():
            logger.info(f"Processing question ID {question_id}")
            
            # Get the full query text
            full_query_text = question_data.get('question')
            
            if not full_query_text:
                logger.warning(f"Question ID {question_id} is missing the question text")
                continue
            
            # Extract just the main question for JSON storage
            main_question = extract_main_question(full_query_text)
            logger.info(f"Main question extracted: {main_question}")
            
            # Extract top k chunks for the question
            top_k_chunks = question_data.get('top_k_chunks', [])
            
            # Get LLM response using the FULL detailed query (not just main question)
            llm_response = get_llm_response(llm_client, full_query_text, top_k_chunks)
            
            # Process the LLM response, but store the main question in the JSON
            processed_response = process_response(llm_response, top_k_chunks, full_query_text, main_question)
            
            # Store the processed response
            results[question_id] = processed_response
        
        # After processing all questions, classify responses
        results = confidence_classifier.classify_response(results, retrieve_data)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "metadata": {
                "country_name": file_path.stem.split('_')[0],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            "questions": {}
        }

def main():
    """
    Main execution function for the LLM response pipeline.
    """
    try:
        # Setup logging
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "llm_response.log"
        
        # Configure logging to both console and file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info("[5_LLM_RESPONSE] Starting LLM response pipeline...")
        
        # Set up directories
        retrieve_dir = project_root / "data" / "retrieve"
        llm_output_dir = project_root / "data" / "llm"
        
        # Create output directory if it doesn't exist
        llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files in the retrieve directory
        retrieve_files = list(retrieve_dir.glob("*.json"))
        
        if not retrieve_files:
            logger.error(f"No JSON files found in {retrieve_dir}")
            return
        
        logger.info(f"Found {len(retrieve_files)} retrieve files to process")
        
        # Process each retrieve file
        for retrieve_file in retrieve_files:
            try:
                logger.info(f"Processing file: {retrieve_file.name}")
                
                # Process the country file
                llm_results = process_country_file(retrieve_file)
                
                # Extract country name for output filename
                country_name = retrieve_file.stem.split('_')[0]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{country_name}_{timestamp}.json"
                output_path = llm_output_dir / output_filename
                
                # Load original retrieve data to get query texts
                with open(retrieve_file, 'r', encoding='utf-8') as f:
                    retrieve_data = json.load(f)
                
                # Create the final output structure
                final_output = {
                    "metadata": {
                        "country_name": country_name,
                        "timestamp": datetime.now().isoformat(),
                        "source_file": retrieve_file.name,
                        "description": f"LLM-generated responses for {country_name} based on retrieved chunks",
                        "question_count": 0
                    },
                    "questions": {}
                }
                
                # Process each question result - llm_results keys are question IDs
                question_count = 0
                for question_id, question_result in llm_results.items():
                    if question_id == "metadata":  # Skip metadata from process_country_file
                        continue
                    
                    question_count += 1
                    
                    # Get corresponding retrieve data for this question
                    retrieve_question_data = retrieve_data.get('questions', {}).get(question_id, {})
                    query_text = retrieve_question_data.get('question', 'Unknown')
                    
                    # Build the question structure
                    final_output["questions"][question_id] = {
                        "question_number": int(question_id.split('_')[1]) if '_' in question_id else question_count,
                        "query_text": query_text,
                        "llm_response": question_result,
                        "chunk_count": len(question_result.get('citations', []))
                    }
                
                # Update question count in metadata
                final_output["metadata"]["question_count"] = question_count
                
                # Save the LLM response file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved LLM responses to: {output_path} ({question_count} questions processed)")
                
            except Exception as e:
                logger.error(f"Error processing {retrieve_file.name}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info("[5_LLM_RESPONSE] LLM response pipeline completed successfully")
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in main: {e}\nTraceback: {traceback_string}")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LLM response script with chunks and prompt from evaluated_chunks.json.')
    
    # Optional arguments
    parser.add_argument('--prompt', type=str, 
                       help='Optional prompt override (if not provided, reads from JSON file)')
    parser.add_argument('--no-guided-json', action='store_true', 
                       help='Disable guided JSON (use fallback parsing)')
    
    # Legacy support for old argument names
    parser.add_argument('--chunks', type=str, 
                       help='JSON string of chunks (legacy support).')
    
    args = parser.parse_args()
    
    try:
        if args.chunks:
            # Legacy support: create temporary file from JSON string
            logger.info("Using legacy JSON string input (--chunks parameter)")
            
            try:
                chunks_list = json.loads(args.chunks)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing chunks JSON string: {e}")
                print(json.dumps({"error": f"Invalid chunks JSON: {str(e)}"}, indent=2, ensure_ascii=False))
                sys.exit(1)
            
            # Create temporary file with both chunks and prompt
            import tempfile
            temp_data = {
                'chunks': chunks_list,
                'prompt': args.prompt or "No prompt provided"
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(temp_data, temp_file, indent=2)
                temp_filepath = temp_file.name
            
            try:
                # Override the chunks file path for main()
                sys.argv = [sys.argv[0], temp_filepath]
                main()
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_filepath)
        else:
            # Primary method: process retrieve files directly
            main()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(json.dumps({"error": f"Execution error: {str(e)}"}, indent=2, ensure_ascii=False))
        sys.exit(1)

    # Usage examples:
    # Primary method (processes all files from data/retrieve directory):
    # python 5_llm_response.py
    #
    # With prompt override:
    # python 5_llm_response.py --prompt "Override question?"
    #
    # Without guided JSON (fallback mode):
    # python 5_llm_response.py --no-guided-json
    #
    # Legacy method (JSON string with prompt required):
    # python 5_llm_response.py --chunks '[{"id": 1, "content": "test"}]' --prompt "What is this about?"