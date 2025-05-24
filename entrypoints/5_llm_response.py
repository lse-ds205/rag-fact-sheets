import sys
from pathlib import Path
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple  
import argparse
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.query import ChunkFormatter, LLMClient, ResponseProcessor

logger = logging.getLogger(__name__)

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

def load_chunks_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing chunks
        
    Returns:
        List of chunk dictionaries
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Loading chunks from file: {filepath}")
        
        # Convert to Path object for better handling
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {filepath}")
            
        if not file_path.suffix.lower() == '.json':
            raise ValueError(f"File must be a JSON file, got: {file_path.suffix}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of chunks
            chunks = data
        elif isinstance(data, dict) and 'chunks' in data:
            # Structured format with chunks key
            chunks = data['chunks']
        elif isinstance(data, dict):
            # Single chunk as dict
            chunks = [data]
        else:
            raise ValueError(f"Invalid JSON structure in file: {filepath}")
        
        logger.info(f"[5_LLM_RESPONSE] Successfully loaded {len(chunks)} chunks from file")
        return chunks
        
    except FileNotFoundError as e:
        logger.error(f"[5_LLM_RESPONSE] File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"[5_LLM_RESPONSE] Invalid JSON in file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"[5_LLM_RESPONSE] Error loading chunks from file {filepath}: {e}")
        raise

def get_llm_response(llm_client: LLMClient, question: str, chunks: List[Dict[str, Any]]) -> Any:
    """
    Get structured response from LLM using the provided chunks and question.
    
    Args:
        llm_client: Initialized LLM client
        question: User's question/prompt
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
        
        # Create prompt
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

def process_response(llm_response: Any, original_chunks: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    """
    Process and validate the LLM response into final JSON format.

    Accepts the structured response from LLM, and original top k chunks. Looks up chunks in the response by ID, then adds chunks to citations.
    Formats the final response with accurate citations.

    Args:
        llm_response: Structured response from LLM (LLMResponseModel or None)
        original_chunks: Original chunk data for validation
        question: Original question for error context
        
    Returns:
        Final JSON-serializable response dictionary
    """
    try:
        logger.info("[5_LLM_RESPONSE] Processing LLM response...")
        
        if not llm_response:
            logger.error("[5_LLM_RESPONSE] No LLM response to process")
            return ResponseProcessor._create_error_response("No response from LLM service", question)
        
        # Process and validate response
        response_processor = ResponseProcessor()
        final_response = response_processor.process_llm_response(llm_response, original_chunks)
        
        logger.info(f"[5_LLM_RESPONSE] Response processed successfully with {len(final_response.get('citations', []))} citations")
        return final_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in process_response: {e}\nTraceback: {traceback_string}")
        
        # Return error response
        return ResponseProcessor._create_error_response(f"Error processing response: {str(e)}", question)

@Logger.log(log_file=project_root / "logs/llm_response.log", log_level="DEBUG")

# NEED TO CHANGE TO CHUNKS INSTEAD OF FILEPATH

def run_script(chunks_filepath: str, prompt: str, supports_guided_json: bool = True) -> Dict[str, Any]:
    """
    Main script function.
    General flow: Setup LLM -> Load chunks -> Get response -> Process response.
    
    Args:
        chunks_filepath (str): Path to JSON file containing chunks
        chunks (Optional[List[Dict]]): Top k relevant chunks as list of chunk dicts (NOT IMPLEMETNED YET)
        prompt (str): User's question/prompt
        supports_guided_json (bool): Whether the LLM API supports guided JSON responses
        
    Returns:
        JSON-serializable LLM response dictionary
    """
    try:
        logger.warning(f"\n\n[5_LLM_RESPONSE] Running script...")
        logger.warning(f"[5_LLM_RESPONSE] Prompt: {prompt}")
        logger.info(f"[5_LLM_RESPONSE] Chunks file: {chunks_filepath}")
        logger.info(f"[5_LLM_RESPONSE] Guided JSON: {supports_guided_json}")
        
        # Validate inputs
        if not prompt:
            raise ValueError("Prompt is required")
        if not chunks_filepath:
            raise ValueError("Chunks filepath is required")
        
        # Step 1: Setup LLM
        llm_client = setup_llm(supports_guided_json=supports_guided_json)
        
        # Step 2: Load chunks
        chunks = load_chunks_from_file(chunks_filepath)
        
        # Step 3: Get response
        llm_response = get_llm_response(llm_client, prompt, chunks)
        
        # Step 4: Process response
        final_response = process_response(llm_response, chunks, prompt)
        
        # Log summary
        if "error" not in final_response:
            logger.warning(f"[5_LLM_RESPONSE] Script completed successfully.")
            logger.info(f"[5_LLM_RESPONSE] Generated answer with {final_response.get('metadata', {}).get('chunks_cited', 0)} citations")
            logger.info(f"[5_LLM_RESPONSE] Primary countries: {final_response.get('metadata', {}).get('primary_countries', [])}")
        else:
            logger.error(f"[5_LLM_RESPONSE] Script completed with error: {final_response.get('error', 'Unknown error')}")
        
        logger.warning("[5_LLM_RESPONSE] Script finished. Exiting.")
        return final_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 5_llm_response.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        
        # Return error response even if pipeline breaks
        return ResponseProcessor._create_error_response(f"Pipeline error: {str(e)}", prompt or "Unknown question")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LLM response script with chunks from a JSON file.')
    
    # Primary arguments (required)
    parser.add_argument('--chunks-file', type=str, required=True, help='Path to JSON file containing chunks.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt/question to execute the script for.')
    
    # Optional arguments
    parser.add_argument('--no-guided-json', action='store_true', help='Disable guided JSON (use fallback parsing)')
    parser.add_argument('--chunks', type=str, help='JSON string of chunks (legacy support, overrides --chunks-file if provided).')
    
    args = parser.parse_args()
    
    try:
        # Determine guided JSON support
        supports_guided_json = not args.no_guided_json
        
        if args.chunks:
            # Legacy support: create temporary file from JSON string
            logger.info("Using legacy JSON string input (--chunks parameter)")
            chunks_list = json.loads(args.chunks)
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(chunks_list, temp_file, indent=2)
                temp_filepath = temp_file.name
            
            try:
                result = run_script(
                    chunks_filepath=temp_filepath, 
                    prompt=args.prompt,
                    supports_guided_json=supports_guided_json
                )
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_filepath)
        else:
            # Primary method: use file directly
            result = run_script(
                chunks_filepath=args.chunks_file, 
                prompt=args.prompt,
                supports_guided_json=supports_guided_json
            )
        
        # Print result for command line usage
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing chunks JSON: {e}")
        error_response = ResponseProcessor._create_error_response(f"Invalid chunks JSON: {str(e)}", args.prompt)
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
    except FileNotFoundError as e:
        logger.error(f"Chunks file not found: {e}")
        error_response = ResponseProcessor._create_error_response(f"File not found: {str(e)}", args.prompt)
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        error_response = ResponseProcessor._create_error_response(f"Execution error: {str(e)}", args.prompt)
        print(json.dumps(error_response, indent=2, ensure_ascii=False))

    # Usage examples:
    # Primary method (file-based with guided JSON):
    # python 5_llm_response.py --chunks-file data/retrieved_chunks.json --prompt "What are the main climate policies?"
    #
    # Without guided JSON (fallback mode):
    # python 5_llm_response.py --chunks-file data/retrieved_chunks.json --prompt "Question?" --no-guided-json
    #
    # Legacy method (JSON string):
    # python 5_llm_response.py --chunks-file dummy.json --chunks '[{"id": 1, "content": "test"}]' --prompt "What is this about?"
    
    
    # When argparse is added, can no longer press the 'run button' in VSCode/Cursor.
    # Instead, need to python xx.py --argument
    # This is to, potentially, allow an alternative way of bridging, 
    #   of the end-to-end communication between interface.py/Github Actions and these sub-entrypoints
    #   because xx.py --argument could work better than calling the function directly
    # E.g. use in terminal: python 5_llm_response.py --chunks '[{"id": 1, "content": "test"}]' --prompt "What is this about?"