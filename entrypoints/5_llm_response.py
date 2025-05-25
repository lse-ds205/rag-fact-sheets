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
            # Look for prompt in various possible keys
            prompt_keys = ['prompt', 'query', 'question', 'user_prompt', 'user_query']
            for key in prompt_keys:
                if key in data and data[key]:
                    prompt = data[key]
                    break
            
            # Look for chunks in various possible keys
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
                    prompt = chunk_meta.get('prompt') or chunk_meta.get('query')
        else:
            raise ValueError(f"Invalid JSON structure in file: {filepath}")
        
        # Validate that we found both prompt and chunks
        if not prompt:
            raise ValueError(f"No prompt found in JSON file. Expected keys: {prompt_keys}")
        if not chunks:
            raise ValueError(f"No chunks found in JSON file. Expected keys: {chunk_keys}")
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
def run_script(chunks_filepath: str, prompt_override: Optional[str] = None, supports_guided_json: bool = True) -> Dict[str, Any]:
    """
    Main script function.
    General flow: Load chunks and prompt -> Setup LLM -> Get response -> Process response.
    
    Args:
        chunks_filepath (str): Path to JSON file containing chunks and prompt
        prompt_override (Optional[str]): Optional prompt override (if None, reads from JSON file)
        supports_guided_json (bool): Whether the LLM API supports guided JSON responses
        
    Returns:
        JSON-serializable LLM response dictionary
    """
    try:
        logger.warning(f"\n\n[5_LLM_RESPONSE] Running script...")
        logger.info(f"[5_LLM_RESPONSE] Chunks file: {chunks_filepath}")
        logger.info(f"[5_LLM_RESPONSE] Guided JSON: {supports_guided_json}")
        
        # Validate inputs
        if not chunks_filepath:
            raise ValueError("Chunks filepath is required")
        
        # Step 1: Load chunks and prompt from file
        chunks, prompt = load_chunks_and_prompt_from_file(chunks_filepath)
        
        # Use prompt override if provided
        if prompt_override:
            logger.info(f"[5_LLM_RESPONSE] Using prompt override instead of JSON prompt")
            prompt = prompt_override
        
        logger.warning(f"[5_LLM_RESPONSE] Using prompt: {prompt}")
        
        # Step 2: Setup LLM
        llm_client = setup_llm(supports_guided_json=supports_guided_json)
        
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
        fallback_prompt = prompt_override or "Unknown question"
        return ResponseProcessor._create_error_response(f"Pipeline error: {str(e)}", fallback_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LLM response script with chunks and prompt from a JSON file.')
    
    # Primary arguments (required)
    parser.add_argument('--chunks-file', type=str, required=True, 
                       help='Path to JSON file containing chunks and prompt.')
    
    # Optional arguments
    parser.add_argument('--prompt', type=str, 
                       help='Optional prompt override (if not provided, reads from JSON file)')
    parser.add_argument('--no-guided-json', action='store_true', 
                       help='Disable guided JSON (use fallback parsing)')
    
    # Legacy support for old argument names
    parser.add_argument('--chunks', type=str, 
                       help='JSON string of chunks (legacy support, overrides --chunks-file if provided).')
    
    args = parser.parse_args()
    
    output_file = Path(args.chunks_file).with_name(Path(args.chunks_file).stem + '_response.json')
    try:
        # Determine guided JSON support
        supports_guided_json = not args.no_guided_json
        
        if args.chunks:
            # Legacy support: create temporary file from JSON string
            logger.info("Using legacy JSON string input (--chunks parameter)")
            
            try:
                chunks_list = json.loads(args.chunks)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing chunks JSON string: {e}")
                error_response = ResponseProcessor._create_error_response(f"Invalid chunks JSON: {str(e)}", args.prompt or "Unknown")
                print(json.dumps(error_response, indent=2, ensure_ascii=False))
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
                result = run_script(
                    chunks_filepath=temp_filepath, 
                    prompt_override=args.prompt,  # Use prompt override for legacy mode
                    supports_guided_json=supports_guided_json
                )
            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_filepath)
        else:
            # Primary method: use file directly, prompt read from JSON
            result = run_script(
                chunks_filepath=args.chunks_file, 
                prompt_override=args.prompt,  # Optional override
                supports_guided_json=supports_guided_json
            )
        
        # Print result for command line usage
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Dump to json file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
    except FileNotFoundError as e:
        logger.error(f"Chunks file not found: {e}")
        error_response = ResponseProcessor._create_error_response(f"File not found: {str(e)}", args.prompt or "Unknown")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        error_response = ResponseProcessor._create_error_response(f"Data error: {str(e)}", args.prompt or "Unknown")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        error_response = ResponseProcessor._create_error_response(f"Execution error: {str(e)}", args.prompt or "Unknown")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
        sys.exit(1)

    # Usage examples:
    # Primary method (reads prompt from JSON file):
    # python 5_llm_response.py --chunks-file data/chunks_with_prompt.json
    #
    # With prompt override:
    # python 5_llm_response.py --chunks-file data/chunks_with_prompt.json --prompt "Override question?"
    #
    # Without guided JSON (fallback mode):
    # python 5_llm_response.py --chunks-file data/chunks_with_prompt.json --no-guided-json
    #
    # Legacy method (JSON string with prompt required):
    # python 5_llm_response.py --chunks '[{"id": 1, "content": "test"}]' --prompt "What is this about?"