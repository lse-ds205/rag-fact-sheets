import sys
from pathlib import Path
import logging
import traceback
from typing import List, Dict, Any, Optional, Union, Tuple  
import argparse
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.query import LLMClient, ResponseProcessor, ChunkFormatter, ConfidenceClassification

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
        
        # Create response processor
        response_processor = ResponseProcessor()
        
        # Process the response with original chunks for citation validation
        final_response = response_processor.process_llm_response(
            llm_response=llm_response,
            original_chunks=original_chunks
        )
        
        logger.info("[5_LLM_RESPONSE] Successfully processed LLM response")
        return final_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in process_response: {e}\nTraceback: {traceback_string}")
        
        # Return error response
        return {
            "answer": f"Error processing response: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "question": question,
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
            
            # Get the question text
            question_text = question_data.get('question')
            
            if not question_text:
                logger.warning(f"Question ID {question_id} is missing the question text")
                continue
            
            # Extract top k chunks for the question
            top_k_chunks = question_data.get('top_k_chunks', [])
            
            # Get LLM response for the question
            llm_response = get_llm_response(llm_client, question_text, top_k_chunks)
            
            # Process the LLM response
            processed_response = process_response(llm_response, top_k_chunks, question_text)
            
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("[5_LLM_RESPONSE] Starting LLM response pipeline...")
        
        # Default file path for chunks data
        chunks_file_path = "data/evaluated_chunks.json"
        
        # Check if file path provided as command line argument
        if len(sys.argv) > 1:
            chunks_file_path = sys.argv[1]
        
        # Convert to absolute path
        file_path = Path(chunks_file_path)
        if not file_path.is_absolute():
            file_path = Path(project_root) / chunks_file_path
        
        logger.info(f"[5_LLM_RESPONSE] Loading chunks from: {file_path}")
        
        # Load chunks and prompt from file
        chunks, prompt = load_chunks_and_prompt_from_file(str(file_path))
        
        # Setup LLM client
        llm_client = setup_llm(supports_guided_json=True)
        
        # Get LLM response
        llm_response = get_llm_response(llm_client, prompt, chunks)
        
        if llm_response is None:
            logger.error("[5_LLM_RESPONSE] Failed to get LLM response")
            return
        
        # Process the response
        final_response = process_response(llm_response, chunks, prompt)
        
        # Save the final response
        output_file = project_root / "data" / "llm_response.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_response, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[5_LLM_RESPONSE] Final response saved to: {output_file}")
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
    
    output_file = Path(args.chunks_file).with_name(Path(args.chunks_file).stem + '_response.json')
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
            # Primary method: use evaluated_chunks.json automatically
            chunks_file_path = project_root / "data" / "evaluated_chunks.json"
            
            if not chunks_file_path.exists():
                logger.error(f"Evaluated chunks file not found: {chunks_file_path}")
                print(json.dumps({"error": f"File not found: {chunks_file_path}"}, indent=2, ensure_ascii=False))
                sys.exit(1)
            
            # Run main function directly
            main()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(json.dumps({"error": f"Execution error: {str(e)}"}, indent=2, ensure_ascii=False))
        sys.exit(1)

    # Usage examples:
    # Primary method (reads from data/evaluated_chunks.json automatically):
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