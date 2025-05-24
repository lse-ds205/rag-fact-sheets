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
from group4py.src.schema import LLMAnswerModel, LLMCitationModel, LLMMetadataModel, LLMResponseModel
logger = logging.getLogger(__name__)

def filter_chunks(top_selected_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and prepare chunks for LLM processing.
    Currently returns all chunks, but can be extended for filtering logic.
    
    Args:
        top_selected_chunks: List of chunk dictionaries from retrieval
        
    Returns:
        Filtered list of chunks
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Filtering {len(top_selected_chunks)} chunks...")
        
        # For now, we'll use all provided chunks
        # Future enhancements could include:
        # - Removing duplicate content
        # - Filtering by minimum similarity score
        # - Grouping by document/country
        # - Selecting diverse chunks
        
        filtered_chunks = top_selected_chunks
        logger.info(f"[5_LLM_RESPONSE] Filtered to {len(filtered_chunks)} chunks.")
        return filtered_chunks
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in filter_chunks: {e}\nTraceback: {traceback_string}")
        raise e

def choose_chunk(engineered_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Choose the final set of chunks to send to LLM.
    Currently returns all chunks for maximum context.
    
    Args:
        engineered_chunks: List of filtered chunks
        
    Returns:
        Final list of chunks for LLM
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Choosing from {len(engineered_chunks)} chunks...")
        
        # For now, we'll use all engineered chunks to provide full context
        # Future enhancements could include:
        # - Selecting top N chunks by similarity
        # - Ensuring representation from different documents
        # - Balancing chunk types (titles, content, etc.)
        
        chosen_chunks = engineered_chunks
        logger.info(f"[5_LLM_RESPONSE] Chose {len(chosen_chunks)} chunks for LLM.")
        return chosen_chunks
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in choose_chunk: {e}\nTraceback: {traceback_string}")
        raise e

def give_answer(question: str, chosen_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate answer using LLM with the provided chunks and question.
    
    Args:
        question: User's question/prompt
        chosen_chunks: List of chunk dictionaries to use as context
        
    Returns:
        Structured response dictionary from LLM
    """
    try:
        logger.info(f"[5_LLM_RESPONSE] Generating answer for question: {question[:100]}...")
        logger.info(f"[5_LLM_RESPONSE] Using {len(chosen_chunks)} chunks as context")
        
        # Format chunks for LLM
        chunk_formatter = ChunkFormatter()
        formatted_context = chunk_formatter.format_chunks_for_context(chosen_chunks)
        
        # Initialize LLM client
        llm_client = LLMClient()
        
        # Create prompt
        full_prompt = llm_client.create_llm_prompt(question, formatted_context)
        
        # Get structured LLM response
        structured_response = llm_client.call_llm(full_prompt)
        
        if not structured_response:
            logger.error("[5_LLM_RESPONSE] No response from LLM")
            return ResponseProcessor._create_error_response("No response from LLM service", question)
        
        # Process and validate response
        response_processor = ResponseProcessor()
        final_response = response_processor.process_llm_response(structured_response, chosen_chunks)
        
        logger.info(f"[5_LLM_RESPONSE] Answer generated successfully with {len(final_response.get('citations', []))} citations")
        return final_response
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[5_LLM_RESPONSE] Error in give_answer: {e}\nTraceback: {traceback_string}")
        
        # Return error response
        return ResponseProcessor._create_error_response(f"Error generating answer: {str(e)}", question)

@Logger.log(log_file=project_root / "logs/llm_response.log", log_level="DEBUG")
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

def run_script(chunks_filepath: str = None, top_selected_chunks: List[Dict[str, Any]] = None, prompt: str = None) -> Dict[str, Any]:
    """
    Main script function to process chunks and generate LLM response.
    
    Args:
        top_selected_chunks: List of chunk dictionaries from retrieval
        prompt: User's question/prompt
        
    Returns:
        Structured JSON response from LLM
    """
    try:
        logger.warning(f"\n\n[5_LLM_RESPONSE] Running script...")
        logger.warning(f"[5_LLM_RESPONSE] Prompt given: {prompt}")
        logger.info(f"[5_LLM_RESPONSE] Processing {len(top_selected_chunks)} chunks")
        
        # Step 1: Filter chunks
        filtered_chunks = filter_chunks(top_selected_chunks)
        
        # Step 2: Choose final chunks
        chosen_chunks = choose_chunk(filtered_chunks)
        
        # Step 3: Generate answer using LLM
        answer = give_answer(prompt, chosen_chunks)
        
        # Log summary
        if "error" not in answer:
            logger.warning(f"[5_LLM_RESPONSE] Script completed successfully.")
            logger.info(f"[5_LLM_RESPONSE] Generated answer with {answer.get('metadata', {}).get('chunks_cited', 0)} citations")
            logger.info(f"[5_LLM_RESPONSE] Primary countries: {answer.get('metadata', {}).get('primary_countries', [])}")
        else:
            logger.error(f"[5_LLM_RESPONSE] Script completed with error: {answer.get('error', 'Unknown error')}")
        
        logger.warning("[5_LLM_RESPONSE] Script finished. Exiting.")
        return answer
        
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 5_llm_response.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        
        # Return error response even if pipeline breaks
        return ResponseProcessor._create_error_response(f"Pipeline error: {str(e)}", prompt or "Unknown question")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the LLM response script with specified chunks and prompt.')
    parser.add_argument('--chunks', type=str, required=True, help='JSON string of chunks to execute the script for.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to execute the script for.')
    args = parser.parse_args()
    
    try:
        chunks_list = json.loads(args.chunks)
        result = run_script(top_selected_chunks=chunks_list, prompt=args.prompt)
        
        # Print result for command line usage
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing chunks JSON: {e}")
        error_response = ResponseProcessor._create_error_response(f"Invalid chunks JSON: {str(e)}")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        error_response = ResponseProcessor._create_error_response(f"Execution error: {str(e)}")
        print(json.dumps(error_response, indent=2, ensure_ascii=False))

    # When argparse is added, can no longer press the 'run button' in VSCode/Cursor.
    # Instead, need to python xx.py --argument
    # This is to, potentially, allow an alternative way of bridging, 
    #   of the end-to-end communication between interface.py/Github Actions and these sub-entrypoints
    #   because xx.py --argument could work better than calling the function directly
    # E.g. use in terminal: python 5_llm_response.py --chunks '[{"id": 1, "content": "test"}]' --prompt "What is this about?"