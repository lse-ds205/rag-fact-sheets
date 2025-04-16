import sys
from pathlib import Path
import traceback
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from query import ChunkEngineering, QueryEngineering
from helpers import Logger, Test, TaskInfo

logger = logging.getLogger(__name__)

@Test.dummy_chunk()
def filter_chunks(top_selected_chunks: List[str]):
    try:
        logger.info("[4_OUTPUT] Filtering chunks...")
        engineered_chunks = ChunkEngineering()
        logger.info("[4_OUTPUT] Chunks filtered.")
        return engineered_chunks
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_OUTPUT] Error in filter_chunks: {e}\nTraceback: {traceback_string}")
        raise e

@Test.dummy_answer()
def choose_chunk(engineered_chunks: List[str]):
    try:
        logger.info("[4_OUTPUT] Choosing chunk...")
        chosen_chunk = ChunkEngineering()
        logger.info("[4_OUTPUT] Chunk chosen.")
        return chosen_chunk
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_OUTPUT] Error in choose_chunk: {e}\nTraceback: {traceback_string}")
        raise e

@Test.dummy_answer()
def give_answer(engineered_prompt: str, chosen_chunk: str):
    try:
        logger.info("[4_OUTPUT] Giving answer...")
        answer = QueryEngineering(), engineered_prompt, chosen_chunk
        dummy_answer = 'some dummy answer'
        logger.info(f"[4_OUTPUT] Answer given: {dummy_answer}")
        return answer
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[4_OUTPUT] Error in give_answer: {e}\nTraceback: {traceback_string}")
        raise e

@Logger.log(log_file=project_root / "logs/output.log", log_level="DEBUG")
@Test.sleep(3)
def run_script(top_selected_chunks: List[str], prompt: str):
    try:
        logger.warning("\n\n[4_OUTPUT] Running script...")
        logger.warning(f"[4_OUTPUT] Prompt given: {prompt}")
        logger.info(f"[4_OUTPUT] Chunks given:\n{'\n'.join(chunk for chunk in top_selected_chunks)}")
        filtered_chunks = filter_chunks(top_selected_chunks)
        chosen_chunk = choose_chunk(filtered_chunks)
        answer = give_answer(prompt, chosen_chunk)
        logger.warning("[4_OUTPUT] Script finished. Exiting.")
        return answer
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 4_output.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the query script with a specified given chunks and prompt.')
    parser.add_argument('--chunks', type=str, required=True, help='JSON string of chunks to execute the script for.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to execute the script for.')
    args = parser.parse_args()
    
    chunks_list = json.loads(args.chunks)
    run_script(top_selected_chunks=chunks_list, prompt=args.prompt)

    # When argparse is added, can no longer press the 'run button' in VSCode/Cursor.
    # Instead, need to python xx.py --argument
    # This is to, potentially, allow an alternative way of bridging, 
    #   of the end-to-end communication between interface.py/Github Actions and these sub-entrypoints
    #   because xx.py --argument could work better than calling the function directly
    # E.g. use in terminal: python 4_output.py --chunks '["chunk 1", "chunk 2", "chunk 3"]' --prompt "I am a prompt"