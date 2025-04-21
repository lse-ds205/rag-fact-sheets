import sys
from pathlib import Path
import traceback
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from database import Connection
from docchunk import Embedding
from evaluator import VectorComparison, RegexComparison, SomeOtherComparison, Evaluator
from query import Booster
from helpers import Logger, Test

logger = logging.getLogger(__name__)

@Test.dummy_embedding()
def embed_prompt(prompt):
    """
    Embed a prompt using the embedding model.
    """
    logger.info(f"[3_RETRIEVE] Prompt given: {prompt}. Embedding prompt...")
    try:
        boosted_prompt = Booster().boost_function(prompt)
        embedded_prompt = Embedding().embed_one(boosted_prompt)
        return embedded_prompt
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[3_RETRIEVE] Error in prompt embedding: {e}\nTraceback: {traceback_string}")
        raise e

@Test.dummy_json()
def retrieve_chunks(embedded_prompt):
    try:
        engine = 'some engine'
        Connection.get_engine(engine)
        Connection.connect(engine)
        retrieved = "Some pgvector function that retrieves chunks"
        return retrieved
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[3_RETRIEVE] Error in chunk retrieval: {e}\nTraceback: {traceback_string}")
        raise e

@Test.dummy_chunk()
def evaluate_chunks(prompt, chunks):
    try:
        vector_comparison = VectorComparison()
        regex_comparison = RegexComparison()
        some_other_comparison = SomeOtherComparison()
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[3_RETRIEVE] Error in chunk evaluation: {e}\nTraceback: {traceback_string}")
        raise e
    
    try:
        evaluated_chunks = Evaluator(), vector_comparison, regex_comparison, some_other_comparison
        return evaluated_chunks
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[3_RETRIEVE] Error in chunk evaluation: {e}\nTraceback: {traceback_string}")
        raise e

@Logger.log(log_file=project_root / "logs/retrieve.log", log_level="DEBUG")
@Test.sleep(3)
@Test.dummy_chunk()
def run_script(prompt: str):
    try:
        logger.warning(f"\n\n[3_RETRIEVE] Running script...")
        logger.info(f"[3_RETRIEVE] Prompt given: {prompt}")
        embedded_prompt = embed_prompt(prompt)
        logger.info(f"[3_RETRIEVE] Prompt embedded. Retrieving the top X chunks...")
        chunks = retrieve_chunks(embedded_prompt)
        logger.info(f"[3_RETRIEVE] Chunks retrieved. Evaluating chunks...")
        evaluated_chunks = evaluate_chunks(prompt, chunks)
        logger.warning(f"[3_RETRIEVE] Chunks evaluated. Narrowed down to top Y chunks. Script exiting.")
        return evaluated_chunks

    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 3_retrieve.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the query script with a specified prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to execute the script for.')
    args = parser.parse_args()
    run_script(prompt=args.prompt)

    # When argparse is added, can no longer press the 'run button' in VSCode/Cursor.
    # Instead, need to python xx.py --argument
    # This is to, potentially, allow an alternative way of bridging, 
    #   of the end-to-end communication between interface.py/Github Actions and these sub-entrypoints
    #   because xx.py --argument could work better than calling the function directly
    # E.g. use in terminal: python 3_retrieve.py --prompt "I am a prompt"