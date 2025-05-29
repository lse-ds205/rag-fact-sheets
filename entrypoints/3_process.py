import sys
from pathlib import Path
import traceback
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from docchunk import DocChunk, Embedding
from helpers import Logger, Test, TaskInfo
from schema import ChunkModel, EmbeddingModel
from database import Connection
from constants.settings import FILE_PROCESSING_CONCURRENCY
logger = logging.getLogger(__name__)

@Test.dummy(["./file path 1", "./file path 2", "./file path 3"])
def get_file_paths():
    """
    Get the file paths. Or any other feasible method.
    """
    pass

@TaskInfo.bryan()
@TaskInfo.completed()
async def process_file_one(file_path: str):
    """
    Process a file and return a list of chunks and embeddings.
    """
    try:
        chunks = DocChunk().chunking_function(file_path)
        logger.info(f"[2_PROCESS] Finished chunking {file_path}, cleaning chunks...")
        for chunk in chunks:
            validated_chunk = ChunkModel(placeholder=chunk)
            logger.debug(f"[2_PROCESS] Validated chunks successfully, cleaning chunks...")

        cleaned = DocChunk().cleaning_function(chunks)
        logger.info(f"[2_PROCESS] Finished cleaning chunks, embedding chunks...")

        embeddings = Embedding().embed_many(cleaned)
        for embedding in embeddings:
            validated_embedding = EmbeddingModel(placeholder_model=validated_chunk, placeholder_embedding=embedding)
            logger.debug(f"[2_PROCESS] Validated embeddings successfully...")

        logger.info(f"[2_PROCESS] File {file_path} processed successfully")
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[2_PROCESS] Error processing file {file_path}: {e}\n\nTraceback:\n{traceback_string}")

async def process_file_many(file_path):
    semaphore = asyncio.Semaphore(FILE_PROCESSING_CONCURRENCY)
    async with semaphore:
        await process_file_one(file_path)

def upload_chunks(embedded_chunks: List[ChunkModel]):
    """
    Upload chunks to the database.
    """
    logger.info(f"[2_PROCESS] Uploading chunks into database...")
    Connection().upload(embedded_chunks)
    logger.warning(f"[2_PROCESS] Uploaded chunks into database successfully")
    pass

@Logger.log(log_file=project_root / "logs/process.log", log_level="DEBUG")
@Test.sleep(3)
async def run_script():
    try:
        logger.warning(f"\n\n[2_PROCESS] Running script...")
        file_paths = get_file_paths()
        tasks = [process_file_many(file_path) for file_path in file_paths]
        embedded_chunks = await asyncio.gather(*tasks)
        logger.warning(f"[2_PROCESS] All files processed successfully. Uploading them into database...")
        upload_chunks(embedded_chunks)
        logger.warning(f"[2_PROCESS] Script completed successfully. Exiting.")
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 2_process.py: {e}\n\nTraceback: {traceback_string}\n\n\n\n")
        raise e

if __name__ == "__main__":
    asyncio.run(run_script())