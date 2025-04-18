import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test, TaskInfo

class DocChunk:
    """
    Chunk class. General methodology:
        Input: Some form of documents (PDFs, etc.)
        Output: A list of chunks
    """
    def __init__(self):
        pass

    @Test.dummy_json()
    @Logger.debug_log()
    def chunking_function(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Some function(s), probably using unstructured, to split the document (in the form of a file path) into chunks.
        Output probably in the form of a list of dictionaries, with the following keys:
            - text: The text of the chunk
            - metadata: A dictionary of metadata about the chunk
        """
        pass

    @Test.dummy_json()
    @Logger.debug_log()
    def cleaning_function(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Some function(s) to clean/process the chunks.
        E.g. remove gibberish text, re-format metadata, etc.
        Optional: (Probably need to use) the internal chunk class to clean the chunks - seems there can potentially be a lot of steps
        """
        chicken = None
        egg = self._Chunk._clean_strategy_one(chicken)
        omelette = self._Chunk._clean_strategy_two(egg)
        meal = self._Chunk._clean_strategy_three(omelette)
        # return meal
        pass

    class _DocChunk:
        """
        Internal chunk class.
        """
        @staticmethod
        def _clean_strategy_one(text):
            """[Do something]"""
            pass

        @staticmethod
        def _clean_strategy_two(text):
            pass
        
        @staticmethod
        def _clean_strategy_three(text):
            pass

class Embedding:
    """
    Embedding class. General methodology:
        Input: DocChunks, in the form of strings
        Output: Embeddings, in the form of a list of floats
    """
    def __init__(self):
        pass

    @Test.dummy_json()
    @Logger.debug_log()
    def embed_many(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Some function(s) to embed the chunks.
        """
        all_embeddings = self.embed_one(None)
        pass

    @Test.dummy_embedding()
    @Logger.debug_log()
    def embed_one(self, chunk: Dict[str, Any]) -> List[float]:
        """
        Some function(s) to embed a single chunk.
        """
        pass