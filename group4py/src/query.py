from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test, TaskInfo
from constants.prompts import (
    BOOSTER_PROMPT_1,
    BOOSTER_PROMPT_2,
    BOOSTER_PROMPT_3,
    CHUNK_PROMPT_1,
    CHUNK_PROMPT_2,
    CHUNK_PROMPT_3,
    PIPELINE_PROMPT_1,
    PIPELINE_PROMPT_2,
    PIPELINE_PROMPT_3
)

class Booster:
    """
    Booster class. General methodology:
    """
    def __init__(self):
        pass
    
    @Logger.debug_log()
    def boost_function(self, prompt: str) -> str:
        """
        Some function(s) to boost the prompt.
        """
        booster_prompt_1 = BOOSTER_PROMPT_1.format(CHUNK_PROMPT_1=prompt)
        pass

class ChunkEngineering:
    """
    Chunk-engineering class. Perhaps to engineer the top k, post-filtered chunks first?
    """
    def __init__(self):
        pass

class QueryEngineering:
    """
    Query-engineering class. After deciding on what's the best chunk, return the answer.
    """
    def __init__(self):
        pass