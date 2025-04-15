import sys
from pathlib import Path
import traceback
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from database import Base, Connection
from evaluator import VectorComparison, RegexComparison, SomeOtherComparison, Evaluator
from query import Booster
from helpers import Logger, Test
from constants.settings import FILE_PROCESSING_CONCURRENCY
logger = logging.getLogger(__name__)

def retrieve_chunks():
    engine = Connection.get_engine()
    connected = Connection.connect()
    pass

def evaluate_chunks(prompt, chunks):
    vector_comparison = VectorComparison
    regex_comparison = RegexComparison()
    some_other_comparison = SomeOtherComparison()
    evaluator = Evaluator()
    pass

def run_script():
    pass

if __name__ == "__main__":
    run_script()