from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from constants.regex import (
    REGEX_WORD_PLACEHOLDER_1, REGEX_WORD_PLACEHOLDER_2, REGEX_WORD_PLACEHOLDER_3,
    REGEX_SENTENCE_PLACEHOLDER_1, REGEX_SENTENCE_PLACEHOLDER_2, REGEX_SENTENCE_PLACEHOLDER_3
)
from helpers import Logger, Test, TaskInfo

class VectorComparison:
    """
    Vector comparison class.
    """
    def __init__(self):
        pass
     
class RegexComparison:
    """
    Regex comparison class.
    """
    def __init__(self):
        pass
    
    @Logger.debug_log()
    def evaluate_regex_1(self, chunk: str):
        regex_1 = REGEX_WORD_PLACEHOLDER_1
        pass

    @Logger.debug_log()
    def evaluate_regex_2(self, chunk: str):
        regex_2 = REGEX_WORD_PLACEHOLDER_2
        pass

    @Logger.debug_log()
    def evaluate_regex_3(self, chunk: str):
        regex_3 = REGEX_WORD_PLACEHOLDER_3
        pass

class SomeOtherComparison:
    """
    Some other comparison class.
    """
    def __init__(self):
        pass

class Evaluator:
    """
    Evaluator class. General methodology:
    """
    def __init__(self):
        pass

    @Logger.debug_log()
    def evaluate_total_score():
        """
        Some function(s) to evaluate the prompt.
        """
        score_1 = Evaluator._Evaluator.evaluate_function_1()
        score_2 = Evaluator._Evaluator.evaluate_function_2()
        total_score = score_1 + score_2
        pass

    @Logger.debug_log()
    def some_other_evaluation():
        pass

    class _Evaluator:

        @Logger.debug_log()
        @staticmethod
        def evaluate_function_1():
            """
            Some function(s) to evaluate the prompt.
            """
            pass

        @Logger.debug_log()
        @staticmethod
        def evaluate_function_2():
            """
            Some function(s) to evaluate the prompt.
            """
            pass