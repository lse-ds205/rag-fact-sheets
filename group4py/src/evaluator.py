from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from constants.prompts import BOOSTER_PROMPT_1, BOOSTER_PROMPT_2, BOOSTER_PROMPT_3
from helpers import Test

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

    def evaluate_total_score():
        """
        Some function(s) to evaluate the prompt.
        """
        score_1 = Evaluator._Evaluator.evaluate_function_1()
        score_2 = Evaluator._Evaluator.evaluate_function_2()
        total_score = score_1 + score_2
        pass

    def some_other_evaluation():
        pass

    class _Evaluator:

        @staticmethod
        def evaluate_function_1():
            """
            Some function(s) to evaluate the prompt.
            """
            pass

        @staticmethod
        def evaluate_function_2():
            """
            Some function(s) to evaluate the prompt.
            """
            pass