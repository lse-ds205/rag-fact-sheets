"""
Helper functions for the project - used for miscellaneous, non-core, furnishing/production type of tasks
"""

import logging
import colorlog
from pathlib import Path
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import logging


class Logger:
    """
    Logger class
    """
    @staticmethod
    def setup_logging(log_file: Path, log_level: str = "INFO") -> None:
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'white',                 
                'INFO': 'green',                 
                'WARNING': 'yellow',          
                'ERROR': 'red',                 
                'CRITICAL': 'bold_red',   
            }
        )

        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(color_formatter)
        
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.handlers = []
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Set higher log levels for noisy third-party libraries
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('unstructured').setLevel(logging.INFO)
        logging.getLogger('pikepdf').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.INFO)

    @staticmethod
    def log(log_file: Path, log_level: str = "INFO"):
        """
        Decorator, typically only wrapped around main entrypoint functions (not the sub-functions), with two objectives: 
        (A) set up color logging
        (B) direct all logs to a specific file
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not logging.getLogger().hasHandlers():
                    Logger.setup_logging(log_file, log_level)
                result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def debug_log():
        """
        Decorator to log a debug message when a function starts running.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                logging.debug(f"<library> {func.__name__} started running...")
                return func(*args, **kwargs)
            return wrapper
        return decorator


class TaskInfo:
    """
    Harmless decorators. To ease communication between groupmates - can use it if you want to / find that it makes collaborative development easier.
    """
    def completed():
        """
        Decorator to indicate that a task has been completed.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def bryan():
        """
        Decorator to indicate that this task is being worked on by Bryan.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def zicheng():
        """
        Decorator to indicate that this task is being worked on by Zicheng.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def michele():
        """
        Decorator to indicate that this task is being worked on by Michele.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def ruikai():
        """
        Decorator to indicate that this task is being worked on by Rui Kai.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


class Test:
    """
    Test class - decorators to be used for testing/production phase. Removed for actual deployment.
    """
    logger = logging.getLogger(__name__)

    @staticmethod
    def sleep(duration: float):
        """
        Decorator to introduce a sleep time between function calls - simulate actual running of scripts for ease of reading logs.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                time.sleep(duration)
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def dummy(dummy: Any) -> Any:
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy
            return wrapper
        return decorator
    
    @staticmethod
    def force_input(*forced_args, **forced_kwargs):
        """
        Decorator to force specific input arguments and keyword arguments to a function.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                args = forced_args if forced_args else args
                kwargs.update(forced_kwargs)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def dummy_chunk() -> List[str]:
        dummy_chunks = ["I am dummy chunk 1", "I am dummy chunk 2", "I am dummy chunk 3"]
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_chunks
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_embedding() -> List[float]:
        dummy_embedding = [random.uniform(-1, 1) for _ in range(128)]    # Gives 128-dimensional embedding, randomly generated
        def decorator(func):
            def wrapper(*args, **kwargs):
                Test.logger.warning("Dummy embedding decorator used - not actual embedding!")
                result = func(*args, **kwargs)
                return dummy_embedding
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_json() -> Dict[str, Any]:
        dummy_dict = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
            "key4": {"nested_key": "nested_value"}
        }
        dummy_json = [dummy_dict, dummy_dict, dummy_dict]
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_json
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_prompt() -> str:
        dummy_prompt = "I am a dummy prompt"
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_prompt
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_answer() -> str:
        dummy_answer = "I am a dummy answer"
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_answer
            return wrapper
        return decorator