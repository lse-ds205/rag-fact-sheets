import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test, TaskInfo

logger = logging.getLogger(__name__)

@TaskInfo.ruikai()
class Detector:
    """
    Detector class. Overall goal is to detect if there's a change in the website.
    """
    def __init__(self):
        pass

    def detect_changes(self):
        """
        Detect changes in the website.
        """
        logger.info("[SCRAPE] Detecting changes in the website...")
        
        import random
        if random.random() < 0.1: # 10% chance of no changes
            logger.info("[SCRAPE] No changes detected.")
            return False
        else:
            logger.info("[SCRAPE] Changes detected.")
            return True


class JonWrapper:
    """
    Wrapper class that wraps around Jon's spider code.
    """
    @staticmethod
    def run_spider():
        """
        Run Jon's spider.
        """
        logger.info("[SCRAPE] Running spider...")
        logger.info("[SCRAPE] Spider ran successfully.")
        pass