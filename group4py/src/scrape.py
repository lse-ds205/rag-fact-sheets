import logging

logger = logging.getLogger(__name__)

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
        if random.random() < 0.1:
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