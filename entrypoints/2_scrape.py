import sys
from pathlib import Path
import traceback
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple  
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.scrape import scrape_documents
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.database import Connection

logger = logging.getLogger(__name__)

@Logger.log(log_file=project_root / "logs/scrape.log", log_level="DEBUG")
@Test.sleep(3)
def run_script():
    """
    Run the script.
    """
    logger.warning("\n\n[1_SCRAPE] Running script...")
    
    detector = Detector()
    changes = detector.detect_changes()

    if changes:
        logger.critical("[1_SCRAPE] Changes detected. Instantiating the entire pipeline...")
        logger.info("[1_SCRAPE] Running spider...")

        try:
            # TODO: figure out how exactly Jon's spider works and wrap around it
            JonWrapper.run_spider()
            logger.warning("[1_SCRAPE] Spider ran successfully. 1_SCRAPE script exiting.")
        except Exception as e:
            traceback = traceback.format_exc()
            logger.error(f"[1_SCRAPE] Error running spider: {e}\n\nTraceback:\n{traceback}")
    else:
        logger.warning("[1_SCRAPE] No changes detected. Script exiting.")

    return changes

if __name__ == "__main__":
    run_script()