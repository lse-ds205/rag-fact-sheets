import sys
from pathlib import Path
import traceback
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from scrape import Detector, JonWrapper
from helpers import Logger, Test

logger = logging.getLogger(__name__)

@Logger.log(log_file=project_root / "logs/scrape.log")
@Test.sleep(3)
def run_script():
    """
    Run the script.
    """
    logger.warning("\n\n[1_SCRAPE] Running script...")
    
    detector = Detector()
    changes = detector.detect_changes()

    if changes:
        logger.warning("[1_SCRAPE] Changes detected. Instantiating the entire pipeline...")
        logger.info("[1_SCRAPE] Running spider...")

        try:
            JonWrapper.run_spider()
            logger.warning("[1_SCRAPE] Spider ran successfully. 1_SCRAPE script exiting.")
        except Exception as e:
            traceback = traceback.format_exc()
            logger.error(f"[1_SCRAPE] Error running spider: {e}\n\nTraceback:\n{traceback}")
    else:
        logger.warning("[1_SCRAPE] No changes detected. Script exiting.")

if __name__ == "__main__":
    run_script()