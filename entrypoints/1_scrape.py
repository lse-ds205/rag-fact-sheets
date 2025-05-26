"""
NDC Document Scraping Entrypoint

Lightweight orchestrator for the NDC document scraping workflow.
All business logic is delegated to the scrape module.

This script now includes document download functionality, which
automatically downloads any new documents found during scraping.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import group4py
from group4py.src.scrape import run_scraping_workflow
from helpers import Logger


@Logger.log(log_file=project_root / "logs/scrape.log", log_level="INFO")
def main():
    """Execute the NDC document scraping workflow."""
    try:
        result = run_scraping_workflow()
        return result
    except Exception as e:
        print(f"Scraping workflow failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()