import click
import importlib
from pathlib import Path
import sys
import asyncio
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test

"""
Top-down settings:
"""

@click.group()
def four():
    """
    Main entry-point command for our stack (Group 4).
    Naming convention assumes intra-group operations as "number"()
    Whilst inter-group interations (possibly commanded by a higher-level stack), as such:
        one() [what group 1 is doing]
        two()
        ...
        etc.
    with "number"() being the main entry-point command for each group.
    Implemented with future-proofing in mind, for scalability post DS205.
    """
    pass

@four.group()
def scrape():
    """
    Scraping operations. Mirrors to /entrypoints (1)
    """
    pass

@four.group()
def process():
    """
    Processing operations. Mirrors to /entrypoints (2)
    """
    pass

@four.group()
def retrieve():
    """
    Retrieval operations. Mirrors to /entrypoints (3)
    """
    pass

# ------------------------------------------------------------

"""
Scraping operations:
"""

@scrape.command()
@click.option('--option1', is_flag=True, help='Enable option 1 for crawling.')
@click.option('--option2', is_flag=True, help='Enable option 2 for crawling.')
async def crawl(option1, option2):
    """
    Crawl the web.

    This command will initiate a web crawling operation using the specified options.
    """
    module_name = 'entrypoints.1_scrape'
    run_script = importlib.import_module(module_name).run_script
    await run_script()

# ------------------------------------------------------------

"""
Processing operations:
"""

@process.command()
def process_pdfs():
    """
    Process the data.
    """
    module_name = 'entrypoints.2_process'
    run_script = importlib.import_module(module_name).run_script
    asyncio.run(run_script())

# ------------------------------------------------------------

if __name__ == "__main__":
    four()