import click
import importlib
from pathlib import Path
import sys
import asyncio
import json
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
def detect():
    """
    Detect changes and process if needed. Mirrors to /entrypoints (1) and (2)
    """
    pass

@four.group()
def query():
    """
    Query operations. Mirrors to /entrypoints (3) and (4)
    """
    pass

@four.group()
def api():
    """
    API operations. Dummy group - potential to be implemented in future.
    """
    pass

@four.group()
def app():
    """
    App operations for front-end. Dummy group - potential to be implemented in future.
    Serves as the only point of communication between front-end and back-end. 
    No other method of interaction is allowed. The Dev Wall of China.
    """
    pass

# ------------------------------------------------------------

"""
Detect operations:
"""

@detect.command()
@click.option('--option1', is_flag=True, help='Placeholder for option 1.')
@click.option('--option2', is_flag=True, help='Placeholder for option 2.')
def run(option1, option2):
    """
    Run detection and processing.
    """
    # Interaction with /entrypoints (1)
    module_name_scrape = 'entrypoints.1_scrape'
    run_script_scrape = importlib.import_module(module_name_scrape).run_script
    changes = run_script_scrape()

    # Interaction with /entrypoints (2). Note: only triggered if changes are detected
    if changes:
        module_name_process = 'entrypoints.2_process'
        run_script_process = importlib.import_module(module_name_process).run_script
        asyncio.run(run_script_process())
        print("[INTERFACE] Changes detected, processed successfully.")
    else:
        print("[INTERFACE] No changes detected, skipping processing.")

# ------------------------------------------------------------

"""
Query operations:
"""

@query.command()
@click.option('--prompt', type=str, required=True, help='Prompt to execute the script for.')
def ask(prompt):
    """
    Run the query script with specified chunks and prompt.
    """
    # Interaction with /entrypoints (3)
    retrieve_module_name = 'entrypoints.3_retrieve'
    retrieve_run_script = importlib.import_module(retrieve_module_name).run_script
    chunks = retrieve_run_script(prompt=prompt)
    
    # Interaction with /entrypoints (4)
    output_module_name = 'entrypoints.4_output'
    output_run_script = importlib.import_module(output_module_name).run_script
    answer = output_run_script(top_selected_chunks=chunks, prompt=prompt)

    print(f"[INTERFACE] Question: {prompt}\n[INTERFACE] Answer: {answer}")

    # In future, potential add entrypoints (5) for API, and (6) for App

# ------------------------------------------------------------

if __name__ == "__main__":
    four()