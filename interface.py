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
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
    
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
def database():
    """
    Climate table operations for database management.
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
    
    # Interaction with /entrypoints (4) - LLM Response
    llm_module_name = 'entrypoints.4_llm_response'
    llm_run_script = importlib.import_module(llm_module_name).run_script
    llm_response = llm_run_script(top_selected_chunks=chunks, prompt=prompt)
    
    # Interaction with /entrypoints (5) - Output Processing
    output_module_name = 'entrypoints.5_output'
    output_run_script = importlib.import_module(output_module_name).run_script
    answer = output_run_script(llm_response=llm_response, prompt=prompt)

    print(f"[INTERFACE] Question: {prompt}\n[INTERFACE] Answer: {answer}")

    # In future, potential add entrypoints (5) for API, and (6) for App

# ------------------------------------------------------------

"""
Database operations:
"""

@database.command()
@click.option('--confirm', is_flag=True, help='Confirm deletion without additional prompt')
@click.option('--table', type=click.Choice(['all', 'doc_chunks', 'documents']), default='all', 
              help='Specify which table to drop rows from')
@click.option('--all', is_flag=True, help='Delete all rows from the table without restrictions')
def drop_rows(confirm, table, all):
    """
    Drop rows from the specified table (doc_chunks or documents) or all tables.
    
    This command helps maintain database health by removing data that might be outdated or no longer needed.
    Use --all to delete all rows unconditionally.
    """
    if not confirm:
        click.confirm(f'Are you sure you want to delete rows from the {table} table? This action cannot be undone.', abort=True)

    try:
        # Load environment variables
        load_dotenv()
        
        # Get database connection from environment or use default
        db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
        
        # Create engine
        engine = create_engine(db_url)
        
        # Connect to the database
        with engine.connect() as conn:
            # Start a transaction
            with conn.begin():
                # Handle the 'all' table option (previously 'database')
                if table == 'all':
                    # Get counts for both tables before deletion
                    doc_chunks_before = conn.execute(text("SELECT COUNT(*) FROM doc_chunks")).scalar()
                    documents_before = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
                    
                    # Delete from both tables
                    if all:
                        conn.execute(text("DELETE FROM doc_chunks"))
                        conn.execute(text("DELETE FROM documents"))
                    else:
                        # Use filtered deletion
                        conn.execute(text("""
                            DELETE FROM doc_chunks
                            WHERE doc_id IN (
                                SELECT doc_id 
                                FROM documents
                            )
                        """))
                        conn.execute(text("""
                            DELETE FROM documents
                            WHERE doc_id IN (
                                SELECT DISTINCT doc_id 
                                FROM doc_chunks
                            )
                        """))
                    
                    # Get counts after deletion
                    doc_chunks_after = conn.execute(text("SELECT COUNT(*) FROM doc_chunks")).scalar()
                    documents_after = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
                    
                    # Report results
                    print(f"[INTERFACE] Deleted {doc_chunks_before - doc_chunks_after} rows from doc_chunks")
                    print(f"[INTERFACE] Deleted {documents_before - documents_after} rows from documents")
                    print(f"[INTERFACE] Total rows deleted: {(doc_chunks_before - doc_chunks_after) + (documents_before - documents_after)}")
                
                else:
                    # Get the count before deletion
                    before_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    
                    # Construct the appropriate SQL based on the table and options
                    if all:
                        # Delete all rows without restriction
                        sql = f"DELETE FROM {table}"
                    else:
                        # Use filtered deletion
                        if table == 'doc_chunks':
                            sql = """
                            DELETE FROM doc_chunks
                            WHERE doc_id IN (
                                SELECT doc_id 
                                FROM documents
                            )
                            """
                        elif table == 'documents':
                            sql = """
                            DELETE FROM documents
                            WHERE doc_id IN (
                                SELECT DISTINCT doc_id 
                                FROM doc_chunks
                            )
                            """
                    
                    # Execute the deletion
                    result = conn.execute(text(sql))
                    
                    # Get the count after deletion
                    after_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    
                    # Calculate how many rows were deleted
                    deleted_count = before_count - after_count
                    
                    print(f"[INTERFACE] Successfully deleted {deleted_count} rows from the {table} table.")
                    print(f"[INTERFACE] Before: {before_count} rows, After: {after_count} rows")
        
    except Exception as e:
        print(f"[INTERFACE] Database error: {e}")

# ------------------------------------------------------------

if __name__ == "__main__":
    four()