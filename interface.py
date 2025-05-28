import click
import importlib
from pathlib import Path
import sys
import asyncio
import json
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
import group4py
from helpers.internal import Logger, Test
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
    Run detection and processing with the new two-step approach.
    """
    # Interaction with /entrypoints (2) - scraping
    module_name_scrape = 'entrypoints.2_scrape'
    run_script_scrape = importlib.import_module(module_name_scrape).run_script
    changes = run_script_scrape()

    # Interaction with /entrypoints (3) - chunking. Note: only triggered if changes are detected
    if changes:
        # Step 1: Chunking
        print("[INTERFACE] Changes detected, starting chunking process...")
        module_name_chunk = 'entrypoints.3_chunk'
        run_script_chunk = importlib.import_module(module_name_chunk).run_script
        asyncio.run(run_script_chunk())
        print("[INTERFACE] Chunking completed successfully.")
        
        # Step 2: Embedding
        print("[INTERFACE] Starting embedding process...")
        module_name_embed = 'entrypoints.3.5_embed'
        run_script_embed = importlib.import_module(module_name_embed).run_script
        asyncio.run(run_script_embed())
        print("[INTERFACE] Embedding completed successfully.")
        
        print("[INTERFACE] All processing completed successfully.")
    else:
        print("[INTERFACE] No changes detected, skipping processing.")

@detect.command()
@click.option('--force', is_flag=True, help='Force reprocessing of all documents.')
def chunk(force):
    """
    Run only the chunking step.
    """
    print("[INTERFACE] Running chunking process...")
    module_name_chunk = 'entrypoints.3_chunk'
    run_script_chunk = importlib.import_module(module_name_chunk).run_script
    asyncio.run(run_script_chunk(force_reprocess=force))
    print("[INTERFACE] Chunking completed successfully.")


@detect.command() 
@click.option('--force', is_flag=True, help='Force regeneration of embeddings.')
def embed(force):
    """
    Run only the embedding step.
    """
    print("[INTERFACE] Running embedding process...")
    module_name_embed = 'entrypoints.3.5_embed'
    run_script_embed = importlib.import_module(module_name_embed).run_script
    asyncio.run(run_script_embed(force_reembed=force))
    print("[INTERFACE] Embedding completed successfully.")

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
    # Interaction with /entrypoints (4)
    retrieve_module_name = 'entrypoints.4_retrieve'
    retrieve_run_script = importlib.import_module(retrieve_module_name).run_script
    chunks = retrieve_run_script(prompt=prompt)
    
    # Interaction with /entrypoints (5) - LLM Response
    llm_module_name = 'entrypoints.5_llm_response'
    llm_run_script = importlib.import_module(llm_module_name).run_script
    llm_response = llm_run_script(top_selected_chunks=chunks, prompt=prompt)
    
    # Interaction with /entrypoints (6) - Output Processing
    output_module_name = 'entrypoints.6_output'
    output_run_script = importlib.import_module(output_module_name).run_script
    answer = output_run_script(llm_response=llm_response, prompt=prompt)

    print(f"[INTERFACE] Question: {prompt}\n[INTERFACE] Answer: {answer}")

    # In future, potential add entrypoints for API and App

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

@database.command()
@click.option('--confirm', is_flag=True, help='Confirm table deletion without additional prompt')
def drop_tables(confirm):
    """
    Drop ALL tables from the database completely.
    
    This command removes the entire table structure, not just the data.
    Use this when you want to completely reset the database schema.
    WARNING: This is irreversible and will delete all data!
    """
    if not confirm:
        click.confirm('Are you sure you want to DROP ALL TABLES? This will permanently delete all table structures and data. This action cannot be undone.', abort=True)

    try:
        # Load environment variables
        load_dotenv()
        
        # Get database connection from environment or use default
        db_url = os.getenv('DATABASE_URL', 'postgresql://climate:climate@localhost:5432/climate')
        
        # Create engine
        engine = create_engine(db_url)
        
        # Connect to the database
        with engine.connect() as conn:
            # Get all table names first
            result = conn.execute(text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename
            """))
            tables = [row[0] for row in result.fetchall()]
            
            if not tables:
                print("[INTERFACE] No tables found to drop.")
                return
            
            print(f"[INTERFACE] Found {len(tables)} tables to drop: {', '.join(tables)}")
            
            # Drop all tables with CASCADE to handle foreign key constraints
            dropped_count = 0
            for table in tables:
                try:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                    conn.commit()
                    print(f"[INTERFACE] Dropped table: {table}")
                    dropped_count += 1
                except Exception as table_error:
                    print(f"[INTERFACE] Warning: Could not drop table {table}: {table_error}")
            
            # Clean up sequences (auto-increment counters)
            try:
                result = conn.execute(text("""
                    SELECT sequencename 
                    FROM pg_sequences 
                    WHERE schemaname = 'public'
                """))
                sequences = [row[0] for row in result.fetchall()]
                
                for sequence in sequences:
                    try:
                        conn.execute(text(f"DROP SEQUENCE IF EXISTS {sequence} CASCADE"))
                        conn.commit()
                        print(f"[INTERFACE] Dropped sequence: {sequence}")
                    except Exception as seq_error:
                        print(f"[INTERFACE] Warning: Could not drop sequence {sequence}: {seq_error}")
            except Exception:
                pass  # Sequences table might not exist in older PostgreSQL versions
            
            # Clean up views
            try:
                result = conn.execute(text("""
                    SELECT viewname 
                    FROM pg_views 
                    WHERE schemaname = 'public'
                """))
                views = [row[0] for row in result.fetchall()]
                
                for view in views:
                    try:
                        conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
                        conn.commit()
                        print(f"[INTERFACE] Dropped view: {view}")
                    except Exception as view_error:
                        print(f"[INTERFACE] Warning: Could not drop view {view}: {view_error}")
            except Exception:
                pass  # Views might not exist
            
            print(f"[INTERFACE] Successfully dropped {dropped_count} tables and cleaned up database objects.")
            print("[INTERFACE] Database schema has been completely reset.")
        
    except Exception as e:
        print(f"[INTERFACE] Database error: {e}")

@database.command()
def show_relationships():
    """
    Display relationship information from the logical_relationships table.
    
    Shows the total count of relationships and a sample of the first 5 records.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        from group4py.src.database import Connection
        from sqlalchemy import text

        conn = Connection()
        conn.connect()

        with conn.get_engine().connect() as db:
            # Get total count of relationships
            result = db.execute(text('SELECT COUNT(*) FROM logical_relationships')).fetchone()
            print(f'[INTERFACE] Relationships in database: {result[0]}')
            
            # Sample of relationships
            sample = db.execute(text('SELECT source_chunk_id, target_chunk_id, relationship_type, confidence FROM logical_relationships LIMIT 5')).fetchall()
            print("\n[INTERFACE] Sample relationships:")
            for row in sample:
                print(f"Source: {row[0]} -> Target: {row[1]} | Type: {row[2]} | Confidence: {row[3]:.2f}")
                
    except Exception as e:
        print(f"[INTERFACE] Database error: {e}")

@database.command()
@click.option('--type', 'rel_type', type=str, help='Filter by relationship type (e.g., SUPPORTS, CONTRADICTS)')
@click.option('--min-confidence', type=float, default=0.0, help='Minimum confidence score (0.0 to 1.0)')
@click.option('--limit', type=int, default=10, help='Maximum number of relationships to show')
def analyze_relationships(rel_type, min_confidence, limit):
    """
    Analyze relationships in the logical_relationships table with filtering options.
    
    Filter by relationship type and/or minimum confidence score.
    """
    try:
        # Load environment variables
        load_dotenv()
        
        from group4py.src.database import Connection
        from sqlalchemy import text

        conn = Connection()
        conn.connect()

        with conn.get_engine().connect() as db:
            # Base query
            query = """
                SELECT source_chunk_id, target_chunk_id, relationship_type, confidence, evidence, method, created_at
                FROM logical_relationships
                WHERE 1=1
            """
            params = {}
            
            # Add filters if provided
            if rel_type:
                query += " AND relationship_type = :rel_type"
                params['rel_type'] = rel_type
                
            if min_confidence > 0:
                query += " AND confidence >= :min_confidence"
                params['min_confidence'] = min_confidence
                
            # Add order by and limit
            query += " ORDER BY confidence DESC LIMIT :limit"
            params['limit'] = limit
            
            # Execute query
            relationships = db.execute(text(query), params).fetchall()
            
            # Get statistics by relationship type
            type_stats = db.execute(text("""
                SELECT relationship_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM logical_relationships
                GROUP BY relationship_type
                ORDER BY count DESC
            """)).fetchall()
            
            # Display results
            print(f"\n[INTERFACE] Relationship Statistics by Type:")
            print("-" * 60)
            print(f"{'Type':<15} | {'Count':>8} | {'Avg Confidence':>15}")
            print("-" * 60)
            for row in type_stats:
                print(f"{row[0]:<15} | {row[1]:>8} | {row[2]:>15.4f}")
            
            # Display filtered relationships
            print(f"\n[INTERFACE] Showing {len(relationships)} relationships:")
            print("-" * 80)
            for row in relationships:
                print(f"Source: {row[0]} → Target: {row[1]} | Type: {row[2]} | Confidence: {row[3]:.2f}")
                print(f"Evidence: {row[4][:100]}{'...' if len(row[4]) > 100 else ''}")
                print(f"Method: {row[5]} | Created: {row[6]}")
                print("-" * 80)
                
    except Exception as e:
        print(f"[INTERFACE] Database error: {e}")

# ------------------------------------------------------------

if __name__ == "__main__":
    four()