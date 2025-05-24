"""
Script for setting up the database tables according to the schema.
Run this script to initialize the PostgreSQL database structure without adding any data.
"""
import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import asyncpg

# Load environment variables from .env file
load_dotenv()

# Set up paths to ensure imports work correctly
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import from the project
from group4py.src.database import Connection, Base
from group4py.src.schema import DatabaseConfig, DocChunk, LogicalRelationship

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_database")

async def setup_hoprag_schema(db_url: str):
    """Setup HopRAG-specific schema with optimized indexes"""
    
    # First, create extensions and add columns to existing doc_chunks table
    schema_sql = """
    -- Enable extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    
    -- Add HopRAG-specific columns to existing doc_chunks table if they don't exist
    DO $$ 
    BEGIN
        -- Add embedding column if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name = 'doc_chunks' AND column_name = 'embedding') THEN
            ALTER TABLE doc_chunks ADD COLUMN embedding VECTOR(768);
        END IF;
        
        -- Add content_hash column if it doesn't exist
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name = 'doc_chunks' AND column_name = 'content_hash') THEN
            ALTER TABLE doc_chunks ADD COLUMN content_hash VARCHAR(64);
        END IF;        -- Add chunk_data as JSONB if it doesn't exist (renamed to avoid SQLAlchemy reserved word conflict)
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name = 'doc_chunks' AND column_name = 'chunk_data') THEN
            IF EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name = 'doc_chunks' AND column_name = 'metadata') THEN
                ALTER TABLE doc_chunks RENAME COLUMN metadata TO chunk_data;
            ELSIF EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name = 'doc_chunks' AND column_name = 'chunk_metadata') THEN
                ALTER TABLE doc_chunks RENAME COLUMN chunk_metadata TO chunk_data;
                ALTER TABLE doc_chunks ALTER COLUMN chunk_data TYPE JSONB USING chunk_data::JSONB;
            ELSE
                ALTER TABLE doc_chunks ADD COLUMN chunk_data JSONB DEFAULT '{}';
            END IF;
        END IF;
    END $$;
    
    -- Logical relationships for graph traversal
    CREATE TABLE IF NOT EXISTS logical_relationships (
        id SERIAL PRIMARY KEY,
        source_chunk_id INTEGER NOT NULL REFERENCES doc_chunks(id) ON DELETE CASCADE,
        target_chunk_id INTEGER NOT NULL REFERENCES doc_chunks(id) ON DELETE CASCADE,
        relationship_type VARCHAR(50) NOT NULL CHECK (
            relationship_type IN ('SUPPORTS', 'EXPLAINS', 'CONTRADICTS', 'FOLLOWS', 'TEMPORAL_FOLLOWS', 'CAUSES')
        ),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        evidence TEXT,
        method VARCHAR(50) DEFAULT 'rule_based',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        CHECK (source_chunk_id != target_chunk_id)
    );
    """
    
    # Index creation commands (must be executed outside transactions)
    index_commands = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_content_gin ON doc_chunks USING gin(to_tsvector('english', content));",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_embedding_hnsw ON doc_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_source_conf ON logical_relationships(source_chunk_id, confidence) WHERE confidence > 0.6;",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rel_target_type ON logical_relationships(target_chunk_id, relationship_type);"
    ]
    
    conn = await asyncpg.connect(db_url)
    try:
        # Create tables first
        await conn.execute(schema_sql)
        logger.info("HopRAG tables created successfully")
        
        # Create indexes one by one (outside of transactions)
        for idx, command in enumerate(index_commands, 1):
            try:
                await conn.execute(command)
                logger.info(f"Created index {idx}/{len(index_commands)}")
            except Exception as e:
                # If index already exists, that's fine
                if "already exists" in str(e).lower():
                    logger.info(f"Index {idx}/{len(index_commands)} already exists, skipping")
                else:
                    logger.warning(f"Failed to create index {idx}: {e}")
        
        logger.info("HopRAG schema setup completed successfully")
    finally:
        await conn.close()

def setup_database(db_url=None, echo=True, hoprag=False):
    """
    Function to set up the database tables in PostgreSQL
    
    Args:
        db_url: PostgreSQL connection URL (optional, will use DATABASE_URL from .env if not provided)
        echo: Whether to echo SQL commands
        hoprag: Whether to set up HopRAG-specific schema and indexes
        
    Returns:
        bool: Success status
    """
    logger.info("Setting up database tables")
    
    # If no db_url provided, get from environment
    if not db_url:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("No DATABASE_URL found in environment variables or .env file")
            return False
    
    # Make sure we're using a PostgreSQL URL
    if not db_url.startswith('postgresql'):
        logger.error("Only PostgreSQL is supported. Please provide a PostgreSQL connection URL.")
        return False
    
    try:
        # Create database configuration
        db_config = DatabaseConfig(
            url=db_url,
            create_tables=True,  # Automatically create tables if they don't exist
            echo=echo  # Print SQL statements (useful for debugging)
        )
        
        # Initialize database connection
        db = Connection(config=db_config)
        
        # Connect to database and create tables
        if db.connect():
            logger.info("Successfully connected to database and created tables")
            
            if hoprag:
                logger.info("Setting up HopRAG schema")
                asyncio.run(setup_hoprag_schema(db_url))
            
            return True
        else:
            logger.error("Failed to connect to database")
            return False
    except ValueError as e:
        logger.error(f"Invalid database URL: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up PostgreSQL database tables for RAG Fact Sheets")
    parser.add_argument(
        "--db-url", 
        type=str,
        default=None,
        help="PostgreSQL connection URL in format: postgresql://username:password@hostname:port/database"
    )
    parser.add_argument(
        "--echo", 
        action="store_true",
        default=True,
        help="Echo SQL statements (useful for debugging)"
    )
    parser.add_argument(
        "--hoprag", 
        action="store_true",
        help="Setup HopRAG-specific schema and indexes"
    )
    
    args = parser.parse_args()
    
    # Use provided URL or get from environment
    db_url = args.db_url
        
    if setup_database(db_url, args.echo, args.hoprag):
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")
        sys.exit(1)