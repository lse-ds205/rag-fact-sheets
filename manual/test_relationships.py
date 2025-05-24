#!/usr/bin/env python3
"""
Test script for fixing logical relationship population
"""

import sys
import asyncio
from pathlib import Path
import logging
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables from .env file
load_dotenv()

# Set up paths to ensure imports work correctly
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import from the project
from group4py.src.schema import DatabaseConfig
from group4py.src.hop_rag import HopRAGGraphProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_relationships")

async def test_relationships():
    """Test building logical relationships"""
    try:
        logger.info("Starting relationship test")
        
        # Initialize HopRAG processor
        config = DatabaseConfig.from_env()
        processor = HopRAGGraphProcessor(config)
        await processor.initialize()
        
        # Check existing relationships
        with processor.db_connection.get_engine().connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
            count_before = result.scalar() or 0
            logger.info(f"Current relationships in database: {count_before}")
            
            # Check schema
            tables = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )).fetchall()
            logger.info(f"Database tables: {[t[0] for t in tables]}")
            
            if 'logical_relationships' in [t[0] for t in tables]:
                # Check column types
                columns = conn.execute(text(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'logical_relationships'"
                )).fetchall()
                logger.info("Logical relationships schema:")
                for col in columns:
                    logger.info(f"  {col[0]} - {col[1]}")
        
        # Process embeddings for any chunks missing them
        logger.info("Processing embeddings...")
        await processor.process_embeddings_batch(batch_size=100)
        
        # Build relationships
        logger.info("Building relationships...")
        await processor.build_relationships_sparse(max_neighbors=30, min_confidence=0.55)
        
        # Check resulting relationships
        with processor.db_connection.get_engine().connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
            count_after = result.scalar() or 0
            logger.info(f"Relationships after processing: {count_after}")
            
            if count_after > 0:
                # Show some example relationships
                sample = conn.execute(text(
                    "SELECT relationship_type, COUNT(*), AVG(confidence) FROM logical_relationships GROUP BY relationship_type"
                )).fetchall()
                logger.info("Relationship types:")
                for row in sample:
                    logger.info(f"  {row[0]}: {row[1]} records, avg confidence: {row[2]:.3f}")
                    
                # Show a few examples
                examples = conn.execute(text(
                    "SELECT source_chunk_id, target_chunk_id, relationship_type, confidence, evidence FROM logical_relationships LIMIT 3"
                )).fetchall()
                logger.info("Example relationships:")
                for row in examples:
                    logger.info(f"  {row[0]} -> {row[1]} | Type: {row[2]} | Confidence: {row[3]:.3f}")
                    logger.info(f"  Evidence: {row[4]}")
        
        # Clean up
        await processor.close()
        logger.info("Test completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(test_relationships())
