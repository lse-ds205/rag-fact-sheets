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

# Load environment variables from .env file
load_dotenv()

# Set up paths to ensure imports work correctly
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import from the project
from group4py.src.database import Connection, Base
from group4py.src.schema import DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_database")

def setup_database(db_url=None, echo=True):
    """
    Function to set up the database tables in PostgreSQL
    
    Args:
        db_url: PostgreSQL connection URL (optional, will use DATABASE_URL from .env if not provided)
        echo: Whether to echo SQL commands
        
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
    
    args = parser.parse_args()
    
    # Use provided URL or get from environment
    db_url = args.db_url
        
    if setup_database(db_url, args.echo):
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")
        sys.exit(1)