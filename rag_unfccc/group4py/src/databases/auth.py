
"""
This module contains the authentication and connection logic for the database.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import supabase
from contextlib import contextmanager

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

logger = logging.getLogger(__name__)


class PostgresConnection:
    def __init__(self):
        logger.info("Initializing PostgresConnection...")
        
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set")
        self.engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def connect(self):
        """Return a database connection as a context manager"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def get_engine(self):
        """Get SQLAlchemy engine instance"""
        return self.engine

    @contextmanager
    def get_session(self):
        """Context-managed session with automatic commit/rollback/close.

        Usage:
            with db.get_session() as session:
                session.add(...)
        """
        if not self.Session:
            raise ValueError("Database connection not established. Call connect() first.")

        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()


class SupabaseConnection:
    def __init__(self):
        self.supabase = supabase.create_client(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_SERVICE_KEY
        )

    def supabase_client(self):
        return self.supabase