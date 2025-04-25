import logging
import sys
from pathlib import Path
from sqlalchemy import Column, String, Integer, Float, Boolean, JSON, Text, DateTime, Date, ForeignKey, text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.types import UserDefinedType
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, date
import json
import os
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.dialects.postgresql import ARRAY

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.schema import DatabaseConfig
from helpers import Logger, Test, TaskInfo

Base = declarative_base()
logger = logging.getLogger(__name__)

class Vector(UserDefinedType):
    """PostgreSQL vector type for pgvector."""
    
    def __init__(self, dim):
        self.dim = dim
    
    def get_col_spec(self):
        return f"vector({self.dim})"
    
    def bind_expression(self, bindvalue):
        return bindvalue
    
    def column_expression(self, col):
        return col
    
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            # Convert Python list to string format needed for pgvector
            return f"[{','.join(str(x) for x in value)}]"
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            # Convert string representation back to Python list
            # This assumes the format is '[x,y,z,...]'
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                return [float(x) for x in value[1:-1].split(',') if x]
            return value
        return process

class Document(Base):
    """SQLAlchemy model for documents."""
    __tablename__ = 'documents'
    
    doc_id = Column(String, primary_key=True)
    
    # Metadata fields
    scraped_at = Column(DateTime, default=datetime.now)
    downloaded_at = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    
    # Download attempt tracking
    last_download_attempt = Column(DateTime, nullable=True)
    download_error = Column(String, nullable=True)
    download_attempts = Column(Integer, default=0)
    
    # Basic document information
    country = Column(String, nullable=False)
    title = Column(String, nullable=True)
    url = Column(String, nullable=False)
    language = Column(String, nullable=True)
    submission_date = Column(Date, nullable=True)
    file_path = Column(String, nullable=True)
    file_size = Column(Float, nullable=True)
    
    # Extracted content
    extracted_text = Column(Text, nullable=True)
    chunks = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationship to chunks
    doc_chunks = relationship("DocChunkDB", back_populates="document")

class DocChunkDB(Base):
    """SQLAlchemy model for document chunks."""
    __tablename__ = 'doc_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey('documents.doc_id'), nullable=False)
    
    # Chunk content and metadata
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    paragraph = Column(Integer, nullable=True)
    language = Column(String, nullable=True)
    
    # Embeddings - using PostgreSQL ARRAY type for vector storage
    transformer_embedding = Column(ARRAY(Float), nullable=True)
    word2vec_embedding = Column(ARRAY(Float), nullable=True)
    
    # Optional metadata about the chunk
    chunk_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationship back to document
    document = relationship("Document", back_populates="doc_chunks")

class ExtractedDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special data types."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

class Connection:
    def __init__(self, config: Optional[DatabaseConfig] = None, db_url: Optional[str] = None):
        """
        Initialize database connection with configuration
        
        Args:
            config: DatabaseConfig instance with connection settings
            db_url: Database URL string (alternative to providing config)
        """
        self.engine = None
        self.session_factory = None
        
        # Get database URL from config, parameter, or environment
        if config:
            self.db_url = config.url
            self.create_tables = config.create_tables
            self.echo = config.echo
        elif db_url:
            self.db_url = db_url
            self.create_tables = False
            self.echo = False
        else:
            self.db_url = os.environ.get('DATABASE_URL')
            self.create_tables = os.environ.get('CREATE_TABLES', 'False').lower() == 'true'
            self.echo = os.environ.get('ECHO_SQL', 'False').lower() == 'true'
            
        if not self.db_url:
            logger.warning("<DATABASE> No database URL provided. Database operations will not work.")
        
        # Verify that we're using PostgreSQL
        if self.db_url and not self.db_url.startswith('postgresql'):
            error_msg = "<DATABASE> Only PostgreSQL is supported. Please provide a PostgreSQL connection URL."
            logger.error(error_msg)
            raise ValueError(error_msg)

    @Logger.debug_log()
    def get_engine(self):
        """Get SQLAlchemy engine, creating if necessary"""
        logger.info("<DATABASE> Getting engine...")
        
        if not self.engine and self.db_url:
            try:
                # Verify that we're using PostgreSQL
                if not self.db_url.startswith('postgresql'):
                    error_msg = "<DATABASE> Only PostgreSQL is supported. Please provide a PostgreSQL connection URL."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                self.engine = create_engine(
                    self.db_url, 
                    echo=self.echo,
                    # Additional engine parameters can be added as needed
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                logger.info("<DATABASE> Engine created successfully")
            except Exception as e:
                logger.error(f"<DATABASE> Error creating engine: {e}")
                raise
                
        return self.engine

    @Logger.debug_log()
    def connect(self):
        """Connect to database and initialize session factory"""
        logger.info("<DATABASE> Connecting...")
        
        try:
            engine = self.get_engine()
            if not engine:
                logger.error("<DATABASE> No engine available, cannot connect")
                return False
                
            # Create session factory
            self.session_factory = sessionmaker(bind=engine)
            
            # Create tables if specified
            if self.create_tables:
                # Try to create the extension if it doesn't exist (for pgvector)
                with engine.connect() as conn:
                    try:
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        conn.commit()
                        logger.info("<DATABASE> Enabled pgvector extension")
                    except Exception as e:
                        logger.warning(f"<DATABASE> Could not create pgvector extension: {e}")
                
                Base.metadata.create_all(engine)
                logger.info("<DATABASE> Created database tables")
                
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logger.warning("<DATABASE> Connected to database successfully")
            return True
            
        except Exception as e:
            logger.error(f"<DATABASE> Connection error: {e}")
            return False

    @Logger.debug_log()
    def get_session(self):
        """Get a new database session"""
        if not self.session_factory:
            self.connect()
        
        if self.session_factory:
            return self.session_factory()
        else:
            raise Exception("Database not connected, cannot create session")

    @Logger.debug_log()
    def upload(self, data: Any, table: str = None):
        """
        Upload data to database
        
        Args:
            data: Data to upload (can be Document, DocChunkDB, or list of either)
            table: Table to upload to ('documents' or 'doc_chunks'). If None, determines automatically from data type.
            
        Returns:
            bool: Success status
        """
        logger.info("<DATABASE> Uploading data into database...")
        
        # Validate table parameter if provided
        if table and table not in ['documents', 'doc_chunks']:
            logger.error(f"<DATABASE> Invalid table specified: {table}. Must be 'documents' or 'doc_chunks'.")
            return False
        
        try:
            session = self.get_session()
            
            try:
                # Handle different types of data
                if isinstance(data, list):
                    # Validate data types if table is specified
                    if table:
                        expected_class = Document if table == 'documents' else DocChunkDB
                        if not all(isinstance(item, expected_class) for item in data):
                            logger.error(f"<DATABASE> Data type mismatch for table '{table}'")
                            return False
                    session.add_all(data)
                else:
                    # Validate data type if table is specified
                    if table:
                        expected_class = Document if table == 'documents' else DocChunkDB
                        if not isinstance(data, expected_class):
                            logger.error(f"<DATABASE> Data type mismatch for table '{table}'")
                            return False
                    session.add(data)
                
                session.commit()
                logger.info(f"<DATABASE> Uploaded to {table or 'database'} successfully")
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"<DATABASE> Error during upload: {e}")
                return False
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"<DATABASE> Session creation error: {e}")
            return False

    @Logger.debug_log()
    def update_processed(self, model_class, record_id, chunks=None, id_field='doc_id', table: str = None):
        """
        Mark a record as processed in the database
        
        Args:
            model_class: SQLAlchemy model class to update
            record_id: Identifier value for the record
            chunks: Optional data to store with the record
            id_field: Name of the identifier field (default: 'doc_id')
            table: Table to update ('documents' or 'doc_chunks'). If None, determines automatically from model_class.
            
        Returns:
            bool: True if update successful, False otherwise
        """
        logger.info(f"<DATABASE> Updating processed status for {model_class.__name__} with {id_field}={record_id}")
        
        # Validate table parameter if provided
        if table and table not in ['documents', 'doc_chunks']:
            logger.error(f"<DATABASE> Invalid table specified: {table}. Must be 'documents' or 'doc_chunks'.")
            return False
            
        # Match model_class with table if specified
        if table:
            expected_class = Document if table == 'documents' else DocChunkDB
            if model_class != expected_class:
                logger.warning(f"<DATABASE> Model class {model_class.__name__} doesn't match specified table '{table}'")
        
        try:
            session = self.get_session()
            
            try:
                # Build dynamic filter
                filter_condition = getattr(model_class, id_field) == record_id
                record = session.query(model_class).filter(filter_condition).first()
                
                if record:
                    # Update processed timestamp
                    setattr(record, 'processed_at', datetime.now())
                    
                    # Update chunks if provided and the model has that field
                    if chunks and hasattr(record, 'chunks'):
                        try:
                            # Convert chunks to JSON string
                            chunks_json = json.dumps(chunks, cls=ExtractedDataEncoder)
                            setattr(record, 'chunks', chunks_json)
                        except Exception as e:
                            logger.error(f"<DATABASE> Error serializing chunks: {e}")
                    
                    session.commit()
                    logger.info(f"<DATABASE> Successfully updated {model_class.__name__} record: {record_id}")
                    return True
                else:
                    logger.warning(f"<DATABASE> {model_class.__name__} record not found: {record_id}")
            except Exception as e:
                logger.error(f"<DATABASE> Database update error: {e}")
                session.rollback()
            finally:
                session.close()
        
        except Exception as e:
            logger.error(f"<DATABASE> Database connection error: {e}")
        
        return False
        
    @Logger.debug_log()
    def get_document_metadata(self, doc_id, table: str = 'documents'):
        """
        Get document metadata from database
        
        Args:
            doc_id: Document identifier
            table: Table to query ('documents' by default). Must be a valid table name.
            
        Returns:
            dict: Dictionary with metadata or None if not found
        """
        # Validate table parameter
        if table not in ['documents', 'doc_chunks']:
            logger.error(f"<DATABASE> Invalid table specified: {table}. Must be 'documents' or 'doc_chunks'.")
            return None
            
        logger.info(f"<DATABASE> Getting metadata for document {doc_id} from table '{table}'")
        
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                if table == 'documents':
                    query = text("SELECT country, title, submission_date FROM documents WHERE doc_id = :doc_id")
                    result = conn.execute(query, {"doc_id": doc_id})
                    row = result.fetchone()
                    if not row:
                        return None
                    
                    submission_date = row[2]
                    if submission_date and isinstance(submission_date, datetime):
                        submission_date = submission_date.isoformat()
                    elif submission_date:
                        submission_date = str(submission_date)
                    
                    return {
                        'country': row[0] or '',
                        'document_title': row[1] or '',
                        'submission_date': submission_date or ''
                    }
                elif table == 'doc_chunks':
                    # Get metadata for chunks related to the document
                    query = text("SELECT chunk_index, language, chunk_metadata FROM doc_chunks WHERE doc_id = :doc_id")
                    result = conn.execute(query, {"doc_id": doc_id})
                    rows = result.fetchall()
                    if not rows:
                        return None
                    
                    chunks_metadata = []
                    for row in rows:
                        metadata = {
                            'chunk_index': row[0],
                            'language': row[1] or '',
                            'metadata': row[2] or {}
                        }
                        chunks_metadata.append(metadata)
                    
                    return {
                        'doc_id': doc_id,
                        'chunks_count': len(chunks_metadata),
                        'chunks_metadata': chunks_metadata
                    }
        except Exception as e:
            logger.error(f"<DATABASE> Error retrieving document metadata from {table}: {e}")
            return None
        
    @Logger.debug_log()
    def check_document_processed(self, doc_id: str) -> Tuple[bool, Optional[Document]]:
        """
        Check if a document has been processed already
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple[bool, Optional[Document]]: 
                - First value: True if document is processed, False otherwise
                - Second value: Document object if found, None otherwise
        """
        logger.info(f"<DATABASE> Checking if document {doc_id} has been processed")
        
        try:
            session = self.get_session()
            
            try:
                # Find the document record
                document = session.query(Document).filter(Document.doc_id == doc_id).first()
                
                # Check if document exists and has been processed
                if document and document.processed_at is not None:
                    logger.info(f"<DATABASE> Document {doc_id} has already been processed")
                    return True, document
                elif document:
                    logger.info(f"<DATABASE> Document {doc_id} exists but hasn't been processed yet")
                    return False, document
                else:
                    logger.info(f"<DATABASE> Document {doc_id} not found in database")
                    return False, None
                    
            except Exception as e:
                logger.error(f"<DATABASE> Error checking document processed status: {e}")
                return False, None
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"<DATABASE> Database connection error: {e}")
            return False, None


