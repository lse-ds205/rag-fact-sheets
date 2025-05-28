"""
Database connection and SQLAlchemy ORM models for PostgreSQL.
"""
import logging
import traceback
import uuid
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, CheckConstraint, Boolean
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func, text

from schema import DatabaseConfig

# Set up logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy base
Base = declarative_base()

class NDCDocumentORM(Base):
    """SQLAlchemy ORM model for NDC documents"""
    __tablename__ = "documents"
    
    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country = Column(String, nullable=False, index=True)
    title = Column(String)
    url = Column(String, nullable=True)  # Also made url nullable since local files don't have URLs
    language = Column(String)
    submission_date = Column(DateTime)
    file_path = Column(String)
    file_size = Column(Float)
    scraped_at = Column(DateTime(timezone=True), nullable=True)  # Made nullable
    downloaded_at = Column(DateTime(timezone=True))
    processed_at = Column(DateTime(timezone=True))
    last_download_attempt = Column(DateTime(timezone=True))
    download_error = Column(String)
    download_attempts = Column(Integer, default=0)
    extracted_text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship to chunks
    chunks = relationship("DocChunkORM", back_populates="document", cascade="all, delete-orphan")

class DocChunkORM(Base):
    """SQLAlchemy ORM model for document chunks"""
    __tablename__ = "doc_chunks"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Core chunk data
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page = Column(Integer)
    paragraph = Column(Integer)
    language = Column(String)
    
    # Embeddings
    transformer_embedding = Column(ARRAY(Float))
    word2vec_embedding = Column(ARRAY(Float))
    
    # HopRAG-specific columns
    # Note: hoprag_embedding will be added/altered by setup_hoprag_schema to VECTOR type
    # as it's a PostgreSQL-specific type that SQLAlchemy doesn't natively support
    content_hash = Column(String(64))
    chunk_data = Column(JSONB, default={})  # Renamed from metadata to avoid conflicts
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to document
    document = relationship("NDCDocumentORM", back_populates="chunks")

class LogicalRelationshipORM(Base):
    """SQLAlchemy ORM model for logical relationships between chunks"""
    __tablename__ = "logical_relationships"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_chunk_id = Column(UUID(as_uuid=True), ForeignKey("doc_chunks.id", ondelete="CASCADE"), nullable=False)
    target_chunk_id = Column(UUID(as_uuid=True), ForeignKey("doc_chunks.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    evidence = Column(Text)
    method = Column(String(50), default="rule_based")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('relationship_type IN (\'SUPPORTS\', \'EXPLAINS\', \'CONTRADICTS\', \'FOLLOWS\', \'TEMPORAL_FOLLOWS\', \'CAUSES\')', name='valid_relationship_types'),
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='valid_confidence_range'),
        CheckConstraint('source_chunk_id != target_chunk_id', name='no_self_relationships')
    )

# Now define relationships after both classes are defined
DocChunkORM.source_relationships = relationship("LogicalRelationshipORM", 
                                                foreign_keys=[LogicalRelationshipORM.source_chunk_id],
                                                backref="source_chunk", 
                                                cascade="all, delete-orphan")

DocChunkORM.target_relationships = relationship("LogicalRelationshipORM", 
                                                foreign_keys=[LogicalRelationshipORM.target_chunk_id],
                                                backref="target_chunk", 
                                                cascade="all, delete-orphan")

class Connection:
    """Database connection handler"""
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.Session = None

    def connect(self) -> bool:
        """Establish connection to the database"""
        try:
            self.engine = create_engine(
                self.config.url,
                echo=self.config.echo
            )
            self.Session = sessionmaker(bind=self.engine)
            
            # Test the connection
            conn = self.engine.connect()
            conn.close()
            
            logger.info("Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False

    def get_engine(self):
        """Get SQLAlchemy engine instance"""
        if not self.engine:
            raise ValueError("Database connection not established. Call connect() first.")
        return self.engine

    def get_session(self):
        """Get a new session"""
        if not self.Session:
            raise ValueError("Database connection not established. Call connect() first.")
        return self.Session()    
    
    def check_document_processed(self, doc_id: str) -> tuple[bool, Optional[NDCDocumentORM]]:
        """
        Check if a document has already been processed.
        
        Args:
            doc_id: Document ID to check (string filename)
            
        Returns:
            Tuple of (is_processed, document) where is_processed is True if document exists
        """
        self.connect()
        session = self.get_session()
        try:
            # Convert string doc_id to UUID using deterministic UUID5 - SAME AS PROCESSING CODE
            doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
            document = session.query(NDCDocumentORM).filter(NDCDocumentORM.doc_id == doc_uuid).first()
            if document:
                # Check if document has been processed (has processed_at timestamp)
                is_processed = document.processed_at is not None
                return is_processed, document
            else:
                return False, None
        finally:
            session.close()

    def upload(self, items: List, table: str = None) -> bool:
        """
        Upload items to the database.
        
        Args:
            items: List of SQLAlchemy ORM objects to upload
            table: Table name (not used, kept for compatibility)
            
        Returns:
            True if successful, False otherwise
        """
        if not items:
            logger.warning("No items to upload")
            return True
            
        session = self.get_session()
        try:
            # Debug first item to check types
            if items:
                first_item = items[0]
                logger.debug(f"Uploading {len(items)} items of type: {type(first_item).__name__}")
                
                # Special debugging for LogicalRelationshipORM
                if hasattr(first_item, 'source_chunk_id'):
                    logger.debug(f"First relationship: id={first_item.id} (type: {type(first_item.id)}), "
                               f"source={first_item.source_chunk_id} (type: {type(first_item.source_chunk_id)}), "
                               f"target={first_item.target_chunk_id} (type: {type(first_item.target_chunk_id)})")
            
            for i, item in enumerate(items):
                try:
                    # Validate UUID fields for LogicalRelationshipORM
                    if hasattr(item, 'source_chunk_id'):
                        # Ensure all UUID fields are proper UUID objects
                        if isinstance(item.id, str):
                            item.id = uuid.UUID(item.id)
                        if isinstance(item.source_chunk_id, str):
                            item.source_chunk_id = uuid.UUID(item.source_chunk_id)
                        if isinstance(item.target_chunk_id, str):
                            item.target_chunk_id = uuid.UUID(item.target_chunk_id)
                    
                    session.add(item)
                    
                except Exception as item_error:
                    logger.error(f"Error adding item {i}: {item_error}")
                    # Skip this item and continue
                    continue
            
            session.commit()
            logger.info(f"Successfully uploaded {len(items)} items to database")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error uploading items to database: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Additional debugging for UUID-related errors
            if "UUID" in str(e) or "sentinel" in str(e):
                logger.error("UUID-related error detected. Checking item types...")
                for i, item in enumerate(items[:3]):  # Check first 3 items
                    if hasattr(item, 'source_chunk_id'):
                        logger.error(f"Item {i}: id={item.id} ({type(item.id)}), "
                                   f"source={item.source_chunk_id} ({type(item.source_chunk_id)}), "
                                   f"target={item.target_chunk_id} ({type(item.target_chunk_id)})")
            
            return False
        finally:
            session.close()    

    def update_processed(self, model_class, doc_id: str, chunks=None, table: str = None):
        """
        Update the processed status of a document.
        
        Args:
            model_class: SQLAlchemy model class (NDCDocumentORM)
            doc_id: Document ID to update (string filename)
            chunks: Chunks data (optional)
            table: Table name (not used, kept for compatibility)
        """
        session = self.get_session()
        try:
            # Convert string doc_id to UUID using deterministic UUID5 - SAME AS PROCESSING CODE
            doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
            document = session.query(model_class).filter(model_class.doc_id == doc_uuid).first()
            if document:
                document.processed_at = datetime.now()
                if chunks:
                    # Optionally store chunk count or other metadata
                    pass
                session.commit()
                logger.info(f"Updated processed status for document {doc_id}")
            else:
                logger.warning(f"Document {doc_id} not found for update")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating processed status for {doc_id}: {e}")
        finally:
            session.close()

    def get_document_by_filename(self, filename: str) -> Optional[NDCDocumentORM]:
        """
        Get document by filename (for testing purposes).
        
        Args:
            filename: Document filename
            
        Returns:
            Document ORM object or None
        """
        session = self.get_session()
        try:
            # Convert filename to UUID the same way as processing code
            doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, filename)
            document = session.query(NDCDocumentORM).filter(NDCDocumentORM.doc_id == doc_uuid).first()
            return document
        finally:
            session.close()

    def count_relationships(self, by_type: bool = False) -> int:
        """
        Count logical relationships in the database.
            
        Args:
            by_type: If True, return counts by relationship type
            
        Returns:
            Count of relationships or dict of counts by type
        """
        session = self.get_session()
        try:
            if by_type:
                # Return counts by relationship type
                from sqlalchemy import func
                result = session.query(
                    LogicalRelationshipORM.relationship_type, 
                    func.count(LogicalRelationshipORM.id)
                ).group_by(LogicalRelationshipORM.relationship_type).all()
                return dict(result)
            else:
                # Return total count
                count = session.query(LogicalRelationshipORM).count()
                return count
        except Exception as e:
            logger.error(f"Error counting relationships: {e}")
            return 0 if not by_type else {}
        finally:
            session.close()

    def check_embedding_quality(self) -> Dict[str, Any]:
        """
        Check the quality and consistency of embeddings in the database.
        
        Returns:
            Dict with statistics about embeddings
        """
        engine = self.get_engine()
        try:
            with engine.connect() as conn:
                # Get embedding statistics
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(hoprag_embedding) as chunks_with_embeddings,
                        COUNT(*) - COUNT(hoprag_embedding) as chunks_without_embeddings
                    FROM doc_chunks
                """)).fetchone()
                
                stats = {
                    'total_chunks': result.total_chunks,
                    'chunks_with_embeddings': result.chunks_with_embeddings,
                    'chunks_without_embeddings': result.chunks_without_embeddings
                }
                
                # Check embedding data types and formats
                type_check = conn.execute(text("""
                    SELECT 
                        pg_typeof(hoprag_embedding) as data_type,
                        COUNT(*) as count,
                        array_length(hoprag_embedding, 1) as typical_length
                    FROM doc_chunks 
                    WHERE hoprag_embedding IS NOT NULL 
                    GROUP BY pg_typeof(hoprag_embedding), array_length(hoprag_embedding, 1)
                    ORDER BY count DESC
                """)).fetchall()
                
                stats['embedding_types'] = [
                    {
                        'data_type': str(row.data_type),
                        'count': row.count,
                        'typical_length': row.typical_length
                    }
                    for row in type_check
                ]
                
                # Check for problematic embeddings
                problematic = conn.execute(text("""
                    SELECT id, hoprag_embedding, 
                           pg_typeof(hoprag_embedding) as embedding_type,
                           array_length(hoprag_embedding, 1) as embedding_length,
                           substring(content, 1, 50) as content_sample
                    FROM doc_chunks 
                    WHERE hoprag_embedding IS NOT NULL 
                    AND (
                        array_length(hoprag_embedding, 1) IS NULL OR 
                        array_length(hoprag_embedding, 1) < 10 OR
                        array_length(hoprag_embedding, 1) > 1000 OR
                        pg_typeof(hoprag_embedding)::text != 'double precision[]'
                    )
                    LIMIT 10
                """)).fetchall()
                
                stats['problematic_embeddings'] = [
                    {
                        'chunk_id': str(row.id),
                        'embedding_type': str(row.embedding_type),
                        'embedding_length': row.embedding_length,
                        'content_sample': row.content_sample,
                        'embedding_sample': str(row.hoprag_embedding)[:100] if row.hoprag_embedding else None
                    }
                    for row in problematic
                ]
                
                logger.info(f"Embedding quality check: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error checking embedding quality: {e}")
            return {}

    def fix_string_embeddings(self, dry_run: bool = True) -> int:
        """
        Fix embeddings that are stored as strings instead of arrays.
        
        Args:
            dry_run: If True, only count what would be fixed, don't actually update
            
        Returns:
            Number of embeddings that need fixing/were fixed
        """
        engine = self.get_engine()
        try:
            with engine.connect() as conn:
                # Find string-type embeddings that should be arrays
                string_embeddings_query = text("""
                    SELECT id, hoprag_embedding
                    FROM doc_chunks 
                    WHERE hoprag_embedding IS NOT NULL 
                    AND pg_typeof(hoprag_embedding)::text != 'double precision[]'
                """)
                
                string_embeddings = conn.execute(string_embeddings_query).fetchall()
                count = len(string_embeddings)
                
                if dry_run:
                    logger.info(f"Found {count} string embeddings that need conversion (dry run)")
                    if count > 0:
                        # Show samples
                        for i, row in enumerate(string_embeddings[:3]):
                            logger.info(f"  Sample {i+1}: {str(row.hoprag_embedding)[:100]}...")
                    return count
                
                if count > 0:
                    logger.info(f"Converting {count} string embeddings to proper arrays...")
                    
                    for i, row in enumerate(string_embeddings):
                        try:
                            # Parse the string embedding
                            embedding_str = str(row.hoprag_embedding)
                            clean_str = embedding_str.strip('[]')
                            embedding_values = [float(x.strip()) for x in clean_str.split(',')]
                            
                            # Update the database with proper array
                            update_query = text("""
                                UPDATE doc_chunks 
                                SET hoprag_embedding = :embedding_array
                                WHERE id = :chunk_id
                            """)
                            
                            conn.execute(update_query, {
                                "embedding_array": embedding_values,
                                "chunk_id": row.id
                            })
                            
                            if (i + 1) % 100 == 0:
                                logger.info(f"  Converted {i + 1}/{count} embeddings...")
                                
                        except Exception as e:
                            logger.warning(f"Failed to convert embedding for chunk {row.id}: {e}")
                            continue
                    
                    conn.commit()
                    logger.info(f"Successfully converted {count} string embeddings to arrays")
                
                return count
                
        except Exception as e:
            logger.error(f"Error fixing string embeddings: {e}")
            return 0

    def get_all_chunks_for_evaluation(self, country: Optional[str] = None, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all chunks from database for comprehensive evaluation.
        
        Args:
            country: Optional country filter
            batch_size: Number of chunks to retrieve per batch
            
        Returns:
            List of chunk dictionaries
        """
        engine = self.get_engine()
        all_chunks = []
        
        try:
            with engine.connect() as conn:
                if country:
                    # Get chunks filtered by country
                    query = text("""
                        SELECT 
                            dc.id,
                            dc.content,
                            dc.chunk_index,
                            dc.paragraph,
                            dc.language,
                            dc.transformer_embedding,
                            dc.word2vec_embedding,
                            dc.chunk_data,
                            d.country,
                            d.doc_id,
                            d.title
                        FROM doc_chunks dc
                        JOIN documents d ON dc.doc_id = d.doc_id
                        WHERE d.country = :country
                        AND dc.transformer_embedding IS NOT NULL
                        ORDER BY dc.id
                    """)
                    result = conn.execute(query, {"country": country}).fetchall()
                else:
                    # Get all chunks
                    query = text("""
                        SELECT 
                            dc.id,
                            dc.content,
                            dc.chunk_index,
                            dc.paragraph,
                            dc.language,
                            dc.transformer_embedding,
                            dc.word2vec_embedding,
                            dc.chunk_data,
                            d.country,
                            d.doc_id,
                            d.title
                        FROM doc_chunks dc
                        JOIN documents d ON dc.doc_id = d.doc_id
                        WHERE dc.transformer_embedding IS NOT NULL
                        ORDER BY dc.id
                    """)
                    result = conn.execute(query).fetchall()
                
                # Convert to list of dictionaries
                for row in result:
                    chunk_dict = {
                        'id': str(row.id),
                        'content': row.content,
                        'chunk_index': row.chunk_index,
                        'paragraph': row.paragraph,
                        'language': row.language,
                        'transformer_embedding': row.transformer_embedding,
                        'word2vec_embedding': row.word2vec_embedding,
                        'chunk_data': row.chunk_data or {},
                        'country': row.country,
                        'doc_id': str(row.doc_id),
                        'title': row.title
                    }
                    all_chunks.append(chunk_dict)
                
                logger.info(f"Retrieved {len(all_chunks)} chunks for evaluation{' for country ' + country if country else ''}")
                return all_chunks
                
        except Exception as e:
            logger.error(f"Error retrieving chunks for evaluation: {e}")
            return []


