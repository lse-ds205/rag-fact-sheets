from datetime import datetime
import sys
import uuid
from pathlib import Path
import logging
from typing import List, Optional
import traceback

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import group4py
from databases.models import NDCDocumentORM

logger = logging.getLogger(__name__)


def check_document_processed(session, doc_id: str) -> tuple[bool, Optional[NDCDocumentORM]]:
    """
    Check if a document has already been processed.
    
    Args:
        doc_id: Document ID to check (string filename)
        
    Returns:
        Tuple of (is_processed, document) where is_processed is True if document exists
    """

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

def upload(session, items: List, table: str = None) -> bool:
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


def update_processed(session, model_class, doc_id: str, chunks=None, table: str = None):
        """
        Update the processed status of a document.
        
        Args:
            model_class: SQLAlchemy model class (NDCDocumentORM)
            doc_id: Document ID to update (string filename)
            chunks: Chunks data (optional)
            table: Table name (not used, kept for compatibility)
        """
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