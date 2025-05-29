"""Database operations for document scraping."""

import logging
from datetime import datetime
from typing import List
import uuid
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import group4py
from databases.auth import PostgresConnection
from databases.models import NDCDocumentORM
from schemas.db_pydantic import NDCDocumentModel
from exceptions import DatabaseConnectionError, DocumentValidationError

logger = logging.getLogger(__name__)
db = PostgresConnection()

def retrieve_existing_documents() -> List[NDCDocumentModel]:
    """
    Retrieve existing documents from the database.
    
    Returns:
        List of documents currently in the database
        
    Raises:
        DatabaseConnectionError: If database connection fails
    """
    logger.info("Retrieving existing documents from database")
    
    try:
        with db.Session() as session:
            db_documents = session.query(NDCDocumentORM).all()
            existing_docs = [_convert_db_to_model(db_doc) for db_doc in db_documents]
            logger.info(f"Retrieved {len(existing_docs)} existing documents from database")
            return existing_docs
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise DatabaseConnectionError(f"Database connection failed: {str(e)}") from e


def insert_new_documents(new_docs: List[NDCDocumentModel]) -> int:
    """
    Insert new documents into the database.
    
    Args:
        new_docs: List of new documents to insert
        
    Returns:
        Number of successfully inserted documents
        
    Raises:
        DatabaseConnectionError: If database connection fails
        DocumentValidationError: If document validation fails
    """
    if not new_docs:
        logger.info("No new documents to insert")
        return 0
    
    logger.info(f"Inserting {len(new_docs)} new documents into database")
    
    try:
        with db.Session() as session:
            db_documents = []
            for doc in new_docs:
                try:
                    db_doc = _convert_base_to_db(doc)
                    db_documents.append(db_doc)
                except Exception as e:
                    logger.error(f"Error creating database object for {doc.url}: {str(e)}")
                    raise DocumentValidationError(f"Document validation failed for {doc.url}: {str(e)}") from e
            
            if not db_documents:
                logger.warning("No valid documents to insert after conversion")
                return 0
            
            session.add_all(db_documents)
            session.commit()
            logger.info(f"Successfully inserted {len(db_documents)} new documents")
            return len(db_documents)
    except (DatabaseConnectionError, DocumentValidationError):
        raise
    except Exception as e:
        logger.error(f"Error during document insertion: {str(e)}")
        raise DatabaseConnectionError(f"Document insertion failed: {str(e)}") from e


def update_existing_documents(updated_docs: List[NDCDocumentModel]) -> int:
    """
    Update existing documents in the database with new metadata.
    
    Args:
        updated_docs: List of documents with updated metadata
        
    Returns:
        Number of successfully updated documents
        
    Raises:
        DatabaseConnectionError: If database connection fails
    """
    if not updated_docs:
        logger.info("No documents to update")
        return 0
    
    logger.info(f"Updating {len(updated_docs)} existing documents in database")
    
    try:
        with db.Session() as session:
            updated_count = 0
            
            for doc in updated_docs:
                try:
                    existing_doc = session.query(NDCDocumentORM).filter(NDCDocumentORM.url == doc.url).first()
                    
                    if existing_doc:
                        _update_document_metadata(existing_doc, doc)
                        updated_count += 1
                        logger.debug(f"Updated document: {existing_doc.doc_id}")
                    else:
                        logger.warning(f"Document not found for update: {doc.url}")
                except Exception as e:
                    logger.error(f"Error updating document {doc.url}: {str(e)}")
                    continue
            
            session.commit()
            logger.info(f"Successfully updated {updated_count} documents")
            return updated_count
    except Exception as e:
        logger.error(f"Error during document updates: {str(e)}")
        raise DatabaseConnectionError(f"Document update failed: {str(e)}") from e


def _convert_db_to_model(db_doc: NDCDocumentORM) -> NDCDocumentModel:
    """Convert SQLAlchemy NDCDocumentORM to NDCDocumentModel."""
    # Create a dictionary of non-None attributes to pass to NDCDocumentModel
    doc_data = {
        "doc_id": db_doc.doc_id,
        "country": db_doc.country,
        "url": db_doc.url,
        "title": db_doc.title,
        "language": db_doc.language,
        "submission_date": db_doc.submission_date,
        "file_path": db_doc.file_path,
        "file_size": db_doc.file_size,
        "scraped_at": db_doc.scraped_at,
        "downloaded_at": db_doc.downloaded_at,
        "processed_at": db_doc.processed_at,
        "last_download_attempt": db_doc.last_download_attempt,
        "download_error": db_doc.download_error,
        "download_attempts": db_doc.download_attempts or 0,
        "extracted_text": db_doc.extracted_text,
    }
    
    # Only add created_at and updated_at if they're not None
    if db_doc.created_at is not None:
        doc_data["created_at"] = db_doc.created_at
    
    if db_doc.updated_at is not None:
        doc_data["updated_at"] = db_doc.updated_at
    
    return NDCDocumentModel(**doc_data)


def _convert_base_to_db(doc: NDCDocumentModel) -> NDCDocumentORM:
    """Convert NDCDocumentModel to SQLAlchemy NDCDocumentORM."""
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, doc.url)
    return NDCDocumentORM(
        doc_id=doc_id,
        country=doc.country,
        title=doc.title,
        url=doc.url,
        language=doc.language,
        submission_date=doc.submission_date,
        scraped_at=datetime.now()
    )


def _update_document_metadata(existing_doc: NDCDocumentORM, new_doc: NDCDocumentModel) -> None:
    """Update existing document with new metadata."""
    existing_doc.title = new_doc.title
    existing_doc.language = new_doc.language
    existing_doc.submission_date = new_doc.submission_date
    existing_doc.country = new_doc.country
    existing_doc.updated_at = datetime.now() 