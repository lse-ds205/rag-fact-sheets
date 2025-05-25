"""Document comparison utilities."""

import logging
from typing import Dict, List, Union

from ..schema import NDCDocumentModel

logger = logging.getLogger(__name__)


def compare_documents(
    existing_docs: List[NDCDocumentModel], 
    new_docs: List[NDCDocumentModel]
) -> Dict[str, List[NDCDocumentModel]]:
    """
    Compare existing and new documents to find differences.
    
    Args:
        existing_docs: Documents currently in database
        new_docs: Documents scraped from website
        
    Returns:
        Dictionary with 'new', 'updated', and 'removed' document lists
    """
    logger.info(f"Comparing {len(existing_docs)} existing docs with {len(new_docs)} new docs")
    
    existing_by_url = {doc.url: doc for doc in existing_docs}
    new_by_url = {doc.url: doc for doc in new_docs}
    
    new_documents = [
        doc for url, doc in new_by_url.items() 
        if url not in existing_by_url
    ]
    
    updated_documents = [
        new_doc for url, new_doc in new_by_url.items()
        if url in existing_by_url and _has_metadata_changed(existing_by_url[url], new_doc)
    ]
    
    removed_documents = [
        doc for url, doc in existing_by_url.items() 
        if url not in new_by_url
    ]
    
    result = {
        'new': new_documents,
        'updated': updated_documents,
        'removed': removed_documents
    }
    
    logger.info(f"Comparison results: {len(new_documents)} new, {len(updated_documents)} updated, {len(removed_documents)} removed")
    return result


def _has_metadata_changed(existing_doc: NDCDocumentModel, new_doc: NDCDocumentModel) -> bool:
    """Check if document metadata has changed."""
    return (
        existing_doc.title != new_doc.title or 
        existing_doc.language != new_doc.language or 
        existing_doc.submission_date != new_doc.submission_date or
        existing_doc.country != new_doc.country
    ) 