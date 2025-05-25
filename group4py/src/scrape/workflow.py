"""Main scraping workflow orchestration."""

import logging
import traceback
from typing import Dict, Any, Optional

from .selenium import scrape_ndc_documents
from .db_operations import retrieve_existing_documents, insert_new_documents, update_existing_documents
from .comparator import compare_documents
from .config import ScrapingConfig, DEFAULT_CONFIG
from .exceptions import WorkflowError, DocumentScrapingError

logger = logging.getLogger(__name__)


def run_scraping_workflow(config: Optional[ScrapingConfig] = None) -> Dict[str, Any]:
    """
    Execute the complete NDC document scraping workflow.
    
    Args:
        config: Optional configuration for scraping operations
    
    Returns:
        Dictionary containing workflow results and statistics
        
    Raises:
        WorkflowError: If workflow execution fails
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    logger.info("Starting NDC document scraping workflow")
    
    try:
        # Step 1: Retrieve existing documents from database
        logger.info("Step 1: Retrieving existing documents from database...")
        existing_docs = retrieve_existing_documents()
        logger.info(f"Found {len(existing_docs)} existing documents in database")
        
        # Step 2: Scrape fresh documents from website
        logger.info("Step 2: Scraping fresh documents from website...")
        try:
            new_docs = scrape_ndc_documents(headless=config.headless, timeout=config.timeout)
            logger.info(f"Scraped {len(new_docs)} documents from website")
        except Exception as e:
            raise DocumentScrapingError(f"Failed to scrape documents: {str(e)}") from e
        
        if not new_docs and config.abort_on_no_docs:
            logger.warning("No documents scraped from website - aborting workflow")
            return _create_result_summary(existing_docs, [], {}, 0, 0)
        
        # Step 3: Compare documents to find changes
        logger.info("Step 3: Comparing documents to identify changes...")
        changes = compare_documents(existing_docs, new_docs)
        
        # Step 4: Process new documents
        inserted_count = 0
        if changes['new']:
            logger.info(f"Step 4: Processing {len(changes['new'])} new documents...")
            inserted_count = insert_new_documents(changes['new'])
            logger.info(f"Successfully inserted {inserted_count}/{len(changes['new'])} new documents")
        else:
            logger.info("Step 4: No new documents to process")
        
        # Step 5: Process updated documents
        updated_count = 0
        if changes['updated']:
            logger.info(f"Step 5: Processing {len(changes['updated'])} updated documents...")
            updated_count = update_existing_documents(changes['updated'])
            logger.info(f"Successfully updated {updated_count}/{len(changes['updated'])} documents")
        else:
            logger.info("Step 5: No documents to update")
        
        # Step 6: Handle removed documents
        if changes['removed']:
            if config.process_removed_docs:
                logger.info(f"Step 6: Processing {len(changes['removed'])} removed documents...")
                # TODO: Implement removed document processing
                logger.info("Removed document processing not yet implemented")
            else:
                logger.info(f"Step 6: Found {len(changes['removed'])} removed documents (not processing)")
                _log_removed_documents(changes['removed'], config.log_removed_limit)
        else:
            logger.info("Step 6: No removed documents found")
        
        # Step 7: Summary
        result = _create_result_summary(existing_docs, new_docs, changes, inserted_count, updated_count)
        _log_workflow_summary(result)
        
        logger.info("NDC document scraping workflow completed successfully!")
        return result
        
    except (DocumentScrapingError, WorkflowError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scraping workflow: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise WorkflowError(f"Workflow execution failed: {str(e)}") from e


def _log_removed_documents(removed_docs, limit: int = 5) -> None:
    """Log details of removed documents."""
    for doc in removed_docs[:limit]:
        logger.info(f"  Removed: {doc.country} - {doc.title} - {doc.url}")
    if len(removed_docs) > limit:
        logger.info(f"  ... and {len(removed_docs) - limit} more")


def _create_result_summary(existing_docs, new_docs, changes, inserted_count, updated_count) -> Dict[str, Any]:
    """Create workflow result summary."""
    return {
        'existing_count': len(existing_docs),
        'scraped_count': len(new_docs),
        'new_count': len(changes.get('new', [])),
        'updated_count': len(changes.get('updated', [])),
        'removed_count': len(changes.get('removed', [])),
        'inserted_count': inserted_count,
        'updated_actual_count': updated_count,
        'success': True
    }


def _log_workflow_summary(result: Dict[str, Any]) -> None:
    """Log comprehensive workflow summary."""
    logger.info("=" * 60)
    logger.info("WORKFLOW SUMMARY:")
    logger.info(f"  Documents in database: {result['existing_count']}")
    logger.info(f"  Documents on website: {result['scraped_count']}")
    logger.info(f"  New documents inserted: {result['inserted_count']}")
    logger.info(f"  Documents updated: {result['updated_actual_count']}")
    logger.info(f"  Documents removed (not processed): {result['removed_count']}")
    logger.info("=" * 60) 