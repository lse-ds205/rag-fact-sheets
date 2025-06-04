"""Main scraping workflow orchestration."""

import logging
import traceback
from typing import Dict, Any, Optional
import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import group4py

from scrape.selenium import scrape_ndc_documents
from scrape.db_operations import retrieve_existing_documents, insert_new_documents, update_existing_documents
from scrape.comparator import compare_documents
from scrape.config import ScrapingConfig, DEFAULT_CONFIG
from scrape.exceptions import (
    WorkflowError, 
    DocumentScrapingError, 
    DocumentDownloadError, 
    UnsupportedFormatError, 
    FileValidationError
)
from scrape.download import download_pdf

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
        DocumentScrapingError: If document scraping fails
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
        downloaded_count = 0
        failed_downloads = 0
        if changes['new']:
            logger.info(f"Step 4: Processing {len(changes['new'])} new documents...")
            inserted_count = insert_new_documents(changes['new'])
            logger.info(f"Successfully inserted {inserted_count}/{len(changes['new'])} new documents")
            
            # Step 4.1: Download new documents
            logger.info(f"Step 4.1: Downloading {len(changes['new'])} new documents...")
            downloaded_count, failed_downloads = _download_new_documents(changes['new'], config)
            logger.info(f"Successfully downloaded {downloaded_count}/{len(changes['new'])} new documents")
            if failed_downloads > 0:
                logger.warning(f"Failed to download {failed_downloads} documents")
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
        result = _create_result_summary(
            existing_docs, 
            new_docs, 
            changes, 
            inserted_count, 
            updated_count, 
            downloaded_count,
            failed_downloads
        )
        _log_workflow_summary(result)
        
        logger.info("NDC document scraping workflow completed successfully!")
        return result
        
    except (DocumentScrapingError, WorkflowError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scraping workflow: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise WorkflowError(f"Workflow execution failed: {str(e)}") from e


def _download_new_documents(new_docs, config: ScrapingConfig) -> tuple:
    """
    Download new documents.
    
    Args:
        new_docs: List of new documents to download
        config: Scraping configuration
        
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    # Create output directory if it doesn't exist
    project_root = Path(__file__).resolve().parents[3]
    output_dir = os.path.join(project_root, "data", "pdfs")
    os.makedirs(output_dir, exist_ok=True)
    
    successful_downloads = 0
    failed_downloads = 0
    
    for doc in new_docs:
        try:
            logger.info(f"Downloading document: {doc.country} - {doc.title} from {doc.url}")
            output_path = download_pdf(
                url=doc.url,
                output_dir=output_dir,
                force_download=False,
                max_retries=config.max_retries,
                timeout=config.timeout,
                country=doc.country,
                language=doc.language,
                submission_date=doc.submission_date
            )
            
            successful_downloads += 1
            logger.info(f"Successfully downloaded document to {output_path}")
            
        except UnsupportedFormatError as e:
            failed_downloads += 1
            logger.error(f"Unsupported format for {doc.url}: {str(e)}")
            
        except FileValidationError as e:
            failed_downloads += 1
            logger.error(f"File validation failed for {doc.url}: {str(e)}")
            
        except DocumentDownloadError as e:
            failed_downloads += 1
            logger.error(f"Failed to download {doc.url}: {str(e)}")
            
        except Exception as e:
            failed_downloads += 1
            logger.error(f"Unexpected error downloading {doc.url}: {str(e)}")
    
    return successful_downloads, failed_downloads


def _log_removed_documents(removed_docs, limit: int = 5) -> None:
    """Log details of removed documents."""
    for doc in removed_docs[:limit]:
        logger.info(f"  Removed: {doc.country} - {doc.title} - {doc.url}")
    if len(removed_docs) > limit:
        logger.info(f"  ... and {len(removed_docs) - limit} more")


def _create_result_summary(
    existing_docs, 
    new_docs, 
    changes, 
    inserted_count, 
    updated_count, 
    downloaded_count=0,
    failed_downloads=0
) -> Dict[str, Any]:
    """Create workflow result summary."""
    return {
        'existing_count': len(existing_docs),
        'scraped_count': len(new_docs),
        'new_count': len(changes.get('new', [])),
        'updated_count': len(changes.get('updated', [])),
        'removed_count': len(changes.get('removed', [])),
        'inserted_count': inserted_count,
        'downloaded_count': downloaded_count,
        'failed_downloads': failed_downloads,
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
    logger.info(f"  New documents downloaded: {result.get('downloaded_count', 0)}")
    if result.get('failed_downloads', 0) > 0:
        logger.info(f"  Failed downloads: {result.get('failed_downloads', 0)}")
    logger.info(f"  Documents updated: {result['updated_actual_count']}")
    logger.info(f"  Documents removed (not processed): {result['removed_count']}")
    logger.info("=" * 60) 