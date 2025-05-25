"""
Document Downloader Module

This module provides functionality to download PDF, DOC, and DOCX documents from URLs,
handling various edge cases and server restrictions.
"""

import os
import logging
import pathlib
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, List

import requests


logger = logging.getLogger("document_downloader")

def download_pdf(
    url: str,
    output_dir: Optional[str] = None,
    force_download: bool = False,
    max_retries: int = 3,
    timeout: int = 30
) -> Optional[str]:
    """
    Download a document (PDF, DOC, DOCX) from a URL and save it to the specified folder.
    
    Args:
        url: URL of the document to download.
        output_dir: Directory to save documents. If None, uses 'data/pdfs' in the project root.
        force_download: If True, download regardless of content type.
        max_retries: Maximum number of retry attempts for failed downloads.
        timeout: Timeout for HTTP requests in seconds.
        
    Returns:
        Path to the downloaded document if successful, None otherwise.
    """

    if not url or not url.startswith(('http://', 'https://')):
        logger.error(f"Invalid URL: {url}")
        return None
    
    if output_dir is None:
        script_dir = pathlib.Path(__file__).parent.parent.parent
        output_dir = os.path.join(script_dir, "data", "pdfs")
    
    logger.info(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    parsed_url = urlparse(url)
    filename = unquote(os.path.basename(parsed_url.path))
    
    supported_extensions = ['.pdf', '.doc', '.docx']
    file_extension = os.path.splitext(filename.lower())[1]
    
    # If filename is empty or doesn't have a supported extension, create a default name
    if not filename or file_extension not in supported_extensions:
        # Try to guess extension from URL if possible
        if url.lower().endswith('.pdf'):
            file_extension = '.pdf'
        elif url.lower().endswith('.doc'):
            file_extension = '.doc'
        elif url.lower().endswith('.docx'):
            file_extension = '.docx'
        else:
            # Default to PDF if unknown
            file_extension = '.pdf'
            
        logger.warning(f"URL doesn't end with a supported extension: {url}")
        filename = f"document_{hash(url) % 10000}{file_extension}"
    
    # Full path to save the file
    output_path = os.path.join(output_dir, filename)
    
    # Download the document
    return _download_with_requests(url, output_path, force_download, max_retries, timeout)


def _download_with_requests(
    url: str,
    output_path: str,
    force_download: bool = False,
    max_retries: int = 3,
    timeout: int = 30
) -> Optional[str]:
    """
    Download a document file using the requests library.
    
    Args:
        url: URL of the document to download.
        output_path: Path to save the file.
        force_download: If True, download regardless of content type.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.
        
    Returns:
        Path to the downloaded file if successful, None otherwise.
    """
    # Use realistic browser headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;'
                  'q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    # Add referer if URL has a domain
    if '://' in url:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        headers['Referer'] = base_url
    
    try:
        logger.info(f"Downloading document from {url}")
        
        # Use a session to maintain cookies and connection
        session = requests.Session()
        
        # Try with retries
        for attempt in range(max_retries):
            try:
                response = session.get(
                    url, 
                    headers=headers, 
                    stream=True, 
                    timeout=timeout, 
                    allow_redirects=True
                )
                response.raise_for_status()
                break
            except (requests.RequestException, requests.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying immediately.")
        
        # If we get redirected, log the final URL
        if response.url != url:
            logger.info(f"Redirected to: {response.url}")
        
        # Expected content types for documents - ONLY PDF, DOC, DOCX
        valid_content_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        
        # Check if URL ends with a valid extension - ONLY PDF, DOC, DOCX
        has_valid_extension = any(url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx'])
        
        # Check content type, but don't fail if URL ends with valid extension or force_download is True
        content_type = response.headers.get('Content-Type', '').lower()
        valid_content_type = any(vct in content_type for vct in valid_content_types)
        
        if not valid_content_type and not force_download and not has_valid_extension:
            logger.error(f"Content is not a supported document type (PDF, DOC, DOCX). Content-Type: {content_type}")
            return None
            
        # If content type doesn't match but URL has valid extension, just log a warning
        if not valid_content_type and has_valid_extension:
            logger.warning(f"URL has document extension but Content-Type is: {content_type}. "
                          "Proceeding with download anyway.")
        
        # Save the document with proper error handling
        try:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except OSError as e:
            logger.error(f"Error writing file to {output_path}: {e}")
            return None
        
        # Verify the file was downloaded and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("Downloaded file is empty or does not exist")
            return None
            
        # Simple validation based on file extension
        file_ext = os.path.splitext(output_path)[1].lower()
        if file_ext == '.pdf' and not _is_pdf_file(output_path):
            logger.warning("Downloaded file does not appear to be a valid PDF")
            # Continue anyway, since we trust the URL
        
        logger.info(f"Document successfully downloaded to {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading document: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def _is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF by examining its header.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        True if the file appears to be a PDF, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4).decode('ascii', errors='ignore')
            return header.startswith('%PDF')
    except Exception as e:
        logger.warning(f"Could not verify PDF format: {e}")
        return False
