"""
Document Downloader Module for Scraping

This module provides functionality to download PDF, DOC, and DOCX documents from URLs,
handling various edge cases and server restrictions.
"""

import os
import logging
import pathlib
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, List

import requests

from .exceptions import DocumentDownloadError, UnsupportedFormatError, FileValidationError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ['.pdf', '.doc', '.docx']
SUPPORTED_CONTENT_TYPES = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
]


def download_pdf(
    url: str,
    output_dir: Optional[str] = None,
    force_download: bool = False,
    max_retries: int = 3,
    timeout: int = 30
) -> str:
    """
    Download a document (PDF, DOC, DOCX) from a URL and save it to the specified folder.
    
    Args:
        url: URL of the document to download.
        output_dir: Directory to save documents. If None, uses 'data/pdfs' in the project root.
        force_download: If True, download regardless of content type.
        max_retries: Maximum number of retry attempts for failed downloads.
        timeout: Timeout for HTTP requests in seconds.
        
    Returns:
        Path to the downloaded document.
        
    Raises:
        DocumentDownloadError: If the download fails.
        UnsupportedFormatError: If the document format is not supported.
        FileValidationError: If the file validation fails.
    """

    if not url or not url.startswith(('http://', 'https://')):
        raise DocumentDownloadError(f"Invalid URL: {url}")
    
    if output_dir is None:
        script_dir = pathlib.Path(__file__).parent.parent.parent.parent
        output_dir = os.path.join(script_dir, "data", "pdfs")
    
    logger.info(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    parsed_url = urlparse(url)
    filename = unquote(os.path.basename(parsed_url.path))
    
    file_extension = os.path.splitext(filename.lower())[1]
    
    if not filename or file_extension not in SUPPORTED_EXTENSIONS:
        if url.lower().endswith('.pdf'):
            file_extension = '.pdf'
        elif url.lower().endswith('.doc'):
            file_extension = '.doc'
        elif url.lower().endswith('.docx'):
            file_extension = '.docx'
        else:
            raise UnsupportedFormatError(f"URL doesn't end with a supported extension: {url}")
            
        logger.warning(f"URL doesn't have a valid filename with extension, using URL extension: {url}")
        filename = f"document_{hash(url) % 10000}{file_extension}"
    
    output_path = os.path.join(output_dir, filename)
    
    return _download_with_requests(url, output_path, force_download, max_retries, timeout)


def _download_with_requests(
    url: str,
    output_path: str,
    force_download: bool = False,
    max_retries: int = 3,
    timeout: int = 30
) -> str:
    """
    Download a document file using the requests library.
    
    Args:
        url: URL of the document to download.
        output_path: Path to save the file.
        force_download: If True, download regardless of content type.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.
        
    Returns:
        Path to the downloaded file if successful.
        
    Raises:
        DocumentDownloadError: If the download fails.
        UnsupportedFormatError: If the document format is not supported.
        FileValidationError: If the file validation fails.
    """

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
    
    if '://' in url:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        headers['Referer'] = base_url
    
    try:
        logger.info(f"Downloading document from {url}")
        
        session = requests.Session()
        
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
                    raise DocumentDownloadError(f"Failed to download after {max_retries} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying immediately.")
        
        if response.url != url:
            logger.info(f"Redirected to: {response.url}")
        
        has_valid_extension = any(url.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
        
        content_type = response.headers.get('Content-Type', '').lower()
        valid_content_type = any(vct in content_type for vct in SUPPORTED_CONTENT_TYPES)
        
        if not valid_content_type and not force_download and not has_valid_extension:
            raise UnsupportedFormatError(
                f"Content is not a supported document type (PDF, DOC, DOCX). Content-Type: {content_type}"
            )
            
        if not valid_content_type and has_valid_extension:
            logger.warning(f"URL has document extension but Content-Type is: {content_type}. "
                          "Proceeding with download anyway.")
        
        try:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except OSError as e:
            raise DocumentDownloadError(f"Error writing file to {output_path}: {str(e)}")
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FileValidationError("Downloaded file is empty or does not exist")
            
        file_ext = os.path.splitext(output_path)[1].lower()
        if file_ext == '.pdf' and not _is_pdf_file(output_path):
            raise FileValidationError("Downloaded file does not appear to be a valid PDF")
        
        logger.info(f"Document successfully downloaded to {output_path}")
        return output_path
        
    except (DocumentDownloadError, UnsupportedFormatError, FileValidationError):
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    except requests.exceptions.RequestException as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise DocumentDownloadError(f"Error downloading document: {str(e)}")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise DocumentDownloadError(f"Unexpected error: {str(e)}")


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