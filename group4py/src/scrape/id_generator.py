"""Document ID generation utilities."""

import hashlib
import logging
import urllib.parse

logger = logging.getLogger(__name__)


def generate_doc_id(url: str) -> str:
    """
    Generate document ID from PDF URL by extracting the filename.
    
    Args:
        url: PDF URL
        
    Returns:
        Document ID based on PDF filename, with fallback to hash
    """
    try:
        filename = url.split('/')[-1]
        if filename.endswith('.pdf'):
            filename = filename[:-4]
        return urllib.parse.unquote(filename)
    except Exception as e:
        logger.warning(f"Error generating doc_id from URL {url}: {str(e)}")
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"doc_{url_hash}" 