# The Nece
import os
import time
import random
import requests
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy import create_engine, select, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class for SQLAlchemy models
Base = declarative_base()

# Define the Document model - Adjust this to your table Rui - The Table Name then appears in the "process_downloads" function
class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)
    status = Column(String(20), nullable=False)
    downloaded = Column(Boolean, default=False)
    downloaded_at = Column(DateTime)
    file_size = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

# Database connection
DB_URL = "postgresql://climate:climate@localhost:5432/climate" ## Adjust the connection as necessary
engine = create_engine(DB_URL)


def create_session() -> requests.Session:
    """Create a requests session with retries and browser-like headers."""
    session = requests.Session()
    
    # Configure retries
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/pdf,application/x-pdf,*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'DNT': '1',
    })
    
    return session

def download_file(session: requests.Session, url: str, save_path: str, 
                 pbar: Optional[tqdm] = None, retry_count: int = 0) -> Tuple[bool, Optional[float]]:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        session: Requests session to use
        url: The URL to download from
        save_path: The full path where to save the file
        pbar: Optional progress bar to update during download
        retry_count: Current retry attempt (for internal use)
        
    Returns:
        tuple: (success: bool, file_size: Optional[float])
    """
    try:
        time.sleep(random.uniform(2, 5))
        
        head_response = session.head(url, allow_redirects=True)
        content_type = head_response.headers.get('Content-Type', '').lower()
        
        if 'pdf' not in content_type and retry_count < 3:
            logger.warning(f"Unexpected content type {content_type}, retrying...")
            return download_file(session, url, save_path, pbar, retry_count + 1)
        
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            
            if pbar:
                pbar.set_description(f"Downloading {Path(save_path).name}")
            
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            
            size_mb = Path(save_path).stat().st_size / (1024 * 1024)
            
            with open(save_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'%PDF'):
                    if retry_count < 3:
                        logger.warning(f"Invalid PDF file, retrying... (attempt {retry_count + 1})")
                        os.remove(save_path)
                        time.sleep(random.uniform(5, 10))
                        return download_file(session, url, save_path, pbar, retry_count + 1)
                    else:
                        logger.error(f"Downloaded file is not a valid PDF after {retry_count} retries")
                        os.remove(save_path)
                        return False, None
                
            return True, size_mb
            
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        if retry_count < 3:
            logger.warning(f"Retrying download... (attempt {retry_count + 1})")
            time.sleep(random.uniform(5, 10))
            return download_file(session, url, save_path, pbar, retry_count + 1)
        return False, None

def process_downloads(db_url: str, download_dir: str = "data", limit: Optional[int] = None) -> Tuple[int, int]:
    """
    Process all active documents from the PostgreSQL database.
    
    Args:
        db_url: PostgreSQL database URL (e.g., 'postgresql://user:password@localhost:5432/dbname')
        download_dir: Directory to save downloaded files (default: "data")
        limit: Optional limit on number of documents to process
        
    Returns:
        tuple: (num_processed: int, num_successful: int)
    """
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Create database engine
    engine = create_engine(db_url)
    
    # Create a requests session for all downloads
    http_session = create_session()
    
    with Session(engine) as session:
        # Query for active documents
        query = select(Document).where( #Change the table name as you see fit here
            Document.status == 'active',
            Document.downloaded == False
        )
        
        active_docs = session.execute(query).scalars().all()
        if limit:
            active_docs = active_docs[:limit]
        
        successful = 0
        
        logger.info(f"Found {len(active_docs)} active documents to download")
        
        with tqdm(total=len(active_docs), desc="Overall progress") as pbar:
            for doc in active_docs:
                # Generate save path using a unique identifier from the document
                save_path = os.path.join(download_dir, f"{doc.id}.pdf")
                
                logger.debug(f"Downloading {doc.url} to {save_path}")
                success, file_size = download_file(http_session, doc.url, save_path, pbar)
                
                if success:
                    # Update document record
                    doc.downloaded = True
                    doc.downloaded_at = datetime.now()
                    doc.file_size = file_size
                    successful += 1
                    logger.info(f"Successfully downloaded {doc.id} ({file_size:.2f}MB)")
                else:
                    logger.warning(f"Failed to download {doc.id}")
                
                session.commit()
                pbar.update(1)
                
                # Add a delay between documents
                time.sleep(random.uniform(3, 7))
    
    return len(active_docs), successful

# Run the downloader
# python 1_download.py