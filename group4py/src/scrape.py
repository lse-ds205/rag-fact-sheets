import logging
from pathlib import Path
import sys
import requests
from scrapy.selector import Selector
from datetime import datetime
from dateutil.parser import parse
from typing import List, Dict, Set, Optional

project_root = Path(__file__).resolve().parents[2]
TEMPORARY_HTML_FILE = project_root / "group4py/src/constants/hardcoded.html"
sys.path.insert(0, str(project_root))
import group4py
from constants.settings import DOCS_URL
from schema import NDCDocumentBase
from helpers import Logger, Test, TaskInfo

logger = logging.getLogger(__name__)

class Detector:
    """
    Detector class. Overall goal is to detect if there's a change in the website.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_all_containers(self):
        """
        Extract all selector containers from the website.
        
        Returns:
            List of container elements from the NDC registry page
        """
        self.logger.info(f"Extracting all containers from {DOCS_URL}")
        selector = Selector(text=TEMPORARY_HTML_FILE.read_text(encoding="utf-8"))
        rows = selector.xpath('//table[contains(@class, "table-hover")]/tbody//tr[contains(@class, "submission")]')
        
        self.logger.info(f"Successfully extracted {len(rows)} containers")
        return rows

    def extract_all_pdf_links(self, containers: List[Selector]) -> List[str]:
        """
        Extract all PDF links from the given list of selector containers.
        
        Args:
            containers: List of selector elements representing rows in the table
        
        Returns:
            List of PDF document URLs
        """
        logger.info("Extracting all PDF links from provided containers")
        
        all_links = []
        for row in containers:
            docs = row.xpath('.//td[2]//a/@href').getall()
            all_links.extend(docs)
        
        logger.info(f"Found {len(all_links)} PDF links")
        return all_links

    def check_new_links(self, new_pdf_links: List[str], old_pdf_links: List[str]) -> Dict[str, List[str]]:
        """
        Compare new and old PDF links to detect changes.
        
        Args:
            new_pdf_links: List of newly extracted PDF links
            old_pdf_links: List of previously extracted PDF links
            
        Returns:
            Dictionary with 'new', 'current', and 'old' links
        """
        self.logger.info("Checking for new links")
        
        new_set = set(new_pdf_links)
        old_set = set(old_pdf_links)
        
        new_links = list(new_set - old_set)
        old_links = list(old_set - new_set)
        current_links = list(new_set.intersection(old_set))
        
        result = {
            'new': new_links,
            'current': current_links,
            'old': old_links
        }
        
        self.logger.info(f"Found {len(new_links)} new links, {len(current_links)} current links, {len(old_links)} old links")
        return result

class DocUpdater:
    @staticmethod
    def extract_metadata_from_containers(containers) -> List[NDCDocumentBase]:
        """
        Extract metadata from container elements and create NDCDocumentBase objects.
        
        Args:
            containers: List of container elements from the NDC registry page
            
        Returns:
            List of NDCDocumentBase objects with metadata
        """
        logger.info("Extracting metadata from containers")
        documents = []
        
        for row in containers:
            # Extract data from columns
            cols = row.xpath('./td')
            
            # Make sure we have enough columns
            if len(cols) < 7:
                continue
                
            country = cols[0].xpath('normalize-space(.)').get()
            langs = cols[1].xpath('.//span/text()').getall()
            docs = cols[1].xpath('.//a/@href').getall()
            doc_titles = cols[1].xpath('.//a/text()').getall()
            
            # Extract date from URL paths
            upload_dates = [doc.split("/")[-2] for doc in docs]
            backup_date = cols[6].xpath('normalize-space(.)').get()
            
            for i, doc in enumerate(docs):
                try:
                    # Get language and title if available
                    lang = langs[i] if i < len(langs) else None
                    doc_title = doc_titles[i] if i < len(doc_titles) else None
                    
                    # Create NDCDocumentBase object
                    document = NDCDocumentBase(
                        country=country,
                        title=doc_title,
                        url=doc,
                        language=lang,
                        submission_date=None,
                        file_path=None,
                        file_size=None
                    )
                    
                    # Set submission date
                    try: 
                        date = upload_dates[i]
                        document.submission_date = parse(date + "/01").date()
                    except: 
                        try:
                            document.submission_date = parse(backup_date).date()
                        except:
                            pass
                            
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Error creating document: {str(e)}")
        
        logger.info(f"Extracted metadata for {len(documents)} documents")
        return documents