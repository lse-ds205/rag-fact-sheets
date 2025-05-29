import logging
from pathlib import Path
import sys
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
from dateutil.parser import parse
from typing import List, Dict, Set, Optional
import uuid

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from constants.settings import DOCS_URL, HEADERS, COOKIES
from schema import NDCDocumentModel
from helpers.internal import Logger, Test, TaskInfo

logger = logging.getLogger(__name__)

class SeleniumDetector:
    """
    Selenium-based Detector class. Overall goal is to detect if there's a change in the website.
    """
    def __init__(self, headless: bool = True, timeout: int = 30):
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.driver = None
        self.headless = headless
        self._setup_driver()
    
    def _setup_driver(self):
        """Initialize the Chrome WebDriver with appropriate options."""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            if 'User-Agent' in HEADERS:
                chrome_options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(self.timeout)
            
            self.logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {str(e)}")
            raise
    
    def _add_cookies(self):
        """Add cookies from settings to the driver."""
        try:
            self.driver.get(DOCS_URL)
            
            for name, value in COOKIES.items():
                try:
                    self.driver.add_cookie({
                        'name': name,
                        'value': value,
                        'domain': '.unfccc.int'
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to add cookie {name}: {str(e)}")
            
            self.logger.info(f"Added {len(COOKIES)} cookies to driver")
            
        except Exception as e:
            self.logger.error(f"Failed to add cookies: {str(e)}")
    
    def _load_page(self):
        """Load the NDC registry page with proper error handling."""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Loading page {DOCS_URL} (attempt {attempt + 1}/{max_retries})")
                
                self._add_cookies()
                
                self.driver.get(DOCS_URL)
                
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.XPATH, '//table[contains(@class, "table-hover")]'))
                )
                
                self.logger.info("Page loaded successfully")
                return True
                
            except TimeoutException:
                self.logger.warning(f"Timeout loading page on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Error loading page on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
        
        return False
    
    def extract_all_containers(self):
        """
        Extract all selector containers from the website using Selenium.
        
        Returns:
            List of WebElement objects representing rows in the table
        """
        try:
            self.logger.info(f"Extracting all containers from {DOCS_URL}")
            
            if not self._load_page():
                raise Exception("Failed to load page")
            
            wait = WebDriverWait(self.driver, self.timeout)
            
            wait.until(
                EC.presence_of_element_located((By.XPATH, '//table[contains(@class, "table-hover")]/tbody'))
            )
            
            time.sleep(2)
            
            rows = self.driver.find_elements(
                By.XPATH, 
                '//table[contains(@class, "table-hover")]/tbody//tr[contains(@class, "submission")]'
            )
            
            self.logger.info(f"Successfully extracted {len(rows)} containers")
            return rows
            
        except Exception as e:
            self.logger.error(f"Error extracting containers: {str(e)}")
            raise
    
    def extract_all_pdf_links(self, containers: List) -> List[str]:
        """
        Extract all PDF links from the given list of WebElement containers.
        
        Args:
            containers: List of WebElement objects representing rows in the table
        
        Returns:
            List of PDF document URLs
        """
        self.logger.info("Extracting all PDF links from provided containers")
        
        all_links = []
        for i, row in enumerate(containers):
            try:
                link_elements = row.find_elements(By.XPATH, './/td[2]//a[@href]')
                
                for link_element in link_elements:
                    href = link_element.get_attribute('href')
                    if href and href.endswith('.pdf'):
                        all_links.append(href)
                
            except Exception as e:
                self.logger.warning(f"Error extracting links from row {i}: {str(e)}")
                continue
        
        self.logger.info(f"Found {len(all_links)} PDF links")
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
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("WebDriver closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing WebDriver: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

class SeleniumDocUpdater:
    @staticmethod
    def extract_metadata_from_containers(containers) -> List[NDCDocumentModel]:
        """
        Extract metadata from WebElement containers and create NDCDocumentBase objects.
        
        Args:
            containers: List of WebElement objects from the NDC registry page
            
        Returns:
            List of NDCDocumentBase objects with metadata
        """
        logger.info("Extracting metadata from containers")
        documents = []
        
        for i, row in enumerate(containers):
            try:
                cols = row.find_elements(By.XPATH, './td')
                
                if len(cols) < 7:
                    logger.warning(f"Row {i} has insufficient columns ({len(cols)}), skipping")
                    continue
                
                try:
                    country = cols[0].text.strip()
                except:
                    country = ""
                
                try:
                    lang_elements = cols[1].find_elements(By.XPATH, './/span')
                    langs = [elem.text.strip() for elem in lang_elements if elem.text.strip()]
                    
                    doc_elements = cols[1].find_elements(By.XPATH, './/a[@href]')
                    docs = []
                    doc_titles = []
                    
                    for doc_elem in doc_elements:
                        href = doc_elem.get_attribute('href')
                        if href and href.endswith('.pdf'):
                            docs.append(href)
                            doc_titles.append(doc_elem.text.strip())
                    
                except Exception as e:
                    logger.warning(f"Error extracting docs from row {i}: {str(e)}")
                    langs, docs, doc_titles = [], [], []
                
                try:
                    backup_date = cols[6].text.strip()
                except:
                    backup_date = ""
                
                upload_dates = []
                for doc in docs:
                    try:
                        parts = doc.split("/")
                        for part in parts:
                            if len(part) == 7 and part.count('-') == 1:  # Format: YYYY-MM
                                upload_dates.append(part)
                                break
                        else:
                            upload_dates.append("")
                    except:
                        upload_dates.append("")
                
                for j, doc in enumerate(docs):
                    try:
                        try:
                            lang = cols[2].text.strip() if cols[2].text.strip() else None
                        except:
                            lang = langs[j] if j < len(langs) else None
                            
                        doc_title = doc_titles[j] if j < len(doc_titles) else None
                        
                        document = NDCDocumentModel(
                            doc_id=uuid.uuid4(),
                            country=country,
                            title=doc_title,
                            url=doc,
                            language=lang,
                            submission_date=None,
                            file_path=None,
                            file_size=None,
                            scraped_at=datetime.now(),
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                        
                        try: 
                            date = upload_dates[j] if j < len(upload_dates) else ""
                            if date:
                                document.submission_date = parse(date + "-01").date()
                        except: 
                            try:
                                if backup_date:
                                    document.submission_date = parse(backup_date).date()
                            except:
                                pass
                                
                        documents.append(document)
                        
                    except Exception as e:
                        logger.error(f"Error creating document {j} from row {i}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing row {i}: {str(e)}")
                continue
        
        logger.info(f"Extracted metadata for {len(documents)} documents")
        return documents

def scrape_ndc_documents(headless: bool = True, timeout: int = 30) -> List[NDCDocumentModel]:
    """
    Convenience function to scrape NDC documents using Selenium.
    
    Args:
        headless: Whether to run browser in headless mode
        timeout: Timeout in seconds for page operations
        
    Returns:
        List of NDCDocumentModel objects
    """
    with SeleniumDetector(headless=headless, timeout=timeout) as detector:
        containers = detector.extract_all_containers()
        documents = SeleniumDocUpdater.extract_metadata_from_containers(containers)
        return documents

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("Starting Selenium-based NDC document scraping")
        
        documents = scrape_ndc_documents(headless=True, timeout=30)
        
        logger.info(f"Successfully scraped {len(documents)} documents")
        
        for i, doc in enumerate(documents[:5]):
            logger.info(f"Document {i+1}: {doc.country} - {doc.title} - {doc.url}")
        
        with SeleniumDetector(headless=True, timeout=30) as detector:
            containers = detector.extract_all_containers()
            pdf_links = detector.extract_all_pdf_links(containers)
            logger.info(f"Extracted {len(pdf_links)} PDF links")
            
            changes = detector.check_new_links(pdf_links, [])
            logger.info(f"Change detection: {len(changes['new'])} new, {len(changes['current'])} current, {len(changes['old'])} old")
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise 