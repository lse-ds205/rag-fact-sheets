import json
import os
from itemadapter import ItemAdapter
import os
from datetime import datetime, date
from dotenv import load_dotenv
from itemadapter import ItemAdapter
import requests
import pdfplumber
from pathlib import Path

from scrapy.exceptions import DropItem

from .models import init_db, get_db_session, CountryModel, CountryPageSectionModel
from .items import CountrySectionItem
from .utils import now_london_time
from .utils import generate_word_embeddings, save_word2vec_model
from sentence_transformers import SentenceTransformer


class ClimateTrackerPipeline:
    def __init__(self):
        self.data_dir = os.path.abspath("climate_tracker/data/json")
        os.makedirs(self.data_dir, exist_ok=True)
        self.countries = set()
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        country_slug = adapter.get('country_slug')
        
        # Save JSON version of the item
        json_file = os.path.join(self.data_dir, f"{country_slug}_{adapter.get('section_title').lower().replace(' ', '_')}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dict(item), f, ensure_ascii=False, indent=2)
        
        # Keep track of processed countries
        self.countries.add(country_slug)
        
        return item
    
    def close_spider(self, spider):
        # Create an index of all countries
        index_file = os.path.join(self.data_dir, "country_index.json")
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.countries), f, indent=2)
        
        spider.logger.info(f"Processed {len(self.countries)} countries in total")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber."""
    try:
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return None
            
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ''
                    all_text += text + "\n\n"
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
                    continue
                    
        return all_text
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

def generate_doc_id(item):
    """Generate a document ID from item metadata."""
    country = item.get('country', 'unknown').lower().replace(" ", "_")
    lang = item.get('language', 'unknown').lower().replace(" ", "_")
    try:
        # Ensure we're using just the date part for the ID
        submission_date = item.get('submission_date')
        if isinstance(submission_date, datetime):
            date_str = submission_date.date().strftime('%Y%m%d')
        elif isinstance(submission_date, date):
            date_str = submission_date.strftime('%Y%m%d')
        else:
            date_str = 'unknown_date'
    except:
        date_str = 'unknown_date'
    
    return f"{country}_{lang}_{date_str}"

class PostgreSQLPipeline:
    """Pipeline for storing NDC documents in PostgreSQL."""

    def __init__(self, db_url=None):
        # Load environment variables
        load_dotenv()
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        print("PostgreSQLPipeline initialized")

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def open_spider(self, spider):
        """Initialize database connection when spider opens."""
        self.logger = spider.logger
        print("PostgreSQLPipeline: Opening spider")
        init_db(self.db_url)  # Create tables if they don't exist
        self.session = get_db_session(self.db_url)

    def close_spider(self, spider):
        """Close database connection when spider closes."""
        print("PostgreSQLPipeline: Closing spider")
        self.session.close()

    def process_item(self, item, spider):
        """Process scraped item and store in PostgreSQL."""
        print(f"PostgreSQLPipeline: Processing item for {item.get('country')}")
        adapter = ItemAdapter(item)
        
        # Convert submission_date to date if it's a datetime
        if 'submission_date' in item:
            submission_date = item['submission_date']
            if isinstance(submission_date, datetime):
                item['submission_date'] = submission_date.date()
        
        # Generate doc_id from metadata (same as future file name)
        doc_id = generate_doc_id(item)
        print(f"PostgreSQLPipeline: Generated doc_id: {doc_id}")

        # Create or update document record
        doc = self.session.query(NDCDocumentModel).filter_by(doc_id=doc_id).first()

        if doc:
            print(f"PostgreSQLPipeline: Document found in database: {doc_id}")
            retrieved_doc_as_dict = adapter.asdict()
            
            # Check if any data has changed
            has_changes = False
            changes = []
            
            for key, value in retrieved_doc_as_dict.items():
                if key in ['downloaded_at', 'processed_at', 'scraped_at']:
                    continue

                if hasattr(doc, key):
                    current_value = getattr(doc, key)
                    
                    if current_value != value:
                        changes.append(f"{key}: {current_value} -> {value}")
                        has_changes = True
                        setattr(doc, key, value)
            
            if has_changes:
                doc.scraped_at = now_london_time()
                print(f"PostgreSQLPipeline: Updating document {doc_id} with changes: {', '.join(changes)}")
            else:
                print(f"PostgreSQLPipeline: No changes detected for document {doc_id}")
                raise DropItem(f"No changes detected for document {doc_id}, skipping update")
        else:
            print(f"PostgreSQLPipeline: Creating new document: {doc_id}")
            doc = NDCDocumentModel(
                doc_id=doc_id,
                country=adapter.get('country'),
                title=adapter.get('title'),
                url=adapter.get('url'),
                language=adapter.get('language'),
                submission_date=adapter.get('submission_date'),
                file_path=None,
                file_size=None,
                extracted_text=None,
                chunks=None,
                downloaded_at=None,
                processed_at=None
            )
            try:
                self.session.add(doc)
                print(f"PostgreSQLPipeline: Added new document to session: {doc_id}")
            except Exception as e:
                print(f"PostgreSQLPipeline: Error adding document: {str(e)}")
                raise DropItem(f"Failed to add document to PostgreSQL: {str(e)}")
        
        try:
            self.session.commit()
            print(f"PostgreSQLPipeline: Committed document to database: {doc_id}")
            # Add doc_id back to the item for downstream processing
            item['doc_id'] = doc_id
        except Exception as e:
            self.session.rollback()
            print(f"PostgreSQLPipeline: Error committing document: {str(e)}")
            raise DropItem(f"Failed to store item in PostgreSQL: {e}")
        
        return item

class TextExtractionPipeline:
    """Pipeline to extract text from PDFs."""
    def __init__(self):
        # Import pdfplumber here to avoid dependency issues
        import pdfplumber
        self.pdfplumber = pdfplumber
        
    def process_item(self, item, spider):
        """Extract text from PDF and update the item."""
        file_path = item.get('file_path')
        if not file_path:
            print("TextExtractionPipeline: No file path provided, skipping text extraction")
            return item
            
        print(f"TextExtractionPipeline: Extracting text from {file_path}")
        text = extract_text_from_pdf(file_path)
        
        if text:
            item['extracted_text'] = text
            print(f"TextExtractionPipeline: Successfully extracted {len(text)} characters")
        else:
            print(f"TextExtractionPipeline: Failed to extract text from {file_path}")
            
        return item

class WordEmbeddingPipeline:
    """Pipeline to generate embeddings from extracted text."""
    def __init__(self, model_dir=None):
        # Load environment variables
        load_dotenv()
        self.model_dir = model_dir or os.path.join(os.getenv('DATA_DIR', 'data'), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def process_item(self, item, spider):
        if not item.get('extracted_text'):
            print("WordEmbeddingPipeline: No extracted text, skipping embedding generation")
            return item
            
        try:
            print(f"WordEmbeddingPipeline: Generating embeddings for {item.get('doc_id')}")
            embeddings = generate_word_embeddings(item['extracted_text'])
            
            if embeddings:
                item['word_embeddings'] = embeddings.get('document_vector')
                
                # Save the model if needed
                if item.get('doc_id') and embeddings.get('model'):
                    model_path = save_word2vec_model(
                        embeddings['model'], 
                        item['doc_id'], 
                        self.model_dir
                    )
                    if model_path:
                        print(f"WordEmbeddingPipeline: Model saved to {model_path}")
            else:
                print(f"WordEmbeddingPipeline: No embeddings generated")
        except Exception as e:
            print(f"WordEmbeddingPipeline: Error generating embeddings: {str(e)}")
            
        return item


class ExtractionPipeline:
    """Pipeline to extract structured information from documents."""
    def process_item(self, item, spider):
        # Import here to avoid circular imports
        from .extraction_organised_2 import PolicyExtractor
        
        print(f"ExtractionPipeline: Extracting information for {item.get('doc_id')}")
        try:
            extractor = PolicyExtractor()
            results = extractor.extract_document(item.get('doc_id'))
            
            if results:
                item.update(results)
                print(f"ExtractionPipeline: Successfully extracted information")
            else:
                print(f"ExtractionPipeline: No information extracted")
                
        except Exception as e:
            print(f"ExtractionPipeline: Error extracting information: {str(e)}")
            
        return item
    




    

from sentence_transformers import SentenceTransformer
from datetime import datetime
from .utils import now_london_time


class TransformerPipeline:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
    def process_item(self, item, spider):
        if item.get('content'):
            item['embedding'] = self.model.encode(item['content'])
            item['model_type'] = 'transformer'
            item['processed_at'] = now_london_time()
        return item

class CountryDataPostgreSQLPipeline:
    """Pipeline for storing Climate Action Tracker country and section data in PostgreSQL."""

    def __init__(self, db_url=None):
        # Explicitly load .env from the project root
        # __file__ refers to pipelines.py
        # .parent.parent.parent should navigate to the project root directory
        project_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(dotenv_path=project_root_env)
        self.db_url = db_url or os.getenv('DATABASE_URL')
        if not self.db_url:
            self.logger.error("DATABASE_URL not found. Ensure .env file is in project root and variable is set.")
            raise ValueError("DATABASE_URL not found in environment variables")
        self.logger_set = False

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your pipeline instance
        # You can access crawler.settings here if needed
        return cls(
            db_url=crawler.settings.get('DATABASE_URL') # Allow override from settings
        )

    def open_spider(self, spider):
        """Initialize database connection and session when spider opens."""
        if not hasattr(self, 'logger'): # Ensure logger is set if not done in init
             self.logger = spider.logger
        self.logger.info(f"Opening spider. Connecting to database: {self.db_url}")
        try:
            init_db(self.db_url)  # Create tables if they don't exist based on models.py
            self.session = get_db_session(self.db_url)
            self.logger.info("Database session established.")
        except Exception as e:
            self.logger.error(f"Error connecting to database or initializing tables: {e}")
            raise # Reraise exception to stop spider if DB connection fails

    def close_spider(self, spider):
        """Close database session when spider closes."""
        self.logger.info("Closing spider. Closing database session.")
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def process_item(self, item, spider):
        if not isinstance(item, CountrySectionItem):
            return item # Not for this pipeline

        adapter = ItemAdapter(item)
        self.logger.debug(f"Processing CountrySectionItem for country_doc_id: {adapter.get('country_doc_id')}, section: {adapter.get('section_title')}")

        try:
            # --- 1. Find or Create Country --- 
            country_doc_id = adapter.get('country_doc_id')
            country_name = adapter.get('country_name')
            country_main_url = adapter.get('country_main_url')
            country_language = adapter.get('language', 'en') # Use section language or default

            country_db_entry = self.session.query(CountryModel).filter_by(doc_id=country_doc_id).first()

            if country_db_entry:
                self.logger.debug(f"Found existing country: {country_doc_id}")
                # Update if necessary (e.g., name or main URL changed, or last_scraped_at)
                if country_db_entry.country_name != country_name: 
                    country_db_entry.country_name = country_name
                if country_db_entry.country_url != country_main_url:
                    country_db_entry.country_url = country_main_url
                country_db_entry.last_scraped_at = now_london_time()
            else:
                self.logger.info(f"Creating new country: {country_doc_id} - {country_name}")
                country_db_entry = CountryModel(
                    doc_id=country_doc_id,
                    country_name=country_name,
                    country_url=country_main_url,
                    language=country_language,
                    # created_at and last_scraped_at have defaults
                )
                self.session.add(country_db_entry)
                # We need to flush to get the country_db_entry if it's new and we need its ID, 
                # but FK is based on doc_id, so direct add is fine before section.

            # --- 2. Find or Create Country Page Section --- 
            section_url = adapter.get('section_url')
            section_title = adapter.get('section_title')
            section_text_content = adapter.get('section_text_content')
            section_language = adapter.get('language', 'en')

            section_db_entry = self.session.query(CountryPageSectionModel).filter_by(section_url=section_url).first()

            if section_db_entry:
                self.logger.debug(f"Found existing section: {section_title} for URL {section_url}")
                # Update existing section
                section_db_entry.section_title = section_title # Title might change slightly
                if section_db_entry.text_content != section_text_content:
                    section_db_entry.text_content = section_text_content
                    section_db_entry.embedding = None # Clear embedding if text changes
                section_db_entry.language = section_language
                section_db_entry.scraped_at = now_london_time()
                # country_doc_id should not change for an existing section_url
            else:
                self.logger.info(f"Creating new section: {section_title} for country {country_doc_id} (URL: {section_url})")
                section_db_entry = CountryPageSectionModel(
                    country_doc_id=country_doc_id, # Link to the parent country
                    section_title=section_title,
                    section_url=section_url,
                    text_content=section_text_content,
                    language=section_language,
                    # embedding is None initially
                    # scraped_at has default
                )
                self.session.add(section_db_entry)
            
            self.session.commit()
            self.logger.info(f"Successfully processed and committed section: {section_title} for country {country_doc_id}")

        except Exception as e:
            self.logger.error(f"Error processing item for section {adapter.get('section_title')} (URL: {adapter.get('section_url')}): {e}")
            self.session.rollback()
            raise DropItem(f"Error in PostgreSQL pipeline: {e}")

        return item