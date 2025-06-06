"""
SQLAlchemy models for the climate policy extractor.
"""
import zoneinfo
from datetime import datetime, date

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, Text, Date, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector  # Add this import


from .my_logging import get_logger
from .utils import now_london_time

Base = declarative_base()
logger = get_logger(__name__)

class CountryModel(Base):
    """Represents a country and its main page URL."""
    __tablename__ = 'countries_v2'

    doc_id = Column(String, primary_key=True) # e.g., "argentina.json" or just "argentina"
    country_name = Column(String, nullable=False) # Full country name
    country_url = Column(String, nullable=False, unique=True) # Main URL, e.g., https://climateactiontracker.org/countries/argentina/
    language = Column(String, default='en')
    created_at = Column(DateTime(timezone=True), default=now_london_time)
    last_scraped_at = Column(DateTime(timezone=True), default=now_london_time, onupdate=now_london_time)

    # Relationship to sections
    sections = relationship("CountryPageSectionModel", back_populates="country", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CountryModel(doc_id={self.doc_id}, country_name={self.country_name})>"

class CountryPageSectionModel(Base):
    """Represents a specific section of a country's page on Climate Action Tracker."""
    __tablename__ = 'country_page_sections_v2'

    id = Column(Integer, primary_key=True, autoincrement=True)
    country_doc_id = Column(String, ForeignKey('countries_v2.doc_id'), nullable=False, index=True)
    section_title = Column(String, nullable=False) # e.g., "Summary", "Targets", "Policies & Action"
    section_url = Column(String, nullable=False, unique=True) # Full URL to the specific section page
    text_content = Column(Text, nullable=True) # The scraped text content of this section
    embedding = Column(Vector(1024), nullable=True) # Embedding of text_content
    language = Column(String, default='en')
    scraped_at = Column(DateTime(timezone=True), default=now_london_time)
    
    # Relationship to parent country
    country = relationship("CountryModel", back_populates="sections")

    def __repr__(self):
        return f"<CountryPageSectionModel(id={self.id}, country_doc_id={self.country_doc_id}, section_title='{self.section_title}')>"

def get_db_session(database_url):
    """Create database session."""
    logger.debug(f"Creating database session for {database_url}")
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()

def init_db(database_url):
    """Initialize database tables."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine) 
