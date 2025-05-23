"""Configuration settings for the scraping module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScrapingConfig:
    """Configuration for scraping operations."""
    
    # Selenium settings
    headless: bool = True
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    
    # Logging settings
    log_removed_limit: int = 5
    
    # Database settings
    batch_size: Optional[int] = None
    
    # Workflow settings
    abort_on_no_docs: bool = True
    process_removed_docs: bool = False


# Default configuration instance
DEFAULT_CONFIG = ScrapingConfig() 