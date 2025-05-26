# Scrape Module Documentation

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [Core Functions](#core-functions)
  - [Database Operations](#database-operations)
  - [Comparison Operations](#comparison-operations)
  - [Download Operations](#download-operations)
  - [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Usage Patterns](#usage-patterns)
- [Testing](#testing)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The `scrape` module provides a robust, production-ready solution for scraping, comparing, and updating NDC (Nationally Determined Contributions) documents from the UNFCCC registry. It follows clean architecture principles with clear separation of concerns, comprehensive error handling, and configurable behavior.

### Key Features

- **Lightweight Entrypoint**: Minimal orchestration layer (25 lines vs 375 lines)
- **Modular Architecture**: Seven focused modules with single responsibilities
- **Custom Exception Hierarchy**: Precise error categorization and handling
- **Configuration Management**: Flexible deployment and runtime configuration
- **Production Ready**: Comprehensive logging, monitoring, and error recovery
- **Integrated Document Download**: Automatic download of new documents with format validation
- **Strict Format Validation**: Only PDF, DOC, and DOCX formats are supported

### Design Principles

- **SOLID Principles**: Single responsibility, open/closed, dependency inversion
- **Clean Architecture**: Clear layer boundaries and dependency direction
- **Fail-Fast**: Early error detection with specific exception types
- **Observability**: Detailed logging and workflow metrics

## Architecture

### Module Structure

```
group4py/src/scrape/
├── __init__.py           # Public API exports
├── workflow.py           # Orchestration layer
├── db_operations.py      # Data access layer
├── comparator.py         # Business logic layer
├── download.py           # Document download functionality
├── selenium.py           # Web scraping layer
├── config.py             # Configuration management
└── exceptions.py         # Custom exception hierarchy
```

### Layer Responsibilities

#### Orchestration Layer (`workflow.py`)
- Coordinates the complete scraping workflow
- Handles high-level error management and recovery
- Provides comprehensive logging and reporting
- Returns structured results for monitoring
- Integrates document downloading for new documents

#### Data Access Layer (`db_operations.py`)
- Encapsulates all database interactions
- Handles data model conversions between layers
- Manages database sessions and transactions
- Provides clean error handling with custom exceptions

#### Business Logic Layer (`comparator.py`)
- Pure comparison logic without side effects
- Document difference detection algorithms
- Easily testable and maintainable
- Focused on domain-specific operations

#### Document Download Layer (`download.py`)
- Handles downloading documents from URLs
- Validates document formats (PDF, DOC, DOCX only)
- Provides robust error handling and retry logic
- Ensures clean up of partial downloads on failure

#### Web Scraping Layer (`selenium.py`)
- Interacts with the UNFCCC website using Selenium
- Extracts document metadata and URLs
- Handles browser automation and page navigation
- Manages browser sessions and cleanup

### Data Flow

```mermaid
graph TD
    A[Entrypoint] --> B[Workflow Orchestrator]
    B --> C[Database Operations]
    B --> D[Selenium Scraper]
    B --> E[Document Comparator]
    B --> F[Document Downloader]
    C --> G[Database]
    D --> H[UNFCCC Website]
    E --> I[Change Detection]
    I --> C
    F --> J[File System]
```

## API Reference

### Core Functions

#### `run_scraping_workflow(config: Optional[ScrapingConfig] = None) -> Dict[str, Any]`

Executes the complete NDC document scraping workflow.

**Parameters:**
- `config` (Optional[ScrapingConfig]): Configuration for scraping operations. Defaults to `DEFAULT_CONFIG`.

**Returns:**
- `Dict[str, Any]`: Workflow results and statistics containing:
  - `existing_count`: Number of documents in database
  - `scraped_count`: Number of documents scraped from website
  - `new_count`: Number of new documents found
  - `updated_count`: Number of documents with changes
  - `removed_count`: Number of documents no longer on website
  - `inserted_count`: Number of documents successfully inserted
  - `downloaded_count`: Number of documents successfully downloaded
  - `failed_downloads`: Number of documents that failed to download
  - `updated_actual_count`: Number of documents successfully updated
  - `success`: Boolean indicating workflow success

**Raises:**
- `WorkflowError`: If workflow execution fails
- `DocumentScrapingError`: If document scraping fails
- `DatabaseConnectionError`: If database connection fails

**Example:**
```python
from group4py.src.scrape import run_scraping_workflow

try:
    result = run_scraping_workflow()
    print(f"Processed {result['inserted_count']} new documents")
    print(f"Downloaded {result['downloaded_count']} new documents")
    print(f"Updated {result['updated_actual_count']} existing documents")
except WorkflowError as e:
    logger.error(f"Workflow failed: {e}")
```

### Database Operations

#### `retrieve_existing_documents() -> List[NDCDocumentModel]`

Retrieves all existing documents from the database.

**Returns:**
- `List[NDCDocumentModel]`: List of documents currently in the database

**Raises:**
- `DatabaseConnectionError`: If database connection or query fails

#### `insert_new_documents(new_docs: List[NDCDocumentBase]) -> int`

Inserts new documents into the database.

**Parameters:**
- `new_docs` (List[NDCDocumentBase]): List of new documents to insert

**Returns:**
- `int`: Number of successfully inserted documents

**Raises:**
- `DatabaseConnectionError`: If database connection fails
- `DocumentValidationError`: If document validation fails

#### `update_existing_documents(updated_docs: List[NDCDocumentBase]) -> int`

Updates existing documents in the database with new metadata.

**Parameters:**
- `updated_docs` (List[NDCDocumentBase]): List of documents with updated metadata

**Returns:**
- `int`: Number of successfully updated documents

**Raises:**
- `DatabaseConnectionError`: If database connection fails

### Comparison Operations

#### `compare_documents(existing_docs: List[NDCDocumentModel], new_docs: List[NDCDocumentBase]) -> Dict[str, List[Union[NDCDocumentModel, NDCDocumentBase]]]`

Compares existing and new documents to find differences.

**Parameters:**
- `existing_docs` (List[NDCDocumentModel]): Documents currently in database
- `new_docs` (List[NDCDocumentBase]): Documents scraped from website

**Returns:**
- `Dict[str, List[Union[NDCDocumentModel, NDCDocumentBase]]]`: Dictionary containing:
  - `new`: Documents found on website but not in database
  - `updated`: Documents with changed metadata
  - `removed`: Documents in database but not on website

### Download Operations

#### `download_pdf(url: str, output_dir: Optional[str] = None, force_download: bool = False, max_retries: int = 3, timeout: int = 30) -> str`

Downloads a document (PDF, DOC, DOCX) from a URL and saves it to the specified folder.

**Parameters:**
- `url` (str): URL of the document to download
- `output_dir` (Optional[str]): Directory to save documents. If None, uses 'data/pdfs' in the project root
- `force_download` (bool): If True, download regardless of content type
- `max_retries` (int): Maximum number of retry attempts for failed downloads
- `timeout` (int): Timeout for HTTP requests in seconds

**Returns:**
- `str`: Path to the downloaded document

**Raises:**
- `DocumentDownloadError`: If the download fails
- `UnsupportedFormatError`: If the document format is not supported (only PDF, DOC, DOCX allowed)
- `FileValidationError`: If the file validation fails

**Example:**
```python
from group4py.src.scrape import download_pdf

try:
    output_path = download_pdf(
        url="https://example.com/document.pdf",
        output_dir="./downloads",
        max_retries=5
    )
    print(f"Document downloaded to {output_path}")
except UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
except DocumentDownloadError as e:
    print(f"Download failed: {e}")
```

### Utility Functions

#### `uuid.uuid5(uuid.NAMESPACE_URL, url) -> UUID`

Generates a deterministic UUID from a URL using the UUID5 algorithm. This creates consistent identifiers for documents based on their source URLs.

**Parameters:**
- `url` (str): PDF URL

**Returns:**
- `UUID`: UUID object generated from the URL

## Configuration

### ScrapingConfig

The `ScrapingConfig` dataclass provides comprehensive configuration options:

```python
@dataclass
class ScrapingConfig:
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
```

### Configuration Examples

#### Development Configuration
```python
from group4py.src.scrape import ScrapingConfig, run_scraping_workflow

dev_config = ScrapingConfig(
    headless=False,           # Show browser for debugging
    timeout=60,               # Longer timeout for slow connections
    log_removed_limit=10,     # Log more removed documents
    abort_on_no_docs=False    # Continue even if no docs scraped
)

result = run_scraping_workflow(dev_config)
```

#### Production Configuration
```python
prod_config = ScrapingConfig(
    headless=True,            # Headless for server deployment
    timeout=30,               # Standard timeout
    max_retries=5,            # More retries for reliability
    retry_delay=10,           # Longer delay between retries
    process_removed_docs=True # Process removed documents
)

result = run_scraping_workflow(prod_config)
```

#### Testing Configuration
```python
test_config = ScrapingConfig(
    headless=True,
    timeout=15,               # Faster timeout for tests
    max_retries=1,            # Single retry for speed
    abort_on_no_docs=False,   # Don't abort in test scenarios
    log_removed_limit=0       # Minimal logging
)
```

## Error Handling

The module implements a comprehensive exception hierarchy for precise error handling:

```
ScrapeError (base)
├── DatabaseConnectionError
├── DocumentScrapingError
├── DocumentValidationError
├── WorkflowError
└── DocumentDownloadError
    ├── UnsupportedFormatError
    └── FileValidationError
```

### Exception Types

#### `ScrapeError`
Base exception for all scraping-related errors.

#### `DatabaseConnectionError`
Raised when database connection or operations fail.

#### `DocumentScrapingError`
Raised when document scraping from the website fails.

#### `DocumentValidationError`
Raised when document validation fails during database operations.

#### `WorkflowError`
Raised when the overall workflow execution fails.

#### `DocumentDownloadError`
Raised when document download fails.

#### `UnsupportedFormatError`
Raised when document format is not supported (only PDF, DOC, DOCX allowed).

#### `FileValidationError`
Raised when file validation fails after download.

### Error Handling Strategy

The module implements a multi-level error handling strategy:

1. **Function-Level Handling**: Each function validates inputs and handles its own specific errors
2. **Module-Level Handling**: The workflow orchestrator catches and logs function-specific errors
3. **Entrypoint-Level Handling**: The entrypoint catches and reports workflow errors

## Usage Patterns

### Basic Usage

The simplest way to use the scrape module is through the `run_scraping_workflow` function:

```python
from group4py.src.scrape import run_scraping_workflow, ScrapingConfig

# Use default configuration
result = run_scraping_workflow()

# Or customize the configuration
config = ScrapingConfig(
    headless=False,  # Show browser window during scraping
    timeout=60,      # Increase timeout for slow connections
    max_retries=5    # More retries for unreliable connections
)
result = run_scraping_workflow(config)

# Access workflow statistics
print(f"Found {result['new_count']} new documents")
print(f"Downloaded {result['downloaded_count']} documents")
print(f"Failed to download {result['failed_downloads']} documents")
```

### Document Download Only

If you need to download documents without the full workflow:

```python
from group4py.src.scrape import download_pdf

try:
    path = download_pdf(
        url="https://unfccc.int/sites/default/files/NDC/2022-06/Rwanda%20NDC_2020.pdf",
        output_dir="./data/documents"
    )
    print(f"Downloaded to {path}")
except Exception as e:
    print(f"Download failed: {e}")
```

## Performance Considerations

### Memory Usage

- Document metadata is stored in lean data structures
- PDF content is streamed during download to minimize memory usage
- Browser sessions are properly closed after use

### Concurrency

- Downloads are currently processed sequentially
- Future versions may implement concurrent downloads with rate limiting
- Consider using process pools for parallel processing of large document sets

### Network Efficiency

- Retry logic handles transient network errors
- Session reuse reduces connection overhead
- Browser automation respects website rate limits

## Troubleshooting

### Common Issues

#### Download Failures

If document downloads are failing:

1. Check URL validity - only PDF, DOC, DOCX formats are supported
2. Verify network connectivity to the document server
3. Increase the timeout value in the configuration
4. Check if the website is implementing anti-scraping measures

#### Browser Automation Issues

If Selenium scraping fails:

1. Ensure Chrome/Chromium and the ChromeDriver are properly installed
2. Try disabling headless mode for debugging
3. Check for browser compatibility issues with the website
4. Increase the timeout value for slow websites

## Contributing

### Adding New Features

When adding new features to the scrape module:

1. Follow the existing architecture patterns
2. Add appropriate exception handling
3. Include comprehensive unit tests
4. Update documentation

### Code Style

Follow these guidelines:

1. Use type hints consistently
2. Document all public functions with docstrings
3. Follow PEP 8 style guidelines
4. Keep functions focused on a single responsibility

## Testing

### Unit Testing

#### Testing Database Operations
```python
import pytest
from unittest.mock import Mock, patch
from group4py.src.scrape.db_operations import retrieve_existing_documents
from group4py.src.scrape.exceptions import DatabaseConnectionError

class TestDatabaseOperations:
    
    @patch('group4py.src.scrape.db_operations.Connection')
    def test_retrieve_existing_documents_success(self, mock_connection):
        """Test successful document retrieval."""
        # Setup
        mock_db = Mock()
        mock_session = Mock()
        mock_connection.return_value = mock_db
        mock_db.connect.return_value = True
        mock_db.get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = []
        
        # Execute
        result = retrieve_existing_documents()
        
        # Assert
        assert isinstance(result, list)
        mock_db.connect.assert_called_once()
        mock_session.close.assert_called_once()
    
    @patch('group4py.src.scrape.db_operations.Connection')
    def test_retrieve_existing_documents_connection_failure(self, mock_connection):
        """Test database connection failure."""
        # Setup
        mock_db = Mock()
        mock_connection.return_value = mock_db
        mock_db.connect.return_value = False
        
        # Execute & Assert
        with pytest.raises(DatabaseConnectionError):
            retrieve_existing_documents()
```

#### Testing Workflow Logic
```python
import pytest
from unittest.mock import Mock, patch
from group4py.src.scrape.workflow import run_scraping_workflow
from group4py.src.scrape.config import ScrapingConfig
from group4py.src.scrape.exceptions import DocumentScrapingError

class TestWorkflow:
    
    @patch('group4py.src.scrape.workflow.retrieve_existing_documents')
    @patch('group4py.src.scrape.workflow.scrape_ndc_documents')
    @patch('group4py.src.scrape.workflow.compare_documents')
    @patch('group4py.src.scrape.workflow.insert_new_documents')
    def test_workflow_success(self, mock_insert, mock_compare, mock_scrape, mock_retrieve):
        """Test successful workflow execution."""
        # Setup
        mock_retrieve.return_value = []
        mock_scrape.return_value = [Mock()]
        mock_compare.return_value = {'new': [Mock()], 'updated': [], 'removed': []}
        mock_insert.return_value = 1
        
        config = ScrapingConfig()
        
        # Execute
        result = run_scraping_workflow(config)
        
        # Assert
        assert result['success'] is True
        assert result['inserted_count'] == 1
        mock_retrieve.assert_called_once()
        mock_scrape.assert_called_once_with(headless=True, timeout=30)
    
    @patch('group4py.src.scrape.workflow.retrieve_existing_documents')
    @patch('group4py.src.scrape.workflow.scrape_ndc_documents')
    def test_workflow_scraping_failure(self, mock_scrape, mock_retrieve):
        """Test workflow failure during scraping."""
        # Setup
        mock_retrieve.return_value = []
        mock_scrape.side_effect = Exception("Scraping failed")
        
        # Execute & Assert
        with pytest.raises(DocumentScrapingError):
            run_scraping_workflow()
```

### Integration Testing

#### End-to-End Testing
```python
import pytest
from group4py.src.scrape import run_scraping_workflow, ScrapingConfig

class TestIntegration:
    
    @pytest.mark.integration
    def test_full_workflow_integration(self):
        """Test complete workflow integration."""
        config = ScrapingConfig(
            headless=True,
            timeout=15,
            abort_on_no_docs=False
        )
        
        try:
            result = run_scraping_workflow(config)
            
            # Verify result structure
            assert 'success' in result
            assert 'existing_count' in result
            assert 'scraped_count' in result
            assert 'inserted_count' in result
            assert 'updated_actual_count' in result
            
            # Verify data types
            assert isinstance(result['existing_count'], int)
            assert isinstance(result['scraped_count'], int)
            assert isinstance(result['success'], bool)
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
```

### Test Configuration

#### pytest.ini
```ini
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests that may take several minutes
    
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test coverage configuration
addopts = 
    --cov=group4py.src.scrape
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
```

## Performance Considerations

### Optimization Strategies

#### Database Optimization
- **Connection Pooling**: Reuse database connections for multiple operations
- **Batch Operations**: Process documents in batches to reduce database round trips
- **Indexing**: Ensure proper indexing on URL and doc_id fields
- **Query Optimization**: Use efficient queries for document retrieval and updates

#### Selenium Optimization
- **Headless Mode**: Use headless browser for production deployments
- **Resource Management**: Properly close WebDriver instances
- **Timeout Configuration**: Set appropriate timeouts for different environments
- **Anti-Detection**: Implement proper delays and user agent rotation

#### Memory Management
- **Streaming Processing**: Process large document sets in chunks
- **Garbage Collection**: Explicitly clean up large objects
- **Memory Monitoring**: Monitor memory usage during long-running operations

### Performance Monitoring

#### Metrics Collection
```python
import time
import psutil
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor performance metrics during scraping operations."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            return {}
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration_seconds': end_time - self.start_time,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_delta_mb': end_memory - self.start_memory,
            'cpu_percent': psutil.cpu_percent()
        }

# Usage example
def monitored_workflow():
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        result = run_scraping_workflow()
        metrics = monitor.stop_monitoring()
        
        logger.info(f"Performance metrics: {metrics}")
        return result, metrics
        
    except Exception as e:
        metrics = monitor.stop_monitoring()
        logger.error(f"Workflow failed. Performance metrics: {metrics}")
        raise
```

### Scalability Considerations

#### Horizontal Scaling
- **Distributed Processing**: Split document processing across multiple workers
- **Queue-Based Architecture**: Use message queues for asynchronous processing
- **Load Balancing**: Distribute scraping load across multiple instances

#### Vertical Scaling
- **Resource Allocation**: Optimize CPU and memory allocation
- **Concurrent Processing**: Use threading or async processing where appropriate
- **Database Scaling**: Scale database resources based on document volume

## Troubleshooting

### Common Issues

#### Database Connection Issues

**Symptoms:**
- `DatabaseConnectionError` exceptions
- Timeout errors during database operations
- Connection pool exhaustion

**Solutions:**
```python
# Check database connectivity
from group4py.src.database import Connection

def test_database_connection():
    """Test database connectivity."""
    try:
        db = Connection()
        if db.connect():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

# Verify environment variables
import os
print(f"DATABASE_URL: {os.environ.get('DATABASE_URL', 'Not set')}")
```

#### Selenium WebDriver Issues

**Symptoms:**
- `DocumentScrapingError` exceptions
- Browser crashes or hangs
- Element not found errors

**Solutions:**
```python
# Test Selenium setup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def test_selenium_setup():
    """Test Selenium WebDriver setup."""
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        print("✅ Selenium WebDriver working")
        driver.quit()
        return True
    except Exception as e:
        print(f"❌ Selenium WebDriver error: {e}")
        return False

# Update ChromeDriver
from webdriver_manager.chrome import ChromeDriverManager
ChromeDriverManager().install()
```

#### Memory Issues

**Symptoms:**
- Out of memory errors
- Slow performance with large document sets
- System resource exhaustion

**Solutions:**
```python
# Monitor memory usage
import psutil

def check_system_resources():
    """Check available system resources."""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    print(f"Memory: {memory.percent}% used ({memory.available / 1024 / 1024:.0f} MB available)")
    print(f"CPU: {cpu}% used")
    
    if memory.percent > 90:
        print("⚠️  High memory usage detected")
    if cpu > 90:
        print("⚠️  High CPU usage detected")

# Process documents in smaller batches
config = ScrapingConfig(batch_size=100)  # Process 100 documents at a time
```

### Debugging Tools

#### Logging Configuration
```python
import logging

def setup_debug_logging():
    """Setup detailed logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Enable SQL logging
    logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    
    # Enable Selenium logging
    logging.getLogger('selenium').setLevel(logging.DEBUG)
```

#### Diagnostic Functions
```python
def run_diagnostics():
    """Run comprehensive system diagnostics."""
    print("🔍 Running system diagnostics...")
    
    # Test database connection
    db_status = test_database_connection()
    
    # Test Selenium setup
    selenium_status = test_selenium_setup()
    
    # Check system resources
    check_system_resources()
    
    # Test configuration
    from group4py.src.scrape import DEFAULT_CONFIG
    print(f"Default config: {DEFAULT_CONFIG}")
    
    # Summary
    print("\n📊 Diagnostic Summary:")
    print(f"Database: {'✅' if db_status else '❌'}")
    print(f"Selenium: {'✅' if selenium_status else '❌'}")
    
    return db_status and selenium_status
```

## Contributing

### Development Setup

#### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd rag-fact-sheets-4

# Create virtual environment
python -m venv group4venv
source group4venv/bin/activate  # On Windows: group4venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Code Quality Standards

#### Linting and Formatting
```bash
# Run black formatter
black group4py/src/scrape/

# Run isort for import sorting
isort group4py/src/scrape/

# Run flake8 for linting
flake8 group4py/src/scrape/

# Run mypy for type checking
mypy group4py/src/scrape/
```

#### Testing Requirements
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/ -m integration

# Run all tests with coverage
pytest --cov=group4py.src.scrape --cov-report=html
```

### Contribution Guidelines

#### Code Standards
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: All public functions must have comprehensive docstrings
- **Error Handling**: Use custom exceptions with proper error chaining
- **Logging**: Include appropriate logging at INFO, WARNING, and ERROR levels
- **Testing**: Maintain 90%+ test coverage for new code

#### Pull Request Process
1. **Feature Branch**: Create feature branch from `main`
2. **Implementation**: Implement changes following code standards
3. **Testing**: Add comprehensive tests for new functionality
4. **Documentation**: Update documentation for API changes
5. **Review**: Submit pull request with detailed description
6. **CI/CD**: Ensure all automated checks pass

#### Architecture Decisions
- **Single Responsibility**: Each module should have one clear purpose
- **Dependency Injection**: Use dependency injection for testability
- **Error Boundaries**: Implement clear error boundaries between layers
- **Configuration**: Make behavior configurable rather than hardcoded
- **Observability**: Include comprehensive logging and metrics

### Release Process

#### Version Management
- **Semantic Versioning**: Follow semver (MAJOR.MINOR.PATCH)
- **Changelog**: Maintain detailed changelog for each release
- **Migration Guides**: Provide migration guides for breaking changes

#### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Security scan completed
- [ ] Backward compatibility verified
- [ ] Release notes prepared

---

## Appendix

### Related Documentation
- [Database Schema Documentation](./database.md)
- [Selenium Scraper Documentation](./selenium.md)
- [API Reference](./api.md)
- [Deployment Guide](./deployment.md)

### External Resources
- [UNFCCC NDC Registry](https://unfccc.int/NDCREG)
- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

### License
This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

*Last updated: 2024-01-XX*
*Version: 1.0.0* 