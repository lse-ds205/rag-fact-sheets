"""Custom exceptions for the scraping module."""


class ScrapeError(Exception):
    """Base exception for scraping operations."""
    pass


class DatabaseConnectionError(ScrapeError):
    """Raised when database connection fails."""
    pass


class DocumentScrapingError(ScrapeError):
    """Raised when document scraping fails."""
    pass


class DocumentValidationError(ScrapeError):
    """Raised when document validation fails."""
    pass


class WorkflowError(ScrapeError):
    """Raised when workflow execution fails."""
    pass


class DocumentDownloadError(ScrapeError):
    """Raised when document download fails."""
    pass


class UnsupportedFormatError(DocumentDownloadError):
    """Raised when document format is not supported."""
    pass


class FileValidationError(DocumentDownloadError):
    """Raised when file validation fails."""
    pass 