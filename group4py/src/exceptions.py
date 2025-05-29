"""
Custom exceptions for this project.
"""



"""Scraping module"""

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




"""Chunking module"""

class ChunkingError(Exception):
    """Base exception for chunking operations."""
    pass


class TextExtractionError(ChunkingError):
    """Raised when text extraction from a document fails."""
    pass


class PDFExtractionError(TextExtractionError):
    """Raised when extracting text from a PDF file fails."""
    pass


class DocxExtractionError(TextExtractionError):
    """Raised when extracting text from a DOCX file fails."""
    pass


class OCRError(TextExtractionError):
    """Raised when OCR text extraction fails."""
    pass


class ChunkProcessingError(ChunkingError):
    """Raised when processing document chunks fails."""
    pass


class ChunkCleaningError(ChunkingError):
    """Raised when cleaning document chunks fails."""
    pass


class ChunkStorageError(ChunkingError):
    """Raised when storing chunks to the database fails."""
    pass


class DocumentProcessingError(ChunkingError):
    """Raised when general document processing fails."""
    pass




"""Embedding module"""

class EmbeddingError(Exception):
    """Base exception for embedding operations."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when an embedding model fails to load."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Raised when the generation of embeddings fails."""
    pass


class ModelNotLoadedError(EmbeddingError):
    """Raised when attempting to use a model that hasn't been loaded."""
    pass


class InvalidInputError(EmbeddingError):
    """Raised when input data for embedding is invalid or malformed."""
    pass


class RelationshipDetectionError(EmbeddingError):
    """Raised when relationship detection between chunks fails."""
    pass


class GraphProcessingError(EmbeddingError):
    """Raised when graph processing operations fail."""
    pass




"""Retrieval module"""




"""LLM module"""



