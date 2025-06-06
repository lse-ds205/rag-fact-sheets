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

class RetrievalError(Exception):
    """Base exception for retrieval operations."""
    pass


class VectorSearchError(RetrievalError):
    """Raised when vector search operations fail."""
    pass


class IndexError(RetrievalError):
    """Raised when vector index operations fail."""
    pass


class QueryProcessingError(RetrievalError):
    """Raised when query processing fails."""
    pass


class ResultRankingError(RetrievalError):
    """Raised when result ranking or filtering fails."""
    pass


class SimilaritySearchError(RetrievalError):
    """Raised when similarity search operations fail."""
    pass


class RetrievalDatabaseError(RetrievalError):
    """Raised when database operations during retrieval fail."""
    pass


class NoResultsFoundError(RetrievalError):
    """Raised when no relevant results are found for a query."""
    pass



"""LLM module"""

class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMModelLoadError(LLMError):
    """Raised when an LLM model fails to load."""
    pass


class LLMGenerationError(LLMError):
    """Raised when text generation fails."""
    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail."""
    pass


class TokenLimitExceededError(LLMError):
    """Raised when input exceeds model's token limit."""
    pass


class LLMResponseValidationError(LLMError):
    """Raised when LLM response validation fails."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid."""
    pass


class PromptTemplateError(LLMError):
    """Raised when prompt template processing fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM operations timeout."""
    pass


class InsufficientContextError(LLMError):
    """Raised when there's insufficient context for generation."""
    pass



