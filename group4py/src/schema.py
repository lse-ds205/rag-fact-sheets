"""
Pydantic schema models for the climate policy extractor.
"""
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field

class Vector(BaseModel):
    """Pydantic model for vector embeddings."""
    values: List[float] = Field(..., description="Vector embedding values")
    dimension: int = Field(..., description="Dimension of the vector")

class NDCDocumentBase(BaseModel):
    """Base Pydantic model for NDC documents."""
    country: str
    title: Optional[str] = None
    url: str
    language: Optional[str] = None
    submission_date: Optional[date] = None
    file_path: Optional[str] = None
    file_size: Optional[float] = None

class NDCDocumentCreate(NDCDocumentBase):
    """Pydantic model for creating new NDC documents."""
    doc_id: str

class NDCDocumentUpdate(BaseModel):
    """Pydantic model for updating NDC documents."""
    downloaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    last_download_attempt: Optional[datetime] = None
    download_error: Optional[str] = None
    download_attempts: Optional[int] = None
    extracted_text: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None

class NDCDocumentModel(NDCDocumentBase):
    """Pydantic model for NDC documents with all fields."""
    doc_id: str
    scraped_at: datetime
    downloaded_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    last_download_attempt: Optional[datetime] = None
    download_error: Optional[str] = None
    download_attempts: int = 0
    extracted_text: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocChunkBase(BaseModel):
    """Base Pydantic model for document chunks."""
    doc_id: str
    content: str
    chunk_index: int
    paragraph: Optional[int] = None
    language: Optional[str] = None

class DocChunkCreate(DocChunkBase):
    """Pydantic model for creating new document chunks."""
    pass

class DocChunkUpdate(BaseModel):
    """Pydantic model for updating document chunks."""
    transformer_embedding: Optional[List[float]] = None
    word2vec_embedding: Optional[List[float]] = None
    chunk_metadata: Optional[Dict[str, Any]] = None

class DocChunkModel(DocChunkBase):
    """Pydantic model for document chunks with all fields."""
    id: int
    transformer_embedding: Optional[Vector] = None
    word2vec_embedding: Optional[Vector] = None
    chunk_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class QueryResult(BaseModel):
    """Pydantic model for query results."""
    chunks: List[DocChunkModel]
    similarity_scores: Optional[List[float]] = None
    total_results: int
    query_time_ms: float

class DatabaseConfig(BaseModel):
    """Pydantic model for database configuration."""
    url: str
    create_tables: bool = False
    echo: bool = False