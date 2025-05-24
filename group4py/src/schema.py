"""
Pydantic schema models for the climate policy extractor.
"""
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import UUID
from pydantic import BaseModel, Field, validator

class Vector(BaseModel):
    """Pydantic model for vector embeddings."""
    values: List[float] = Field(..., description="Vector embedding values")
    dimension: int = Field(..., description="Dimension of the vector")

class DatabaseConfig(BaseModel):
    """Pydantic model for database configuration."""
    url: str
    create_tables: bool = False
    echo: bool = False
    host: Optional[str] = None
    port: Optional[int] = 5432
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables"""
        import os
        return cls(
            url=os.getenv('DATABASE_URL', ''),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'hoprag_db'),
            username=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

class NDCDocumentModel(BaseModel):
    """Pydantic model for NDC documents with all fields."""
    doc_id: UUID
    country: str
    title: Optional[str] = None
    url: str
    language: Optional[str] = None
    submission_date: Optional[date] = None
    file_path: Optional[str] = None
    file_size: Optional[float] = None
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

class DocChunk(BaseModel):
    """Unified Pydantic model for document chunks - matches doc_chunks table structure"""
    # Primary key
    id: Optional[UUID] = None
    
    # Core chunk data
    doc_id: str = Field(..., description="Foreign key to documents table")
    content: str = Field(..., min_length=10)
    chunk_index: int = Field(..., ge=0)
    paragraph: Optional[int] = None
    language: Optional[str] = None    # Embeddings (original columns)
    transformer_embedding: Optional[List[float]] = None
    word2vec_embedding: Optional[List[float]] = None
    
    # Metadata field (renamed to chunk_data in the database to avoid SQLAlchemy reserved word conflict)
    chunk_data: Optional[Dict[str, Any]] = Field(default_factory=dict)  # JSONB metadata stored in chunk_data column
    
    # HopRAG-specific columns (added by setup script)
    embedding: Optional[List[float]] = None  # Vector embedding for HopRAG (VECTOR type)
    content_hash: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True

class DocChunkUpdate(BaseModel):
    """Pydantic model for updating document chunks."""
    transformer_embedding: Optional[List[float]] = None
    word2vec_embedding: Optional[List[float]] = None
    embedding: Optional[List[float]] = None
    content_hash: Optional[str] = None
    chunk_data: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    """Pydantic model for query results."""
    chunks: List[DocChunk]
    similarity_scores: Optional[List[float]] = None
    total_results: int
    query_time_ms: float

class LogicalRelationship(BaseModel):
    """Logical relationship between chunks for HopRAG"""
    id: Optional[UUID] = None
    source_chunk_id: UUID = Field(...)
    target_chunk_id: UUID = Field(...)
    relationship_type: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Optional[str] = None
    method: str = Field(default="rule_based")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('relationship_type')
    def validate_relationship_type(cls, v):
        allowed = {'SUPPORTS', 'EXPLAINS', 'CONTRADICTS', 'FOLLOWS', 'TEMPORAL_FOLLOWS', 'CAUSES'}
        if v.upper() not in allowed:
            raise ValueError(f'Must be one of {allowed}')
        return v.upper()