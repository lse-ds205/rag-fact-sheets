"""
Pydantic schema models for the climate policy extractor.
"""
from datetime import datetime, date, timezone
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
            # host=os.getenv('DB_HOST', 'localhost'),
            # port=int(os.getenv('DB_PORT', '5432')),
            # database=os.getenv('DB_NAME', 'hoprag_db'),
            # username=os.getenv('DB_USER', 'postgres'),
            # password=os.getenv('DB_PASSWORD', '')
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
    hoprag_embedding: Optional[List[float]] = None  # Vector embedding for HopRAG (VECTOR type)
    content_hash: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    
    class Config:
        from_attributes = True

class DocChunkUpdate(BaseModel):
    """Pydantic model for updating document chunks."""
    transformer_embedding: Optional[List[float]] = None
    word2vec_embedding: Optional[List[float]] = None
    hoprag_embedding: Optional[List[float]] = None
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
    created_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    
    @validator('relationship_type')
    def validate_relationship_type(cls, v):
        allowed = {'SUPPORTS', 'EXPLAINS', 'CONTRADICTS', 'FOLLOWS', 'TEMPORAL_FOLLOWS', 'CAUSES'}
        if v.upper() not in allowed:
            raise ValueError(f'Must be one of {allowed}')
        return v.upper()

# ------------------------------------------------------------------------------------------------
# LLM Response Models
# ------------------------------------------------------------------------------------------------

class LLMAnswerModel(BaseModel):
    """Pydantic model for LLM answer structure. Included in LLMResponseModel."""
    summary: str = Field(..., description="Brief 2-3 sentence summary of the main answer")
    detailed_response: str = Field(..., description="Comprehensive answer to the question with full context and analysis")

class LLMCitationModel(BaseModel):
    """Pydantic model for LLM citation structure. Included in LLMResponseModel."""
    id: int = Field(..., description="Chunk ID from the database")
    doc_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Full chunk content")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    paragraph: Optional[int] = Field(None, description="Paragraph number within the document")
    language: Optional[str] = Field(None, description="Language of the chunk")
    chunk_metadata: Dict[str, Any] = Field(..., description="Metadata associated with the chunk")
    country: str = Field(..., description="Country associated with the document")
    cos_similarity_score: float = Field(..., description="Cosine similarity score between query and chunk")
    how_used: str = Field(..., description="Explanation of how this chunk contributed to the answer")

class LLMMetadataModel(BaseModel):
    """Pydantic model for LLM response metadata. Included in LLMResponseModel."""
    chunks_cited: int = Field(..., description="Number of chunks cited in the response")
    primary_countries: List[str] = Field(..., description="Main countries discussed in the response")

class LLMResponseModel(BaseModel):
    """Complete Pydantic model for LLM response structure. Contains LLM answer, citations, and metadata."""
    question: str = Field(..., description="The original question/prompt that was asked")
    answer: LLMAnswerModel = Field(..., description="The main answer content")
    citations: List[LLMCitationModel] = Field(..., description="List of cited chunks with explanations")
    metadata: LLMMetadataModel = Field(..., description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are Afghanistan's main climate policies?",
                "answer": {
                    "summary": "Afghanistan's NDC focuses on capacity building and climate resilience measures.",
                    "detailed_response": "Based on the provided documents, Afghanistan's climate policies prioritize..."
                },
                "citations": [
                    {
                        "id": 3064,
                        "doc_id": "afghanistan_english_20220601",
                        "content": "Capacity Building Needs",
                        "chunk_index": 58,
                        "paragraph": 10,
                        "language": None,
                        "chunk_metadata": {
                            "element_types": ["Title"],
                            "page_number": 5,
                            "filename": "afghanistan_english_20220601.pdf"
                        },
                        "country": "Afghanistan",
                        "cos_similarity_score": 0.998662,
                        "how_used": "This chunk provided information about capacity building requirements"
                    }
                ],
                "metadata": {
                    "chunks_cited": 5,
                    "primary_countries": ["Afghanistan"]
                }
            }
        }