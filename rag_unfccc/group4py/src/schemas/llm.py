from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


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