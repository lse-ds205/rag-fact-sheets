import uuid
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np


@dataclass
class ChunkData:
    """Memory-efficient chunk representation with consistent UUID handling"""
    chunk_id: uuid.UUID
    content: str
    embedding: Optional[np.ndarray] = None
    chunk_index: Optional[int] = None

@dataclass
class RelationshipScore:
    """Relationship detection result with consistent UUID types"""
    source_id: uuid.UUID
    target_id: uuid.UUID
    relationship_type: str
    confidence: float
    evidence: str
    method: str = "hybrid"

@dataclass
class GraphAnalysisResult:
    """Graph analysis result with consistent UUID handling"""
    chunk_id: uuid.UUID
    content: str
    centrality_scores: Dict[str, float]
    community_id: int
    final_score: float

@dataclass
class NodeClassification:
    """Classification result for a single node with UUID consistency"""
    chunk_id: uuid.UUID
    content: str
    centrality_scores: Dict[str, float]
    combined_score: float
    classification: str
    rank_within_class: int
    confidence_level: str