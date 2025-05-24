"""
HopRAG Graph Processor with Memory-Optimized BFS and Analysis
Implements embeddings, relationship detection, and graph analysis
"""

import asyncio
import numpy as np
import json
import gc
import re
import logging
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text
from database import Connection, Document, DocChunkORM
from schema import DatabaseConfig, LogicalRelationship

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkData:
    """Memory-efficient chunk representation"""
    chunk_id: int
    content: str
    embedding: Optional[np.ndarray] = None

@dataclass
class RelationshipScore:
    """Relationship detection result"""
    source_id: int
    target_id: int
    relationship_type: str
    confidence: float
    evidence: str
    method: str = "hybrid"

@dataclass
class GraphAnalysisResult:
    """Graph analysis result for ranking"""
    chunk_id: int
    content: str
    centrality_scores: Dict[str, float]
    community_id: int
    final_score: float

@dataclass
class NodeClassification:
    """Classification result for a single node"""
    chunk_id: int
    content: str
    centrality_scores: Dict[str, float]
    combined_score: float
    classification: str
    rank_within_class: int
    confidence_level: str

class MemoryOptimizedEmbedder:
    """Memory-efficient embedding generator"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized embedder: {model_name} (dim: {self.embedding_dim})")
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts in batches to manage memory"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            # Force garbage collection every few batches
            if i % (self.batch_size * 10) == 0:
                gc.collect()
        
        return np.vstack(embeddings) if embeddings else np.array([])

class LogicalRelationshipDetector:
    """Memory-efficient relationship detection"""
    
    def __init__(self):
        self.climate_patterns = {
            'SUPPORTS': [
                (r'\d+(\.\d+)?\s*(MtCO2e|%|GW|MW|billion|million)', r'target|goal|commitment|reduce|achieve'),
                (r'invest.*\$?\d+.*billion', r'achieve|implement|deploy|fund'),
                (r'solar|wind|renewable.*\d+', r'emission.*\reduction|target|goal'),
                (r'carbon tax.*\d+', r'revenue|fund|support|finance'),
                (r'efficiency.*\d+.*%', r'reduction|saving|target')
            ],
            'EXPLAINS': [
                (r'NDC.*Nationally Determined Contribution', r'NDC(?!\w)'),
                (r'GHG.*greenhouse gas', r'GHG(?!\w)'),
                (r'carbon tax.*mechanism.*price', r'carbon tax'),
                (r'renewable energy.*includes.*solar.*wind', r'renewable'),
                (r'adaptation.*refers to|means', r'adaptation'),
                (r'mitigation.*refers to|means', r'mitigation')
            ],            'CONTRADICTS': [
                (r'target.*\d+.*MtCO2e', r'target.*\d+.*MtCO2e'),
                (r'by \d{4}', r'by \d{4}'),
                (r'increase.*emissions', r'reduce.*emissions'),
                (r'not.*feasible|impossible', r'will.*implement|committed')
            ],
            'FOLLOWS': [
                (r'phase 1|first phase', r'phase 2|second phase|next phase'),
                (r'by 2030', r'after 2030|post-2030|2035|2040|2050'),
                (r'short.?term', r'medium.?term|long.?term'),
                (r'pilot|trial', r'scale.*up|full.*deployment')
            ],
            'CAUSES': [
                (r'due to|because of|result of', r'therefore|thus|consequently'),
                (r'leads to|results in|causes', r'impact|effect|consequence'),
                (r'if.*then', r'will.*result|outcome')
            ]
        }
    
    def detect_relationships_batch(self, source_chunks: List[ChunkData], 
                                 target_chunks: List[ChunkData]) -> List[RelationshipScore]:
        """Detect relationships in batches for memory efficiency"""
        relationships = []
        
        for source_chunk in source_chunks:
            for target_chunk in target_chunks:
                if source_chunk.chunk_id != target_chunk.chunk_id:
                    relationship = self._detect_single_relationship(source_chunk, target_chunk)
                    if relationship and relationship.confidence > 0.5:
                        relationships.append(relationship)
        
        return relationships
    
    def _detect_single_relationship(self, source: ChunkData, target: ChunkData) -> Optional[RelationshipScore]:
        """Detect relationship between two chunks"""
        best_relationship = None
        best_confidence = 0.0
        
        source_text = source.content.lower()
        target_text = target.content.lower()
        
        for rel_type, patterns in self.climate_patterns.items():
            for source_pattern, target_pattern in patterns:
                if (re.search(source_pattern, source_text, re.IGNORECASE) and 
                    re.search(target_pattern, target_text, re.IGNORECASE)):
                    
                    confidence = self._calculate_confidence(source, target, source_pattern, target_pattern)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_relationship = RelationshipScore(
                            source_id=source.chunk_id,
                            target_id=target.chunk_id,
                            relationship_type=rel_type,
                            confidence=confidence,
                            evidence=f"Pattern: {source_pattern} -> {target_pattern}",
                            method="pattern_matching"
                        )
        
        return best_relationship
    
    def _calculate_confidence(self, source: ChunkData, target: ChunkData, 
                            source_pattern: str, target_pattern: str) -> float:
        """Calculate confidence score for relationship"""
        # Semantic similarity component
        if source.embedding is not None and target.embedding is not None:
            semantic_sim = cosine_similarity(
                source.embedding.reshape(1, -1), 
                target.embedding.reshape(1, -1)
            )[0][0]
        else:
            semantic_sim = 0.3  # Default if embeddings not available
        
        # Pattern strength (more specific patterns = higher strength)
        pattern_strength = 0.8 if r'\d+' in source_pattern else 0.6
        
        # Distance penalty (closer chunks more likely to be related)
        # Convert chunk_id to int if it's a string to avoid TypeError
        try:
            source_id = int(source.chunk_id) if isinstance(source.chunk_id, str) else source.chunk_id
            target_id = int(target.chunk_id) if isinstance(target.chunk_id, str) else target.chunk_id
            distance_penalty = max(0.1, 1.0 - abs(source_id - target_id) * 0.001)
        except (ValueError, TypeError):
            # Fallback if conversion fails or other errors occur
            distance_penalty = 0.5  # Use a neutral value
            logger.warning(f"Could not calculate distance penalty between chunks {source.chunk_id} and {target.chunk_id}. Using default value.")
        
        # Combined confidence
        confidence = (
            0.3 * semantic_sim + 
            0.5 * pattern_strength + 
            0.2 * distance_penalty
        )
        
        return min(confidence, 1.0)

class HopRAGGraphProcessor:
    """Main graph processing engine with memory optimization"""
    
    def __init__(self, db_config: DatabaseConfig, embedding_model: str = "climatebert/distilroberta-base-climate-f"):
        self.db_config = db_config
        self.db_connection = Connection(config=db_config)
        self.embedder = MemoryOptimizedEmbedder(embedding_model)
        self.relationship_detector = LogicalRelationshipDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize database connections"""
        success = self.db_connection.connect()
        if not success:
            raise RuntimeError("Failed to connect to database")
        logger.info("Graph processor initialized")
    
    async def process_embeddings_batch(self, batch_size: int = 500):
        """Generate embeddings for chunks without embeddings"""
        
        engine = self.db_connection.get_engine()
        
        # Get chunks without embeddings using the correct table name
        with engine.connect() as conn:
            chunks_data = conn.execute(text("""
                SELECT id, content 
                FROM doc_chunks 
                WHERE transformer_embedding IS NULL 
                ORDER BY id
                LIMIT :batch_limit
            """), {"batch_limit": batch_size * 5}).fetchall()
        
        if not chunks_data:
            logger.info("No chunks need embeddings")
            return
        
        logger.info(f"Processing embeddings for {len(chunks_data)} chunks")
        
        # Process in smaller batches to manage memory
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i + batch_size]
            
            # Extract texts and generate embeddings
            texts = [row.content for row in batch]
            embeddings = self.embedder.encode_batch(texts)
            
            # Update database with correct column name
            with engine.connect() as conn:
                for j, row in enumerate(batch):
                    conn.execute(text(
                        "UPDATE doc_chunks SET transformer_embedding = :embedding WHERE id = :chunk_id"
                    ), {
                        "embedding": embeddings[j].tolist(),
                        "chunk_id": row.id
                    })
                conn.commit()
            
            logger.info(f"Updated embeddings for batch {i//batch_size + 1}")
            
            # Force garbage collection
            del embeddings, texts
            gc.collect()
    
    async def get_doc_chunk_count(self, doc_id: str) -> int:
        """Get the number of chunks for a specific document"""
        engine = self.db_connection.get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM doc_chunks 
                WHERE doc_id = :doc_id
            """), {"doc_id": doc_id}).scalar()
            
        return result or 0
    
    async def build_relationships_sparse(self, max_neighbors: int = 50, min_confidence: float = 0.6, 
                                       doc_id: str = None, force_commit: bool = False):
        """Build relationships using sparse approach for memory efficiency"""
        
        engine = self.db_connection.get_engine()
        
        # Get all chunks with embeddings using correct table/column names
        # If doc_id is provided, filter chunks by that document
        with engine.connect() as conn:
            if doc_id:
                logger.info(f"Building relationships for document: {doc_id}")
                chunks_data = conn.execute(text("""
                    SELECT id, content, transformer_embedding
                    FROM doc_chunks 
                    WHERE transformer_embedding IS NOT NULL
                    AND doc_id = :doc_id
                    ORDER BY id
                """), {"doc_id": doc_id}).fetchall()
            else:
                chunks_data = conn.execute(text("""
                    SELECT id, content, transformer_embedding
                    FROM doc_chunks 
                    WHERE transformer_embedding IS NOT NULL
                    ORDER BY id
                """)).fetchall()
        
        if len(chunks_data) < 2:
            logger.warning(f"Not enough chunks with embeddings {f'for document {doc_id}' if doc_id else ''}")
            return
        
        logger.info(f"Building relationships for {len(chunks_data)} chunks")
        
        # Convert to numpy for efficient operations
        chunk_ids = [row.id for row in chunks_data]
        embeddings = np.array([row.transformer_embedding for row in chunks_data])
        
        # Use approximate nearest neighbors for efficiency
        nbrs = NearestNeighbors(n_neighbors=min(max_neighbors, len(chunks_data)), 
                              algorithm='auto', metric='euclidean')
        nbrs.fit(embeddings)
        
        all_relationships = []
        batch_size = 100
        
        for i in range(0, len(chunks_data), batch_size):
            batch_end = min(i + batch_size, len(chunks_data))
            batch_chunks = []
            
            # Create ChunkData objects for current batch
            for j in range(i, batch_end):
                chunk_data = ChunkData(
                    chunk_id=chunk_ids[j],
                    content=chunks_data[j].content,
                    embedding=embeddings[j]
                )
                batch_chunks.append(chunk_data)
            
            # Find neighbors and detect relationships
            for k, source_chunk in enumerate(batch_chunks):
                actual_index = i + k
                
                # Get nearest neighbors
                distances, neighbor_indices = nbrs.kneighbors([embeddings[actual_index]])
                
                # Create target chunks from neighbors
                target_chunks = []
                for neighbor_idx in neighbor_indices[0]:
                    if neighbor_idx != actual_index:  # Skip self
                        target_chunk = ChunkData(
                            chunk_id=chunk_ids[neighbor_idx],
                            content=chunks_data[neighbor_idx].content,
                            embedding=embeddings[neighbor_idx]
                        )
                        target_chunks.append(target_chunk)
                
                # Detect relationships
                relationships = self.relationship_detector.detect_relationships_batch(
                    [source_chunk], target_chunks
                )
                
                # Filter by confidence
                filtered_relationships = [
                    rel for rel in relationships 
                    if rel.confidence >= min_confidence
                ]
                
                all_relationships.extend(filtered_relationships)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks_data) + batch_size - 1)//batch_size}")
            
            # Cleanup memory
            del batch_chunks
            gc.collect()
        
        # Insert relationships (using ORM to ensure consistency)
        if all_relationships:
            logger.info(f"Inserting {len(all_relationships)} relationships into database")
            
            # Create relationships table if it doesn't exist
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS logical_relationships (
                        id SERIAL PRIMARY KEY,
                        source_chunk_id INTEGER NOT NULL,
                        target_chunk_id INTEGER NOT NULL,
                        relationship_type VARCHAR(50) NOT NULL,
                        confidence FLOAT NOT NULL,
                        evidence TEXT,
                        method VARCHAR(50) DEFAULT 'rule_based',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_chunk_id) REFERENCES doc_chunks(id),
                        FOREIGN KEY (target_chunk_id) REFERENCES doc_chunks(id)
                    )
                """))
                conn.commit()
                
                # Insert relationships in batches for better performance
                try:
                    # Use batch execution for better performance
                    for i in range(0, len(all_relationships), 100):
                        batch = all_relationships[i:i+100]
                        # Execute batch as transaction
                        conn.begin()
                        for rel in batch:
                            conn.execute(text("""
                                INSERT INTO logical_relationships 
                                (source_chunk_id, target_chunk_id, relationship_type, confidence, evidence, method)
                                VALUES (:source_id, :target_id, :rel_type, :confidence, :evidence, :method)
                            """), {
                                "source_id": rel.source_id,
                                "target_id": rel.target_id,
                                "rel_type": rel.relationship_type,
                                "confidence": rel.confidence,
                                "evidence": rel.evidence,
                                "method": rel.method
                            })
                        conn.commit()
                        logger.info(f"Inserted batch {i//100 + 1} of {(len(all_relationships) + 99)//100}")
                        
                    # Force commit if requested (extra safety)
                    if force_commit:
                        conn.commit()
                        logger.info("Forced final commit of relationship data")
                
                except Exception as e:
                    logger.error(f"Error inserting relationships: {e}")
                    conn.rollback()
                    raise
            
            logger.info(f"Successfully inserted {len(all_relationships)} relationships into logical_relationships table")
        else:
            logger.warning("No relationships detected to insert")
        
        # Cleanup
        del embeddings, all_relationships
        gc.collect()
    
    async def bfs_multi_hop_retrieve(self, query: str, max_hops: int = 3, 
                                   top_k: int = 20) -> List[Dict]:
        """Optimized BFS multi-hop retrieval"""
        
        engine = self.db_connection.get_engine()
        
        with engine.connect() as conn:
            # Optimized BFS query with correct table names
            result = conn.execute(text("""
                WITH RECURSIVE hop_expansion AS (
                    -- Initial query matching with relevance scoring
                    SELECT 
                        dc.id as chunk_id,
                        dc.content,
                        ARRAY[dc.id] as path,
                        0 as hops,
                        1.0 as confidence,
                        ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', :query)) as initial_relevance
                    FROM doc_chunks dc
                    WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', :query)
                    ORDER BY initial_relevance DESC
                    LIMIT 5
                    
                    UNION ALL
                    
                    -- Recursive expansion with optimizations
                    SELECT 
                        lr.target_chunk_id as chunk_id,
                        dc.content,
                        he.path || lr.target_chunk_id,
                        he.hops + 1,
                        he.confidence * lr.confidence * 0.9 as confidence,
                        he.initial_relevance * 0.8
                    FROM hop_expansion he
                    JOIN logical_relationships lr ON he.chunk_id = lr.source_chunk_id
                    JOIN doc_chunks dc ON lr.target_chunk_id = dc.id
                    WHERE he.hops < :max_hops 
                      AND lr.confidence > 0.7
                      AND he.confidence > 0.4
                      AND NOT (lr.target_chunk_id = ANY(he.path))
                      AND array_length(he.path, 1) < 20
                )
                SELECT DISTINCT 
                    chunk_id, 
                    content, 
                    hops, 
                    confidence,
                    initial_relevance
                FROM hop_expansion
                WHERE confidence > 0.3
                ORDER BY confidence DESC, initial_relevance DESC, hops ASC
                LIMIT :top_k;
            """), {"query": query, "max_hops": max_hops, "top_k": top_k * 2}).fetchall()
        
        return [dict(row._mapping) for row in result]
    
    async def analyze_graph_structure(self, chunk_ids: List[int]) -> List[GraphAnalysisResult]:
        """Analyze graph structure and compute centrality measures"""
        
        if not chunk_ids:
            return []
        
        engine = self.db_connection.get_engine()
        
        # Get subgraph data with correct table names
        with engine.connect() as conn:
            # Get nodes
            nodes_data = conn.execute(text("""
                SELECT id as chunk_id, content 
                FROM doc_chunks 
                WHERE id = ANY(:chunk_ids)
            """), {"chunk_ids": chunk_ids}).fetchall()
            
            # Get edges within subgraph
            edges_data = conn.execute(text("""
                SELECT source_chunk_id, target_chunk_id, confidence, relationship_type
                FROM logical_relationships 
                WHERE source_chunk_id = ANY(:chunk_ids) AND target_chunk_id = ANY(:chunk_ids)
                AND confidence > 0.5
            """), {"chunk_ids": chunk_ids}).fetchall()
        
        if not edges_data:
            # Return basic results if no relationships
            return [
                GraphAnalysisResult(
                    chunk_id=row.chunk_id,
                    content=row.content,
                    centrality_scores={'degree': 0.0, 'pagerank': 1.0/len(nodes_data), 'betweenness': 0.0},
                    community_id=0,
                    final_score=0.5
                )
                for row in nodes_data
            ]
        
        # Build NetworkX graph for analysis
        G = nx.DiGraph()
        
        # Add nodes
        for row in nodes_data:
            G.add_node(row.chunk_id, content=row.content)
        
        # Add edges
        for row in edges_data:
            G.add_edge(row.source_chunk_id, row.target_chunk_id, 
                      weight=row.confidence, type=row.relationship_type)
        
        # Compute centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            pagerank = nx.pagerank(G, weight='weight', max_iter=100)
            
            # Betweenness centrality (expensive, so limit to smaller graphs)
            if len(G.nodes()) < 1000:
                betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            else:
                betweenness_centrality = {node: 0.0 for node in G.nodes()}
            
        except Exception as e:
            logger.warning(f"Centrality calculation failed: {e}")
            # Fallback to simple measures
            degree_centrality = {node: G.degree(node) for node in G.nodes()}
            max_degree = max(degree_centrality.values()) if degree_centrality else 1
            degree_centrality = {k: v/max_degree for k, v in degree_centrality.items()}
            
            pagerank = {node: 1.0/len(G.nodes()) for node in G.nodes()}
            betweenness_centrality = {node: 0.0 for node in G.nodes()}
        
        # Community detection
        try:
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            communities = nx.community.greedy_modularity_communities(G_undirected)
            
            # Create community mapping
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
        except:
            # Fallback: single community
            community_map = {node: 0 for node in G.nodes()}
        
        # Create analysis results
        results = []
        for row in nodes_data:
            chunk_id = row.chunk_id
            
            centrality_scores = {
                'degree': degree_centrality.get(chunk_id, 0.0),
                'pagerank': pagerank.get(chunk_id, 0.0),
                'betweenness': betweenness_centrality.get(chunk_id, 0.0)
            }
            
            # Calculate final score (weighted combination)
            final_score = (
                0.3 * centrality_scores['degree'] +
                0.5 * centrality_scores['pagerank'] +
                0.2 * centrality_scores['betweenness']
            )
            
            results.append(GraphAnalysisResult(
                chunk_id=chunk_id,
                content=row.content,
                centrality_scores=centrality_scores,
                community_id=community_map.get(chunk_id, 0),
                final_score=final_score
            ))
        
        return sorted(results, key=lambda x: x.final_score, reverse=True)
    
    async def get_top_ranked_nodes(self, query: str, max_hops: int = 3) -> Dict:
        """Get top 20 ranked nodes using BFS + graph analysis"""
        
        start_time = datetime.utcnow()
        
        # Step 1: BFS retrieval
        bfs_results = await self.bfs_multi_hop_retrieve(query, max_hops, top_k=50)
        
        if not bfs_results:
            return {
                "query": query,
                "total_nodes": 0,
                "top_nodes": [],
                "execution_time_ms": 0,
                "method": "bfs_graph_analysis"
            }
        
        # Step 2: Graph analysis
        chunk_ids = [result['chunk_id'] for result in bfs_results]
        analysis_results = await self.analyze_graph_structure(chunk_ids)
        
        # Step 3: Combine BFS relevance with graph analysis
        bfs_scores = {result['chunk_id']: result['confidence'] for result in bfs_results}
        
        final_results = []
        for analysis in analysis_results:
            bfs_score = bfs_scores.get(analysis.chunk_id, 0.0)
            
            # Combined score: BFS relevance + graph centrality
            combined_score = 0.6 * bfs_score + 0.4 * analysis.final_score
            
            final_results.append({
                "chunk_id": analysis.chunk_id,
                "content": analysis.content,
                "bfs_confidence": bfs_score,
                "centrality_scores": analysis.centrality_scores,
                "community_id": analysis.community_id,
                "combined_score": combined_score
            })
        
        # Sort by combined score and take top 20
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_20 = final_results[:20]
        
        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return {
            "query": query,
            "total_nodes": len(final_results),
            "top_nodes": top_20,
            "execution_time_ms": execution_time,
            "method": "bfs_graph_analysis",
            "graph_stats": {
                "communities_found": len(set(r["community_id"] for r in final_results)),
                "max_centrality": max(r["centrality_scores"]["pagerank"] for r in final_results) if final_results else 0
            }
        }
    
    async def close(self):
        """Cleanup resources"""
        # No need to close Connection as it's synchronous
        self.executor.shutdown(wait=True)
        logger.info("Graph processor closed")

class HopRAGClassifier:
    """
    Classifier that categorizes nodes based on centrality score thresholds
    and ranks top 20 nodes into different importance categories
    """
    
    def __init__(self):
        # Hard-coded thresholds for centrality scores
        self.thresholds = {
            'degree': {
                'high': 0.7,    # High connectivity nodes
                'medium': 0.4,  # Medium connectivity nodes
                'low': 0.0      # Low connectivity nodes (everything else)
            },
            'pagerank': {
                'high': 0.15,   # High influence nodes
                'medium': 0.05, # Medium influence nodes
                'low': 0.0      # Low influence nodes (everything else)
            },
            'betweenness': {
                'high': 0.3,    # High bridging nodes
                'medium': 0.1,  # Medium bridging nodes
                'low': 0.0      # Low bridging nodes (everything else)
            }
        }
        
        # Classification categories
        self.categories = {
            'CORE_HUB': 'High centrality in all measures - critical knowledge nodes',
            'AUTHORITY': 'High PageRank - influential reference nodes',
            'CONNECTOR': 'High betweenness - bridge nodes between topics',
            'PERIPHERAL': 'Lower centrality - supporting information nodes'
        }
    
    def classify_nodes(self, processor_results: Dict) -> Dict:
        """
        Classify nodes from processor results into categories based on centrality thresholds
        
        Args:
            processor_results: Results from processor.get_top_ranked_nodes()
        
        Returns:
            Dict: Structured classification results with rankings
        """
        if not processor_results.get('top_nodes'):
            return self._empty_classification_result(processor_results)
        
        top_nodes = processor_results['top_nodes'][:20]  # Ensure we only take top 20
        
        # Classify each node
        classified_nodes = []
        for node in top_nodes:
            classification = self._classify_single_node(node)
            classified_nodes.append(classification)
        
        # Group by classification and rank within each group
        classification_groups = self._group_and_rank_nodes(classified_nodes)
        
        # Generate final structured output
        result = self._generate_classification_output(
            processor_results, 
            classification_groups, 
            classified_nodes
        )
        
        return result
    
    def _classify_single_node(self, node: Dict) -> NodeClassification:
        """Classify a single node based on its centrality scores"""
        centrality = node['centrality_scores']
        
        # Extract centrality values
        degree = centrality.get('degree', 0.0)
        pagerank = centrality.get('pagerank', 0.0)
        betweenness = centrality.get('betweenness', 0.0)
        
        # Determine classification category
        classification = self._determine_category(degree, pagerank, betweenness)
        
        # Determine confidence level based on score distribution
        confidence_level = self._determine_confidence_level(degree, pagerank, betweenness)
        
        return NodeClassification(
            chunk_id=node['chunk_id'],
            content=node['content'],
            centrality_scores=centrality,
            combined_score=node['combined_score'],
            classification=classification,
            rank_within_class=0,  # Will be set during grouping
            confidence_level=confidence_level
        )
    
    def _determine_category(self, degree: float, pagerank: float, betweenness: float) -> str:
        """Determine the category based on centrality thresholds"""
        
        # Count how many measures are "high"
        high_counts = sum([
            degree >= self.thresholds['degree']['high'],
            pagerank >= self.thresholds['pagerank']['high'],
            betweenness >= self.thresholds['betweenness']['high']
        ])
        
        # Specific classification logic
        if high_counts >= 2:
            return 'CORE_HUB'
        elif pagerank >= self.thresholds['pagerank']['high']:
            return 'AUTHORITY'
        elif betweenness >= self.thresholds['betweenness']['high']:
            return 'CONNECTOR'
        else:
            return 'PERIPHERAL'
    
    def _determine_confidence_level(self, degree: float, pagerank: float, betweenness: float) -> str:
        """Determine confidence level based on score magnitude"""
        avg_score = (degree + pagerank + betweenness) / 3
        
        if avg_score >= 0.6:
            return 'HIGH'
        elif avg_score >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _group_and_rank_nodes(self, classified_nodes: List[NodeClassification]) -> Dict[str, List[NodeClassification]]:
        """Group nodes by classification and rank within each group"""
        groups = {category: [] for category in self.categories.keys()}
        
        # Group nodes
        for node in classified_nodes:
            groups[node.classification].append(node)
        
        # Sort and rank within each group
        for category, nodes in groups.items():
            # Sort by combined score descending
            nodes.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Assign ranks within the group
            for i, node in enumerate(nodes):
                node.rank_within_class = i + 1
        
        return groups
    
    def _generate_classification_output(self, processor_results: Dict, 
                                      classification_groups: Dict[str, List[NodeClassification]], 
                                      all_classified_nodes: List[NodeClassification]) -> Dict:
        """Generate the final structured JSON output"""
        
        # Summary statistics
        total_nodes = len(all_classified_nodes)
        category_counts = {cat: len(nodes) for cat, nodes in classification_groups.items()}
        
        # Detailed results for each category
        category_details = {}
        for category, nodes in classification_groups.items():
            category_details[category] = {
                'description': self.categories[category],
                'count': len(nodes),
                'nodes': [
                    {
                        'chunk_id': node.chunk_id,
                        'content': node.content[:200] + "..." if len(node.content) > 200 else node.content,
                        'centrality_scores': node.centrality_scores,
                        'combined_score': round(node.combined_score, 4),
                        'rank_within_category': node.rank_within_class,
                        'confidence_level': node.confidence_level
                    }
                    for node in nodes
                ]
            }
        
        # Overall ranking (all nodes sorted by combined score)
        all_classified_nodes.sort(key=lambda x: x.combined_score, reverse=True)
        overall_ranking = [
            {
                'overall_rank': i + 1,
                'chunk_id': node.chunk_id,
                'classification': node.classification,
                'combined_score': round(node.combined_score, 4),
                'confidence_level': node.confidence_level
            }
            for i, node in enumerate(all_classified_nodes)
        ]
        
        # Threshold information
        threshold_info = {
            'thresholds_used': self.thresholds,
            'classification_criteria': {
                'CORE_HUB': 'High centrality in 2+ measures',
                'AUTHORITY': 'High PageRank score',
                'CONNECTOR': 'High betweenness centrality',
                'PERIPHERAL': 'Lower centrality scores'
            }
        }
        
        return {
            'metadata': {
                'query': processor_results.get('query', ''),
                'processing_method': processor_results.get('method', 'unknown'),
                'total_nodes_processed': processor_results.get('total_nodes', 0),
                'top_nodes_classified': total_nodes,
                'execution_time_ms': processor_results.get('execution_time_ms', 0),
                'classification_timestamp': datetime.utcnow().isoformat(),
                'graph_stats': processor_results.get('graph_stats', {})
            },
            'classification_summary': {
                'total_classified': total_nodes,
                'category_distribution': category_counts,
                'confidence_distribution': {
                    'HIGH': len([n for n in all_classified_nodes if n.confidence_level == 'HIGH']),
                    'MEDIUM': len([n for n in all_classified_nodes if n.confidence_level == 'MEDIUM']),
                    'LOW': len([n for n in all_classified_nodes if n.confidence_level == 'LOW'])
                }
            },
            'category_details': category_details,
            'overall_ranking': overall_ranking,
            'classification_config': threshold_info
        }
    
    def _empty_classification_result(self, processor_results: Dict) -> Dict:
        """Return empty result structure when no nodes to classify"""
        return {
            'metadata': {
                'query': processor_results.get('query', ''),
                'processing_method': processor_results.get('method', 'unknown'),
                'total_nodes_processed': 0,
                'top_nodes_classified': 0,
                'execution_time_ms': processor_results.get('execution_time_ms', 0),
                'classification_timestamp': datetime.utcnow().isoformat(),
                'graph_stats': {}
            },
            'classification_summary': {
                'total_classified': 0,
                'category_distribution': {cat: 0 for cat in self.categories.keys()},
                'confidence_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            },
            'category_details': {
                cat: {'description': desc, 'count': 0, 'nodes': []}
                for cat, desc in self.categories.items()
            },
            'overall_ranking': [],
            'classification_config': {
                'thresholds_used': self.thresholds,
                'classification_criteria': {
                    'CORE_HUB': 'High centrality in 2+ measures',
                    'AUTHORITY': 'High PageRank score',
                    'CONNECTOR': 'High betweenness centrality',
                    'PERIPHERAL': 'Lower centrality scores'
                }
            }
        }
    
    def save_classification_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """Save classification results to JSON file"""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"hoprag_classification_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Classification results saved to: {filename}")
        return filename
    
    def print_classification_summary(self, results: Dict):
        """Print a human-readable summary of classification results"""
        metadata = results['metadata']
        summary = results['classification_summary']
        
        print(f"\n=== HopRAG Classification Results ===")
        print(f"Query: {metadata['query']}")
        print(f"Nodes Classified: {summary['total_classified']}")
        print(f"Processing Time: {metadata['execution_time_ms']}ms")
        
        print(f"\n--- Category Distribution ---")
        for category, count in summary['category_distribution'].items():
            print(f"{category}: {count} nodes")
        
        print(f"\n--- Confidence Distribution ---")
        for level, count in summary['confidence_distribution'].items():
            print(f"{level}: {count} nodes")
        
        print(f"\n--- Top 5 Overall Ranking ---")
        for i, node in enumerate(results['overall_ranking'][:5]):
            print(f"{i+1}. Chunk {node['chunk_id']} ({node['classification']}) - Score: {node['combined_score']}")

async def main():
    """Main processing function"""
    try:
        # Initialize
        config = DatabaseConfig.from_env()
        processor = HopRAGGraphProcessor(config)
        await processor.initialize()
        
        # Step 1: Generate embeddings
        logger.info("Step 1: Generating embeddings...")
        await processor.process_embeddings_batch(batch_size=500)
        
        # Step 2: Build relationships
        logger.info("Step 2: Building relationships...")
        await processor.build_relationships_sparse(max_neighbors=30, min_confidence=0.6)
        
        # Step 3: Test query processing
        test_query = "Singapore NDC 2030 emission targets"
        logger.info(f"Step 3: Testing query: {test_query}")
        
        results = await processor.get_top_ranked_nodes(test_query, max_hops=3)
        
        # Save results
        output_file = f"hoprag_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Found {results['total_nodes']} total nodes, returning top 20")
        logger.info(f"Execution time: {results['execution_time_ms']}ms")
        
        # Print summary
        print(f"\n=== HopRAG Results Summary ===")
        print(f"Query: {results['query']}")
        print(f"Total nodes found: {results['total_nodes']}")
        print(f"Execution time: {results['execution_time_ms']}ms")
        print(f"Communities found: {results['graph_stats']['communities_found']}")
        print(f"\nTop 5 nodes:")
        for i, node in enumerate(results['top_nodes'][:5], 1):
            print(f"{i}. Score: {node['combined_score']:.3f} | Content: {node['content'][:100]}...")
        
        # Classification
        logger.info("Classifying top nodes...")
        classifier = HopRAGClassifier()
        classification_results = classifier.classify_nodes(results)
        
        # Save classification results
        classifier.save_classification_results(classification_results)
        
        await processor.close()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())