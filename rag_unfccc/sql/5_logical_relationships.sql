CREATE TABLE logical_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    source_chunk_id UUID NOT NULL REFERENCES doc_chunks(id) ON DELETE CASCADE,
    target_chunk_id UUID NOT NULL REFERENCES doc_chunks(id) ON DELETE CASCADE,

    relationship_type VARCHAR(50) NOT NULL CHECK (
        relationship_type IN (
            'SUPPORTS',
            'EXPLAINS',
            'CONTRADICTS',
            'FOLLOWS',
            'TEMPORAL_FOLLOWS',
            'CAUSES'
        )
    ),

    confidence DOUBLE PRECISION NOT NULL CHECK (
        confidence >= 0.0 AND confidence <= 1.0
    ),

    evidence TEXT,
    method VARCHAR(50) DEFAULT 'rule_based',

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CHECK (source_chunk_id != target_chunk_id)
);