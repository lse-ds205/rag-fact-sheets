CREATE TABLE doc_chunks (
    id UUID PRIMARY KEY,
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    paragraph INTEGER,
    language TEXT,
    transformer_embedding vector,
    word2vec_embedding vector,
    hoprag_embedding FLOAT[],
    chunk_data JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    page INTEGER,
    content_hash VARCHAR(64)
);
