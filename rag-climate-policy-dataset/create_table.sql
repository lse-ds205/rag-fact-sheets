-- This file creates a new table in the database to store document embeddings.
-- It includes a vector column for storing the embeddings and a country code for each document.
-- It also ensures that the vector extension is enabled in the database.
-- Can alter the table structure as needed

CREATE EXTENSION IF NOT EXISTS vector;

-- This line drops the table if it already exists, be careful if there's data already
-- Only use it when the table is not created or there is some other error:
-- DROP TABLE IF EXISTS document_embeddings; 
-- (unhighlight above line to use it)

CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT,
    document_title TEXT,
    country_code TEXT,
    original_text TEXT,
    source_hyperlink TEXT,
    climatebert_embedding VECTOR(768),
    word2vec_embedding VECTOR(768)
);
