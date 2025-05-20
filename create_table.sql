-- This file creates a new table in the database to store document embeddings.
-- It includes a vector column for storing the embeddings and a country code for each document.
-- It also ensures that the vector extension is enabled in the database.
-- Can alter the table structure as needed

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS document_embeddings;

CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT,
    country_code TEXT,
    embedding VECTOR(768)
);
