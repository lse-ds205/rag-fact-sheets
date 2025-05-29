# RAG Fact Sheets Database Schema

This document outlines the database schema for the RAG (Retrieval-Augmented Generation) Fact Sheets system, which stores country-specific document data, questions, and answers with appropriate relationships.

## Overview

The database is organized into 7 interconnected tables with cascading delete relationships to maintain data integrity. The schema supports document storage, chunking, semantic relationships, and question-answer functionality.

## Database Diagram

```mermaid
erDiagram
    countries ||--o{ documents : "has many"
    documents ||--o{ doc_chunks : "contains"
    doc_chunks ||--o{ logical_relationships : "source"
    doc_chunks ||--o{ logical_relationships : "target"
    countries ||--o{ questions_answers : "has answers"
    questions ||--o{ questions_answers : "has responses"
    doc_chunks ||--o{ citations : "cited in"
    questions_answers ||--o{ citations : "contains"
    
    countries {
        VARCHAR id PK "Country name"
    }
    
    questions {
        INTEGER id PK
        TEXT question "Question text"
    }
    
    documents {
        UUID doc_id PK
        TEXT country FK "CASCADE"
        TIMESTAMPTZ created_at
        TEXT title
        TEXT url
    }
    
    doc_chunks {
        UUID id PK
        UUID doc_id FK "CASCADE"
        TEXT content
        INTEGER chunk_index
        ARRAY transformer_embedding
    }
    
    logical_relationships {
        UUID id PK
        UUID source_chunk_id FK "CASCADE"
        UUID target_chunk_id FK "CASCADE"
        VARCHAR relationship_type
        FLOAT confidence
    }
    
    questions_answers {
        UUID id PK
        TEXT country FK "CASCADE"
        INTEGER question FK "CASCADE"
        TEXT summary
        TEXT detailed_response
    }
    
    citations {
        UUID id PK
        UUID cited_chunk_id FK "CASCADE"
        UUID cited_in_answer_id FK "CASCADE"
    }
```

## Database Setup

To set up the database, run:

```bash
python setup_database.py 'postgresql://username:password@localhost:5432/dbname'
```

## Schema Diagram

```
countries (1)
    ↓ ON DELETE CASCADE
documents (3) 
    ↓ ON DELETE CASCADE
doc_chunks (4) ←→ logical_relationships (5)
    ↓ ON DELETE CASCADE     ↑ ON DELETE CASCADE
questions (2) → questions_answers (6) → citations (7)
                ↑ ON DELETE CASCADE     ↑ ON DELETE CASCADE
```

## Table Descriptions

### 1. countries

Primary reference table containing country identifiers.

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key, country name |

### 2. questions

Pre-defined questions for document analysis.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| question | TEXT | The question text |

### 3. documents

Stores document metadata with link to countries.

| Column | Type | Description |
|--------|------|-------------|
| doc_id | UUID | Primary key |
| scraped_at | TIMESTAMPTZ | When document was scraped |
| downloaded_at | TIMESTAMPTZ | When document was downloaded |
| processed_at | TIMESTAMPTZ | When document was processed |
| last_download_attempt | TIMESTAMPTZ | Last time download was attempted |
| download_error | TEXT | Error message if download failed |
| download_attempts | INTEGER | Number of download attempts |
| country | TEXT | Foreign key to countries(id) with CASCADE delete |
| title | TEXT | Document title |
| url | TEXT | Source URL |
| language | TEXT | Document language |
| submission_date | DATE | When document was submitted |
| file_path | TEXT | Path to stored file |
| file_size | DOUBLE PRECISION | File size in bytes |
| extracted_text | TEXT | Plain text extracted from document |
| chunks | JSONB | JSON representation of chunks |
| created_at | TIMESTAMPTZ | Record creation timestamp |
| updated_at | TIMESTAMPTZ | Record update timestamp |

### 4. doc_chunks

Stores segments of documents for efficient retrieval.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| doc_id | UUID | Foreign key to documents(doc_id) with CASCADE delete |
| content | TEXT | Chunk text content |
| chunk_index | INTEGER | Position within document |
| paragraph | INTEGER | Paragraph number |
| language | TEXT | Chunk language |
| transformer_embedding | DOUBLE PRECISION[] | Vector embedding from transformer model |
| word2vec_embedding | DOUBLE PRECISION[] | Word2Vec embedding |
| hoprag_embedding | INTEGER[] | HopRAG embedding |
| chunk_data | JSONB | Additional chunk metadata |
| created_at | TIMESTAMPTZ | Record creation timestamp |
| updated_at | TIMESTAMPTZ | Record update timestamp |
| page | INTEGER | Page number in document |
| content_hash | TEXT | Hash of content for deduplication |

### 5. logical_relationships

Defines semantic relationships between document chunks.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (auto-generated) |
| source_chunk_id | UUID | Foreign key to doc_chunks(id) with CASCADE delete |
| target_chunk_id | UUID | Foreign key to doc_chunks(id) with CASCADE delete |
| relationship_type | VARCHAR(50) | Type of relationship (SUPPORTS, EXPLAINS, etc.) |
| confidence | DOUBLE PRECISION | Confidence score (0.0-1.0) |
| evidence | TEXT | Supporting evidence for relationship |
| method | VARCHAR(50) | Method used to establish relationship |
| created_at | TIMESTAMPTZ | Record creation timestamp |

### 6. questions_answers

Stores answers to questions about specific countries.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (auto-generated) |
| country | TEXT | Foreign key to countries(id) with CASCADE delete |
| timestamp | TIMESTAMPTZ | When answer was generated |
| question | INTEGER | Foreign key to questions(id) with CASCADE delete |
| summary | TEXT | Brief answer summary |
| detailed_response | TEXT | Full detailed response |
| citations | UUID[] | Array of citation IDs |

### 7. citations

Links document chunks to specific answers.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (auto-generated) |
| cited_chunk_id | UUID | Foreign key to doc_chunks(id) with CASCADE delete |
| cited_in_answer_id | UUID | Foreign key to questions_answers(id) with CASCADE delete |

## Data Flow

1. Countries and questions are loaded as reference data
2. Documents are associated with countries
3. Documents are processed into chunks
4. Chunks are analyzed for logical relationships
5. Questions are answered using document chunks
6. Answers include citations to source chunks

## Deletion Behavior

All tables implement `ON DELETE CASCADE` for their foreign key relationships:

- When a country is deleted, all associated documents, chunks, answers and citations are removed
- When a document is deleted, all its chunks and related data are removed
- When a chunk is deleted, all relationships and citations referencing it are removed
- When a question is deleted, all answers to that question are removed
- When an answer is deleted, all citations in that answer are removed

This ensures database integrity and prevents orphaned records. 