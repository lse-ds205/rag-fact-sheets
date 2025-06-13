# RAG Climate Fact Sheets 

## Overview 

This repository contains the collective work of three groups from the DS205 course that developed automated RAG-based systems for extracting, analysing, and synthesising climate policy data from different authoritative sources. Each group adopted distinct approaches and tackled unique challenges in processing climate policy documents that together form a comprehensive framework for automating climate policy analysis.

## 🌐 UNFCCC NDC Analysis (`rag_unfccc`)
- Source: UNFCCC Nationally Determined Contributions (NDCs)
- Focus: Automated monitoring and standardised fact-sheet generation for newly published NDC documents
- Key Innovation: HopRAG architecture for multi-hop reasoning in complex policy documents

```mermaid 
graph TD
    A[Web Scraper<br/>Selenium Bot] --> B{New NDC<br/>Document?}
    B -->|Yes| C[PDF Processing]
    B -->|No| D[Skip]
    C --> E[Text Chunking<br/>512 tokens]
    E --> F[Dual Embedding<br/>ClimateBERT + Word2Vec]
    F --> G[HopRAG Processing]
    G --> H[Multi-hop<br/>Reasoning]
    H --> I[Fact-Sheet<br/>Generation]
    I --> J[Email Alert<br/>+ API Access]
    
    style A fill:#E3F2FD,stroke:#333,font-size:14px
    style B fill:#E3F2FD,stroke:#333,font-size:14px
    style C fill:#E8F5E9,stroke:#333,font-size:14px
    style D fill:#E8F5E9,stroke:#333,font-size:14px
    style E fill:#E8F5E9,stroke:#333,font-size:14px
    style F fill:#FFF3E0,stroke:#333,font-size:14px
    style G fill:#FCE4EC,stroke:#333,font-size:14px
    style H fill:#FCE4EC,stroke:#333,font-size:14px
    style I fill:#F3E5F5,stroke:#333,font-size:14px
    style J fill:#E0F2F1,stroke:#333,font-size:14px
```

## 🎯 Climate Action Tracker Analysis (`rag_cat`)
- Source: ClimateActionTracker.org
- Focus: Structured extraction of policy targets with confidence scoring
- Key Innovation: Hybrid NLP pipeline combining NER, dependency parsing, and semantic similarity

```mermaid
graph TD
    A[CAT Website<br/>Scraping] --> B[Text Extraction<br/>by Country]
    B --> C[Database Storage<br/>PostgreSQL]
    C --> D[Chunk Generation<br/>Sentence-based]
    D --> E[BAAI/bge-m3<br/>Embeddings]
    E --> F[Semantic Search<br/>Top-K Retrieval]
    F --> G[Policy Extraction<br/>Pipeline]
    G --> H[NER + SpaCy<br/>Processing]
    H --> I[Dependency<br/>Parsing]
    I --> J[Confidence<br/>Scoring]
    J --> K[Structured Output<br/>JSON/Markdown]
    K --> L[Q&A Boxes<br/>Generation]
    
    style A fill:#E3F2FD,stroke:#333
    style E fill:#FFF3E0,stroke:#333
    style G fill:#FCE4EC,stroke:#333
    style J fill:#E8F5E9,stroke:#333
    style L fill:#F3E5F5,stroke:#333
```

## 📊 Climate Policy Radar Dataset Analysis (`rag_policy_radar`)
- Source: Climate Policy Radar Database
- Focus: ASCOR methodology implementation for systematic climate legislation assessment
- Key Innovation: Multi-tier LLM strategy with human-in-the-loop validation

```mermaid
graph TD
    A[Climate Policy<br/>Radar Dataset] --> B[Document<br/>Ingestion]
    B --> C[Embedding Generation<br/>ClimateBERT/Word2Vec]
    C --> D[Retrieval System]
    D --> E{Tool Selection}
    E --> F[Pillar Tool<br/>CP1.a/CP1.b]
    E --> G[Custom Report<br/>Generation]
    E --> H[Sectoral<br/>Analysis]
    F --> I[Chain of Thought<br/>Evaluation]
    G --> J[Human-in-Loop<br/>Validation]
    H --> K[Temporal<br/>Analysis]
    I --> L[Structured<br/>Output]
    J --> L
    K --> L
    
    style A fill:#E8F5E9,stroke:#333
    style C fill:#FFF3E0,stroke:#333
    style E fill:#E3F2FD,stroke:#333
    style J fill:#FCE4EC,stroke:#333
    style L fill:#F3E5F5,stroke:#333
```