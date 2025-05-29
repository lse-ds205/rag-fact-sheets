# LLM Response Module Documentation

---

## 📘 Table of Contents

1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Processing Pipeline](#processing-pipeline)
4. [Helper Classes](#helper-classes)
5. [Input/Output Format](#inputoutput-format)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Error Handling](#error-handling)

---

## Overview

The `5_llm_response.py` script is the brain of our RAG pipeline—it takes retrieved chunks from the previous stage and generates structured answers using a large language model. This isn't just a simple API call; it's a sophisticated processor that handles multiple retrieval sources, validates responses, and produces consistent JSON output.

**What it does**: Processes country-specific retrieved chunks through an LLM to generate structured answers with validated citations, confidence scoring, and proper metadata.

**Why it matters**: Raw chunks are useless without interpretation. This module transforms context into actionable insights while maintaining traceability through citation validation.

---

## Core Functions

| Function | Purpose | Key Logic |
|----------|---------|-----------|
| `setup_llm()` | Initialize LLM client | Configures API client with guided JSON support |
| `load_chunks_and_prompt_from_file()` | Parse input JSON | Flexible parser for multiple retrieval formats |
| `extract_main_question()` | Clean question text | Removes instructional formatting from prompts |
| `get_llm_response()` | Generate LLM answer | Formats context, calls API, returns structured response |
| `process_response()` | Validate output | Citation validation and final JSON formatting |
| `process_country_file()` | Handle country data | Orchestrates processing for all questions per country |
| `merge_chunks_from_directories()` | Combine sources | Merges retrieve + retrieve_hop data intelligently |

---

## Processing Pipeline

```mermaid
flowchart TD
    A[Input JSON Files] --> B[Load & Parse Data]
    B --> C[Setup LLM Client]
    C --> D[For Each Country]
    D --> E[Merge Retrieve Sources]
    E --> F[For Each Question]
    F --> G[Format Chunks]
    G --> H[Generate LLM Response]
    H --> I[Validate Citations]
    I --> J[Calculate Confidence]
    J --> K[Save Results]
    K --> L[Next Question]
    L --> F
    K --> M[Next Country]
    M --> D
```

### The Workflow Explained

1. **Input Discovery**: Scans both `data/retrieve/` and `data/retrieve_hop/` for JSON files
2. **Data Merging**: Intelligently combines chunks from both retrieval strategies per question
3. **LLM Processing**: Formats context and generates structured responses using guided JSON
4. **Validation**: Ensures citations reference actual chunks and scores confidence
5. **Output**: Creates timestamped JSON files with complete metadata and traceability

---

## Helper Classes

### ChunkFormatter
Handles context preparation for the LLM:
- Formats chunk lists into readable context blocks
- Maintains chunk IDs for citation validation
- Optimizes context length for token limits

### LLMClient
Manages API interactions:
- **Guided JSON Mode**: Enforces response structure via API schema
- **Fallback Mode**: Parses unstructured responses with error handling
- **Configuration**: Uses environment variables for model, temperature, tokens

### ResponseProcessor
Validates and enriches LLM outputs:
- **Citation Validation**: Ensures all references exist in original chunks
- **Error Handling**: Creates consistent error responses for failures
- **Data Enrichment**: Adds chunk metadata to citations

### ConfidenceClassification
Scores response reliability:
- **Retrieval Quality**: Analyzes chunk relevance scores
- **Response Coherence**: Evaluates answer completeness
- **Citation Coverage**: Measures how well sources support claims

---

## Input/Output Format

### Input Structure Visualization

```mermaid
graph TD
    A[Country JSON File] --> B[metadata]
    A --> C[questions]
    
    B --> B1[country: Brazil]
    B --> B2[timestamp: 2024-01-01]
    B --> B3[source info]
    
    C --> D[question_1]
    C --> E[question_2]
    C --> F[question_n]
    
    D --> D1[question text]
    D --> D2[evaluated_chunks]
    
    D2 --> G[chunk_1]
    D2 --> H[chunk_2]
    D2 --> I[chunk_n]
    
    G --> G1[id: chunk_123]
    G --> G2[content: Brazil commits...]
    G --> G3[metadata]
    
    G3 --> G31[source: brazil_ndc.pdf]
    G3 --> G32[confidence_score: 0.89]
    G3 --> G33[page_number: 15]
    
    style A fill:#e1f5fe
    style D2 fill:#f3e5f5
    style G fill:#e8f5e8
```

### Output Structure Visualization

```mermaid
graph TD
    A[LLM Response File] --> B[metadata]
    A --> C[questions]
    
    B --> B1[country_name: Brazil]
    B --> B2[timestamp: 2024-01-01T12:30:00]
    B --> B3[source_file: brazil.json]
    B --> B4[question_count: 10]
    
    C --> D[question_1]
    C --> E[question_2]
    C --> F[question_n]
    
    D --> D1[question_number: 1]
    D --> D2[query_text: What are targets?]
    D --> D3[llm_response]
    D --> D4[chunk_count: 8]
    
    D3 --> G[answer: Brazil has committed...]
    D3 --> H[confidence: high]
    D3 --> I[citations]
    D3 --> J[key_points]
    
    I --> K[citation_1]
    I --> L[citation_2]
    
    K --> K1[chunk_id: chunk_123]
    K --> K2[source: brazil_ndc.pdf]
    K --> K3[relevance_score: 0.89]
    
    J --> J1[37% reduction by 2025]
    J --> J2[43% goal by 2030]
    
    style A fill:#fff3e0
    style D3 fill:#e3f2fd
    style I fill:#f1f8e9
    style J fill:#fce4ec
```

### Data Flow Transformation

```mermaid
flowchart LR
    A[Raw Chunks] --> B[Chunk Formatter]
    B --> C[LLM Context]
    C --> D[LLM API]
    D --> E[Raw Response]
    E --> F[Response Processor]
    F --> G[Validated Output]
    
    H[Original Chunks] --> F
    I[Citation IDs] --> F
    
    subgraph "Input Sources"
        J[data/retrieve/]
        K[data/retrieve_hop/]
    end
    
    J --> A
    K --> A
    
    subgraph "Processing"
        B
        D
        F
    end
    
    subgraph "Output"
        L[data/llm/country_timestamp.json]
    end
    
    G --> L
    
    style A fill:#ffebee
    style G fill:#e8f5e8
    style L fill:#fff3e0
```

### Example JSON Structures

#### Input Example
```json
{
  "metadata": {
    "country": "Brazil",
    "timestamp": "2024-01-01T12:00:00"
  },
  "questions": {
    "question_1": {
      "question": "What are Brazil's emission targets?",
      "evaluated_chunks": [
        {
          "id": "chunk_123",
          "content": "Brazil commits to reducing emissions by 37%...",
          "metadata": {
            "source": "brazil_ndc.pdf",
            "confidence_score": 0.89
          }
        }
      ]
    }
  }
}
```

#### Output Example
```json
{
  "metadata": {
    "country_name": "Brazil",
    "timestamp": "2024-01-01T12:30:00",
    "source_file": "brazil.json",
    "question_count": 10
  },
  "questions": {
    "question_1": {
      "question_number": 1,
      "query_text": "What are Brazil's emission targets?",
      "llm_response": {
        "answer": "Brazil has committed to reducing greenhouse gas emissions by 37% below 2005 levels by 2025...",
        "confidence": "high",
        "citations": [
          {
            "chunk_id": "chunk_123",
            "source": "brazil_ndc.pdf",
            "relevance_score": 0.89
          }
        ],
        "key_points": [
          "37% reduction target by 2025",
          "Additional 43% goal by 2030"
        ]
      },
      "chunk_count": 8
    }
  }
}
```

---

## Configuration

### Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_MODEL` | Model identifier | `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| `LLM_TEMPERATURE` | Response randomness | `0.1` |
| `LLM_MAX_TOKENS` | Response length limit | `4000` |
| `AI_API_KEY` | API authentication | Required |
| `AI_BASE_URL` | API endpoint | Required |

### Guided JSON Support
The module automatically detects whether the LLM API supports guided JSON:
- **Enabled**: Enforces response structure via API schema for reliability
- **Disabled**: Falls back to regex parsing with error recovery

---

## Usage Examples

### Standard Operation
```bash
# Process all countries from retrieve directories
python 5_llm_response.py

# Override with custom prompt
python 5_llm_response.py --prompt "Summarize climate commitments"

# Disable guided JSON for compatibility
python 5_llm_response.py --no-guided-json
```

### Legacy Support
```bash
# Direct JSON input (legacy mode)
python 5_llm_response.py --chunks '[{"id":1,"content":"test"}]' --prompt "Analyze this"
```

---

## Error Handling

The module implements comprehensive error handling:

### Input Validation
- **Missing Files**: Graceful handling when directories don't exist
- **Malformed JSON**: Clear error messages for parsing failures
- **Missing Keys**: Flexible parsing that tries multiple data structures

### API Failures
- **Connection Issues**: Retry logic with exponential backoff
- **Rate Limits**: Automatic throttling and queue management
- **Parsing Errors**: Fallback to regex extraction when JSON fails

### Output Consistency
- **Failed Responses**: Creates error entries instead of crashing
- **Partial Success**: Continues processing other countries/questions
- **Logging**: Comprehensive logs for debugging and monitoring

The error responses maintain the same structure as successful ones, ensuring downstream processes can handle failures gracefully.

---

## Design Choices

### Why Merge Two Retrieval Sources?
The module combines data from `retrieve/` (semantic similarity) and `retrieve_hop/` (graph relationships) because each captures different aspects of relevance. This dual approach provides richer context for more comprehensive answers.

### Why Guided JSON?
Structured responses are critical for downstream processing. Guided JSON ensures consistency while fallback parsing maintains compatibility with simpler APIs.

### Why Per-Country Processing?
Climate policy documents are inherently country-specific. Processing by country ensures clear attribution and enables parallel processing for scale. 