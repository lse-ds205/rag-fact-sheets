# Climate Policy RAG System

A comprehensive RAG (Retrieval-Augmented Generation) pipeline for extracting and analyzing climate policy information from the Climate Action Tracker website to generate structured country-specific fact sheets for policy analysts.

## Project Overview

This project builds upon previous NDC (Nationally Determined Contributions) extraction work to create an intelligent information retrieval system that helps policy analysts access relevant climate policy facts from Climate Action Tracker. The system follows the ASCOR methodology framework to identify and extract key climate policy indicators for informed assessment.

### Key Features

- Web Scraping Pipeline: Automated extraction of climate policy data from Climate Action Tracker
- Semantic Search: BGE-M3 embeddings for intelligent information retrieval
- Fact Sheet Generation: Automated creation of structured Markdown reports
- Multi-dimensional Analysis: Country-specific and thematic policy analysis
- Citation Support: All facts include source URLs for verification

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Scraping  │───▶│   Data Storage  │───▶│   Embedding     │
│   (Scrapy)      │    │   (PostgreSQL)  │    │   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             ▼
│   Fact Sheets   │◀───│   Information   │    ┌─────────────────┐
│   (Markdown)    │    │   Retrieval     │◀───│   Vector Search │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Policy Target Extraction & Evaluation                   │
│  - LLM-based Extraction                                 │
│  - Precision/Recall/F1 Metrics                          │
└─────────────────────────────────────────────────────────┘
        │                                           ▲
        │                                           │
        │                                    ┌──────┴───────┐
        │                                    │  Ground      │
        │                                    │  Truth       │
        │                                    └──────────────┘
        │
        ├─────────────────┐
        │                 │
        ▼                 ▼
┌─────────────────┐    ┌─────────────────┐
│  QA Boxes       │    │  Visualization  │
│  Generation     │    │  (t-SNE/Heatmap)│
└─────────────────┘    └─────────────────┘
```

## Repository Structure

```
climate_tracker/
├── climate_tracker/                    # Main package directory
│   ├── spiders/                       # Scrapy web scraping spiders
│   │   ├── __init__.py
│   │   └── climate_action_tracker.py
│   │
│   ├── scripts/                       # Data processing scripts
│   │   ├── init-db.py             # Database initialization
│   │   ├── create-table.py        # Table creation
│   │   ├── store.py              # Data storage
│   │   ├── generate_embeddings.py # Embedding generation
│   │   ├── information_retrieval.py # RAG pipeline
│   │   ├── policy_extraction.py   # Policy target extraction
│   │   ├── tsne_and_heatmap.py   # Data visualization
│   │   ├── evaluate_extraction.py # Extraction evaluation
│   │   └── qa_boxes.py           # QA box generation
│   │
│   ├── data/                         # Data directories
│   │   └── full_text/               # Scraped data outputs
│   │       ├── MD/                  # Markdown format (legacy)
│   │       ├── structured/          # JSON structured data (legacy)
│   │       └── unstructured/        # Raw text data (legacy)
│   │
│   ├── items.py                     # Scrapy data models
│   ├── models.py                    # SQLAlchemy database models
│   ├── middlewares.py              # Scrapy middleware components
│   ├── pipelines.py                # Data processing pipelines
│   ├── settings.py                 # Scrapy configuration
│   ├── embedding_utils.py          # BGE-M3 embedding utilities
│   ├── my_logging.py              # Custom logging configuration
│   └── utils.py                   # Utility functions
│
├── policy_targets_pages/              # Policy target analysis pages
│
├── retrieved_country_reports_v2_chunked/  # Generated fact sheets
│
├── output/                              # Output directory
│   ├── policy_targets_output.json       # Extracted policy targets
│   ├── tsne_plot.png                    # t-SNE visualization
│   ├── heatmap.png                      # Similarity heatmap
│   ├── confidence_heatmap.png          # Confidence scores heatmap
│   ├── country_f1_scores.png           # Country-level F1 score visualization
│   ├── label_f1_scores.png             # Label-level F1 score visualization
│   ├── evaluation_summary.md           # Summary of evaluation metrics
│   ├── ground_truth_template.json      # Template for ground truth data
│   ├── low_confidence_summary.csv      # Summary of low confidence predictions
│   ├── qa_boxes.md                     # Generated QA boxes in markdown format
│   └── evaluation_metrics.json          # Extraction evaluation results
│
├── reports/                             # Generated reports
│   ├── QA_Boxes_Report.md              # QA boxes report
│   ├── Evaluate_Policy_Extraction.md    # Policy extraction evaluation report
│   ├── Generate_Embeddings.md          # Embedding generation report
│   ├── Information_Retrieval.md        # Information retrieval report
│   ├── Introduction.md                 # Project introduction
│   ├── Policy_Extraction.md            # Policy extraction report
│   └── Scripts_01-03_Report.md         # Initial scripts report
│
├── tasks.py                            # Management CLI commands
├── scrapy.cfg                         # Scrapy project configuration
├── requirements.txt                   # Python dependencies
├── README.md                         # Main project README
├── CLIMATEREADME.md                  # Climate-specific documentation
└── LICENSE                           # Project license file
```

## Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- Git

### Installation

#### Step 1
Clone the repository

```bash
git clone "https://github.com/MaxL75/DS205-final-project-team-CPR.git"
cd DS205-final-project-team-CPR
```

#### Step 2
Set up and activate a virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

#### Step 3
Install dependencies

```bash
pip install -r requirements.txt
```

B) And to complete the setup, make sure the English SpaCy model is installed (this isn't handled by requirements.txt directly). After installing dependencies, run this:
```bash
python -m spacy download en_core_web_sm
```

#### Step 4
Install PostgreSQL and `pgvector` Extension (if not already installed)

If you're on **macOS** and don't already have [Homebrew](https://brew.sh/) installed, you can install it by running:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then, install PostgreSQL and the pgvector extension:

```bash
brew install postgresql
brew install pgvector
```
Once PostgreSQL is installed and running, connect to your database using the psql command-line tool:
```bash
psql -U postgres
```

Then, enable the vector extension in your database by running:
```bash
CREATE EXTENSION IF NOT EXISTS vector;
```

Then exit:
```bash 
\q
```

This enables vector similarity search functionality used by the project.

#### Step 5
Set up environment variables

Create a .env file from the provided example:

```bash
cp .env.example .env
```

Then edit the .env file with your actual PostgreSQL credentials and model path. For example:
```bash
# Example if your database is named climate_db:
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/climate_db
EMBEDDING_MODEL=models/local/BAAI-bge-m3
CHUNK_DIR=data/full_text/structured
```

---

### Running the Pipeline

First run the scrapy crawl: 

```bash
### Extracting the data: Run this script to extract the data from the websites, output in MD, Structured and Unstructured Format.
cd climate_tracker
scrapy crawl climate_action_tracker_fulltext
```

---
### Running the Pipeline (Option A - using `climate_tracker/tasks.py`)

Can use the tasks.py CLI to manage and execute each step of the pipeline.

This project includes a CLI management tool in tasks.py that lets you run all key steps in the Climate Action Tracker pipeline.

#### Step 1: View available commands
To see all available tasks:
```bash
PYTHONPATH=. python -m climate_tracker.tasks --help 
```

You should see a list of commands like:
```sql
Commands:
  create_table           Step 2: Create the `countries` table.
  drop_db                Drop all tables from the database.
  evaluate               Step 8: Evaluate policy extraction results.
  generate_embeddings    Step 4: Generate embeddings for document sections.
  information_retrieval  Step 5: Run information retrieval pipeline.
  init_db                Step 1: Initialize the database with vector...
  list_tables            List all tables in the database.
  policy_extraction      Step 6: Extract policy targets.
  qa_boxes               Step 9: Generate QA boxes.
  recreate_db            Recreate all database tables (destructive).
  store                  Step 3: Store extracted text into the database.
  visualize              Step 7: Generate TSNE and heatmap visualizations.
```

#### Step 2: Run a specific task

Each command can be run individually, for example:
```bash
# Initialize the database
PYTHONPATH=. python -m climate_tracker.tasks -- init_db

# Store text into the database
PYTHONPATH=. python -m climate_tracker.tasks -- store

# Generate document embeddings
PYTHONPATH=. python -m climate_tracker.tasks -- generate_embeddings

```

---

### Running the Pipeline (Option B - running each script individually)


#### Step 1: Initialize the database

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/init_db.py
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/create_table.py
```

#### Step 2: Data Storage
Assuming you're in the project root: 

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/store.py
```

#### Step 3: Generate Embeddings

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/generate_embeddings.py
```
For one country: 
```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/generate_embeddings.py --only-country mexico
```

#### Step 4: Generate Fact Sheets (Information Retrieval)

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/information_retrieval.py```
```

#### Step 5: Run Refined Policy Extraction
Assuming your markdown reports are saved in retrieved_country_reports_v2_chunked:

```bash
# All countries
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/policy_extraction.py --save
# Specific country (e.g., Nigeria)
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/policy_extraction.py --country nigeria --save
```

#### Step 6: t-SNE and Heatmap Plot 
These plots are based on the similarity scores from the refined policy extraction in Step 6

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/tsne_and_heatmap.py
```

#### Step 7: Running Metrics Between Ground Truth and Predicted Model 
```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/evaluate_extraction.py
```

#### Step 8: Q&A Boxes

```bash
PYTHONPATH=. python climate_tracker/climate_tracker/scripts/qa_boxes.py
```

---


## Sample Outputs


### Sample Country Report Structure (Factsheet)

The system generates comprehensive fact sheets for each country, addressing five key policy questions: 
* "Does the country have a net zero target, and if so, what year is the target set for?"
* "Does the country have a multi-sector climate strategy that sets quantified sector-specific emission targets or projections for key sectors like Electricity, Transport, Industry, LULUCF/Agriculture, and any other fifth sector with significant emissions?"
* "Does the country have an energy efficiency law or a strategic framework for national energy efficiency, AND has it set an energy efficiency target (economy-wide or sectoral)?"
* "Has the country set a net zero electricity target aligned with 1.5°C (e.g., by 2035 for high-income, 2040 for China, 2045 for rest of world), or an equivalent economy-wide net zero commitment?"
* "Does the country have a carbon pricing mechanism in place (e.g., carbon tax or emissions trading system)?"


```markdown

## Country: Brazil

---

### Question 1: Does the country have a net zero target, and if so, what year is the target set for?

**Answer/Evidence (Similarity: 0.8234):**

> Brazil has committed to achieving net zero greenhouse gas emissions by 2050, as outlined in its updated NDC submission...

**Source URL:** [https://climateactiontracker.org/countries/brazil/targets/](https://climateactiontracker.org/countries/brazil/targets/)
```


### Sample Country Report Structure (Policy Extraction)

The policy extraction stage processes the retrieved information from the factsheets, to generate structured, interpretable answers to key climate policy questions (and using more advanced NLP techniques).


```markdown

Net Zero Target

- **Answer**: `yes`

- **Explanation**: Mentions net zero target.

- **Year(s)**: 2022, 2050

- **Confidence**: 0.8189

- **Source URL**: https://climateactiontracker.org/countries/argentina/targets/

> Answer/Evidence (Similarity: 0.8251): Further information on how the CAT rates countries (against modelled domestic pathways and fair share) can be found here. **    ## Net zero and other long-term target(s)   We evaluate the net zero target as: Poor. **   In November 2022, Argentina submitted...

```


### Sample Country Report Structure (Q&A Boxes)

Nigeria – Sector Targets

**Question:** What sector-specific climate targets has Nigeria implemented?  
**Answer:** ✅ Yes  
**Explanation:** Mentions sector-specific targets.  
**Target Year(s):** 2021, 2023, 2030, 2060  
**Sectors:** Sectoral targets

**Quote:**  
> No quote provided.

**Confidence:** 82.53%  
**Source:** [Link](https://climateactiontracker.org/countries/nigeria/policies-action/)

----

## Data Output Structure

The scraping system generates data in three formats to support different analysis approaches:

#### 1. Structured Database Storage (Primary Method)

- **Purpose**: Optimized for RAG pipeline with semantic search capabilities
- **Location**: PostgreSQL database with pgvector extension
- **Structure**:
  - `countries_v2` table: Country metadata and main URLs
  - `country_page_sections_v2` table: Individual sections with embeddings
- **Advantage**: Direct semantic search, proper normalization, embedding storage

#### 2. Markdown Files (Human-Readable Format)

- **Purpose**: Human-readable format for manual inspection and verification
- **Location**: `data/full_text/MD/` directory
- **Structure**: Individual .md files per country with section headings
- **Use Case**: Quality assurance, manual fact-checking, documentation

#### 3. JSON Outputs (Legacy Formats)

- **Structured JSON** (`data/full_text/structured/`): Section-organized data
- **Unstructured JSON** (`data/full_text/unstructured/`): Concatenated text blocks
- **Purpose**: Backwards compatibility and alternative processing workflows

---

## Key Analytical Questions

The system addresses five critical policy assessment questions:

1. **Net Zero Targets**: Timeline and commitment analysis
2. **Multi-sector Strategies**: Sectoral emission targets and projections
3. **Energy Efficiency**: Laws, frameworks, and targets
4. **Electricity Decarbonization**: Net zero electricity commitments
5. **Carbon Pricing**: Existing mechanisms and policies

---

## Technical Implementation

### Data Processing Pipeline

1. **Web Scraping**: Scrapy spider extracts structured content from Climate Action Tracker
2. **Data Storage**: PostgreSQL with pgvector for efficient vector operations
3. **Text Chunking**: NLTK-based sentence segmentation for optimal context windows
4. **Embedding Generation**: BGE-M3 model for multilingual semantic understanding
5. **Similarity Search**: Cosine similarity for relevant information retrieval

---

### Database Schema

The system uses a normalized PostgreSQL schema optimized for semantic search:

```sql
-- Countries table (countries_v2)
CREATE TABLE countries_v2 (
    doc_id VARCHAR PRIMARY KEY,              -- e.g., "argentina"
    country_name VARCHAR NOT NULL,           -- "Argentina"
    country_url VARCHAR NOT NULL UNIQUE,    -- Main country page URL
    language VARCHAR DEFAULT 'en',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_scraped_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sections table (country_page_sections_v2) 
CREATE TABLE country_page_sections_v2 (
    id SERIAL PRIMARY KEY,
    country_doc_id VARCHAR REFERENCES countries_v2(doc_id),
    section_title VARCHAR NOT NULL,         -- "Summary", "Targets", etc.
    section_url VARCHAR NOT NULL UNIQUE,   -- Specific section URL
    text_content TEXT,                      -- Scraped text content
    embedding VECTOR(1024),                 -- BGE-M3 embeddings (pgvector)
    language VARCHAR DEFAULT 'en',
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);
```