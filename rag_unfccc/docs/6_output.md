# Output Module Documentation

---

## 📘 Overview

The `6_output.py` script converts structured LLM response JSON files into professional PDF fact sheets. It's the final stage of our RAG pipeline—taking analyzed climate policy data and producing client-ready reports.

**What it does**: Finds JSON files in `data/llm/`, extracts country-specific content, and generates styled PDFs with metadata tables, Q&A sections, and source citations.

---

## 🔁 How It Works

### Key Functions

| Function | Purpose | Input/Output |
|----------|---------|--------------|
| `get_available_llm_files()` | Discovers JSON files | Returns list of file paths |
| `load_llm_response_data()` | Parses JSON & extracts country | JSON path → country name + data |
| `create_pdf_style()` | Defines ReportLab styles | None → StyleSheet with colors |
| `generate_pdf_report()` | Assembles final PDF | Data + styles → PDF file |

---

## Input Format

### Expected JSON Structure

The module expects JSON files in the `data/llm/` directory with this structure:

```mermaid
graph TD
    A[LLM JSON File] --> B[metadata]
    A --> C[questions]
    
    B --> B1[country_name]
    B --> B2[timestamp]
    B --> B3[other metadata]
    
    C --> D[question_1]
    C --> E[question_2]
    C --> F[question_N]
    
    D --> G[llm_response]
    G --> H[question]
    G --> I[answer]
    G --> J[citations]
    
    I --> K[summary]
    I --> L[detailed_response]
    
    J --> M[Citation 1]
    J --> N[Citation 2]
    
    M --> O[content]
    M --> P[country]
    M --> Q[how_used]
    M --> R[cos_similarity_score]
```

### JSON Processing Flow

```mermaid
flowchart LR
    A[Raw JSON] --> B{Parse Metadata}
    B --> C[Extract Country Name]
    C --> D{Process Questions}
    D --> E[Parse Answer Structure]
    E --> F{Handle Malformed JSON}
    F -->|Clean Data| G[Format for PDF]
    F -->|Parse Errors| H[Apply Fallback]
    G --> I[Generate PDF Elements]
    H --> I
```

### Why This Input Structure?

The structure accommodates the output from our LLM analysis stage while being flexible enough to handle malformed JSON strings within the data. The metadata extraction allows for country-specific report generation, while the nested question structure supports multi-question fact sheets.

**Error Handling**: The script handles malformed JSON strings within the data structure—common with LLM outputs—by attempting multiple parsing strategies before falling back to safe defaults.

---

## PDF Generation Pipeline

```mermaid
flowchart TD
    A[Start Main Function] --> B[Setup Directory]
    B --> C[Find JSON Files]
    C --> D[Create Styles]
    D --> E[Process Each File]
    
    E --> F[Load JSON Data]
    F --> G[Extract Country]
    G --> H[Build PDF Elements]
    
    H --> I[Add Title]
    I --> J[Add Metadata]
    J --> K[Add Answers]
    K --> L[Add Citations]
    L --> M[Save PDF]
    
    M --> N{More Files?}
    N -->|Yes| E
    N -->|No| O[Complete]
```

### Processing Modules

```mermaid
graph LR
    A[File Discovery] --> B[Data Loading]
    B --> C[Style Creation]
    C --> D[Content Generation]
    D --> E[PDF Assembly]
```

### Element Generation Order

1. **Title Section**: Country-specific header with styled title
2. **Metadata Table**: Summary information (question count, citations, timestamp)
3. **Answer Sections**: Each question processed with summary/detailed breakdown
4. **Citations Section**: Grouped by country with relevance scores

This order ensures logical document flow while allowing for flexible content based on available data.

---

## 🚀 How to Use It

### Basic Usage

```bash
python entrypoints/6_output.py
```

This processes all JSON files in `data/llm/` and generates corresponding PDFs in `outputs/factsheets/`.

### Output Format

**Filename**: `{Country}_climate_policy_factsheet_{timestamp}.pdf`

**Example**: `Germany_climate_policy_factsheet_20241201_143022.pdf`

### PDF Contents

Each report contains:
- **Title**: Country-specific header
- **Metadata Table**: Question count, citations, generation timestamp  
- **Q&A Sections**: Structured answers with summaries and detailed analysis
- **Citations**: Grouped by country with relevance scores

### Expected Output

```
[6_OUTPUT] Starting PDF generation from LLM response files...
[6_OUTPUT] Found 3 LLM response files
[6_OUTPUT] Processing: germany_responses.json
[6_OUTPUT] Generating PDF report for Germany...
[6_OUTPUT] PDF report generated: Germany_climate_policy_factsheet_20241201_143022.pdf
```

**Integration**: Returns the output directory path for use in automated pipelines.

---

That's it. The script handles the rest—styling, error recovery, and file management—automatically. 
