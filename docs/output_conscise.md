# Output Module Documentation

---

## 📘 Overview

The `6_output.py` script converts structured LLM response JSON files into professional PDF fact sheets. It's the final stage of our RAG pipeline—taking analyzed climate policy data and producing client-ready reports.

**What it does**: Finds JSON files in `data/llm/`, extracts country-specific content, and generates styled PDFs with metadata tables, Q&A sections, and source citations.

---

## 🔁 How It Works

### Core Pipeline

```mermaid
flowchart LR
    A[Find JSON Files] --> B[Extract Country Data]
    B --> C[Build PDF Elements]
    C --> D[Apply Styling]
    D --> E[Save PDF Report]
```

### Key Functions

| Function | Purpose | Input/Output |
|----------|---------|--------------|
| `get_available_llm_files()` | Discovers JSON files | Returns list of file paths |
| `load_llm_response_data()` | Parses JSON & extracts country | JSON path → country name + data |
| `create_pdf_style()` | Defines ReportLab styles | None → StyleSheet with colors |
| `generate_pdf_report()` | Assembles final PDF | Data + styles → PDF file |

### Expected JSON Structure

The script expects this structure in `data/llm/*.json`:

```json
{
  "metadata": {
    "country_name": "Germany",
    "timestamp": "2024-01-01 12:00:00"
  },
  "questions": {
    "question_1": {
      "llm_response": {
        "question": "What are the key climate commitments?",
        "answer": {
          "summary": "Brief overview...",
          "detailed_response": "Detailed analysis..."
        },
        "citations": [
          {
            "content": "Source text...",
            "country": "Germany",
            "cos_similarity_score": 0.85
          }
        ]
      }
    }
  }
}
```

**Error Handling**: The script handles malformed JSON strings within the data structure—common with LLM outputs—by attempting multiple parsing strategies before falling back to safe defaults.

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