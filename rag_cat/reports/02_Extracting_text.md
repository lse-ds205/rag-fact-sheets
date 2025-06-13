# Climate Action Tracker Scraper Report

**Author**: *[Your Name]*  
**Spider Name**: `climate_action_tracker_fulltext`  
**Version**: 1.0  
**Date**: {{DATE}}  
**Framework**: Scrapy  

---

## ✅ Project Initialization

- Created a clean project structure for the CAT scraping pipeline under the `climate_tracker` module.
- Wrote and structured the initial `README.md` to describe the project goal, scraping targets, and how to run the scraper.
- Set up version tracking (`version = "1.0"`) within the spider to maintain reproducibility.

---

## 🎯 Purpose

This Scrapy spider extracts full-text climate policy data for each country listed on [climateactiontracker.org](https://climateactiontracker.org). It collects structured and unstructured content from multiple sections (e.g., Summary, Targets, Policies) and saves them in Markdown and JSON formats for downstream use in analytics or NLP pipelines.

---

## 🕸️ Web Scraping with Scrapy

- Built a custom `ClimateActionTrackerSpider` to crawl and parse country-specific pages from the CAT website.
- Identified and followed structured navigation paths (`/countries/{country_name}` and sub-sections like "Policies & Actions", "Targets", etc.).
- Parsed HTML content into plain text while preserving semantic structure using Markdown-style formatting (e.g., `<b>` tags → `**text**`, headers → `##` or `###`).
- Implemented custom logic to prioritize and sort sections in a predefined order:
  ```python
  ["Summary", "Targets", "Policies & Actions", "Net Zero Targets", "Assumptions", "Sources"]
  ```

---

## 📂 Output Files

- **Markdown Files**: `data/full_text/MD/{country}.md` — readable summaries by section.
- **Structured JSON**: `data/full_text/structured/{country}.json` — nested by section.
- **Unstructured JSON**: `data/full_text/unstructured/{country}.json` — plain paragraph list with section headers and metadata.

---

## 🗂️ Data Output and Storage

Designed and implemented three output formats per country:

- Markdown files (`.md`) with hierarchical headers per section.
- Structured JSON with `country_name`, `sections`, and associated `urls`.
- Unstructured JSON with all text content as a continuous Markdown-style blob, plus metadata (`timestamp`, `version`).

Created separate storage directories for each format:

```
data/full_text/MD
data/full_text/structured
data/full_text/unstructured
```

---

## 🔧 Pipeline Support

- Created `items.py` for future item definition (currently unused but ready for pipeline integration).
- Built the foundation in `pipelines.py` for post-processing or database storage pipelines if needed.
- Added a `closed(self, reason)` hook to re-export all country data at the end of the crawl (for safety and completeness).

---

## 📌 Section Handling

The spider processes the following sections:
- Summary (parsed directly from main country page)
- Targets
- Policies & Actions
- Net Zero Targets
- Assumptions
- Sources

Other sections are included if discovered.

---

## ✅ Example Entry (Structured JSON)

```json
{
  "country_name": "Canada",
  "sections": {
    "Summary": {
      "content": ["Canada has committed to..."],
      "url": "https://climateactiontracker.org/countries/canada/"
    }
  }
}
```

---

## ⚠️ Edge Case Handling

- If section text contains bold tags, they're converted to Markdown.
- Missing content is safely skipped.
- Redundant country entries are deduplicated via slug.
- Directories are created automatically if missing.

---

## 🏁 Final Notes

This spider enables reproducible collection of policy-relevant text data for climate analysis, suitable for downstream NLP modeling or search pipelines. Further extensions could add:
- PDF/CSV export
- Language detection
- Named entity recognition tagging
