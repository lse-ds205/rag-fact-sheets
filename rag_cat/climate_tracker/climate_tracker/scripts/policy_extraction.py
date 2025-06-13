#!/usr/bin/env python3
"""
06_policy_extraction.py

Refined extraction component that processes pre-generated Markdown reports
(from 05_information_retrieval.py) to extract structured policy info for evaluation.
Includes confidence scores from cosine similarity.
Also saves per-country Markdown reports in `output/policy_targets_pages/`.
"""
import os
import re
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import logging
import spacy
import json
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Set paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "retrieved_country_reports_v2_chunked"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_JSON = OUTPUT_DIR / "policy_targets_output.json"
MARKDOWN_OUTPUT_DIR = PROJECT_ROOT / "policy_targets_pages"
MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Mapping
QUESTION_INDEX_MAPPING = {
    0: "net_zero_target",
    1: "sector_targets",
    2: "efficiency_target",
    3: "electricity_net_zero",
    4: "carbon_pricing"
}

PREDEFINED_QUESTIONS = [
    "Does the country have a net zero target, and if so, what year is the target set for?",
    "Does the country have a multi-sector climate strategy that sets quantified sector-specific emission targets or projections for key sectors like Electricity, Transport, Industry, LULUCF/Agriculture, and any other fifth sector with significant emissions?",
    "Does the country have an energy efficiency law or a strategic framework for national energy efficiency, AND has it set an energy efficiency target (economy-wide or sectoral)?",
    "Has the country set a net zero electricity target aligned with 1.5°C (e.g., by 2035 for high-income, 2040 for China, 2045 for rest of world), or an equivalent economy-wide net zero commitment?",
    "Does the country have a carbon pricing mechanism in place (e.g., carbon tax or emissions trading system)?"
]

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Enhanced tagging using spacy entities and rule-based heuristics
def tag_entities(text):
    tags = {}
    doc = nlp(text)

    extracted_years = set()
    regex_years = re.findall(r"\b(20\d{2})\b", text)
    match_phrases = re.findall(r"(?:by|target year|before|achieve by)\s+(20\d{2})", text, re.I)
    if re.search(r"\bmid[-\s]?century\b", text, re.I):
        extracted_years.add("2050")
    ner_years = {ent.text.strip() for ent in doc.ents if ent.label_ == "DATE" and re.match(r"20\d{2}$", ent.text.strip())}
    extracted_years.update(regex_years + match_phrases)
    extracted_years.update(ner_years)
    if extracted_years:
        tags["YEAR"] = sorted(extracted_years)

    sector_keywords = r"\b(electricity|power|transport|mobility|industry|industrial|agriculture|agricultural|lulucf|energy|sectoral targets)\b"
    sectors = re.findall(sector_keywords, text, re.I)
    if sectors:
        tags["SECTORS"] = sorted({s.lower().capitalize() for s in sectors})
        tags["SECTOR"] = True

    if any(ent.label_ == "LAW" or ent.text.lower() in ["climate act", "energy act"] for ent in doc.ents):
        tags["POLICY_TYPE"] = True

    if re.search(r"net zero|carbon neutrality", text, re.I):
        tags["TARGET"] = "net zero"
    if re.search(r"strategy|framework|law|policy|act", text, re.I):
        tags["POLICY_TYPE"] = True

    return tags

def extract_structured_info(label, text, question_embedding, embedder, source_url):
    tags = tag_entities(text)
    doc = nlp(text)
    negation = any(tok.dep_ == "neg" for tok in doc)

    base = {
        "yes_no": "no",
        "explanation": None,
        "year": tags.get("YEAR"),
        "sectors": tags.get("SECTORS"),
        "quote": text[:300],
        "confidence": None,
        "source_url": source_url
    }

    try:
        chunk_embedding = embedder.encode_batch([text])[0]
        confidence = float(cosine_similarity(question_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0])
        base["confidence"] = round(confidence, 4)
        if confidence < 0.7:
            base["yes_no"] = "low_confidence"
            base["explanation"] = "Low semantic similarity."
            return base
    except Exception as e:
        logger.warning(f"Could not compute confidence score: {e}")

    if label == "net_zero_target":
        if "TARGET" in tags:
            base["yes_no"] = "yes"
            base["explanation"] = "Mentions net zero target."

    elif label == "sector_targets":
        if "SECTOR" in tags:
            base["yes_no"] = "yes"
            base["explanation"] = "Mentions sector-specific targets."

    elif label == "efficiency_target":
        mentions_efficiency = any(token.lemma_ in ["efficiency", "energy", "saving", "reduce"] for token in doc)
        mentions_policy = "POLICY_TYPE" in tags
        mentions_target = re.search(r"\b(target|goal|objective|reduce emissions by|save energy|%|percent)\b", text, re.I)

        if mentions_policy and mentions_efficiency:
            if mentions_target:
                base["yes_no"] = "yes"
                base["explanation"] = "Mentions energy efficiency law or strategy and sets an efficiency target."
            else:
                base["yes_no"] = "soft_yes"
                base["explanation"] = "Mentions energy efficiency law or strategy, but target is unclear or missing."
        elif mentions_efficiency:
            base["yes_no"] = "soft_yes"
            base["explanation"] = "Mentions energy efficiency themes, but no clear policy or target."

    elif label == "electricity_net_zero":
        text_lower = text.lower()
        if "net zero" in text_lower and "scenario" in text_lower and re.search(r"electricity|power|grid|renewable energy", text_lower):
            base["yes_no"] = "soft_yes"
            base["explanation"] = "Only scenario-based evidence for net zero electricity."
        elif "net zero" in text_lower and re.search(r"electricity|power|grid|renewable energy", text_lower):
            base["yes_no"] = "yes"
            base["explanation"] = "Mentions net zero electricity or decarbonised power sector."

    elif label == "carbon_pricing":
        if re.search(r"carbon (tax|pricing|trading|ETS|market)", text, re.I):
            base["yes_no"] = "yes"
            base["explanation"] = "Mentions carbon pricing mechanism."

    if negation and base["yes_no"] in ["yes", "soft_yes"]:
        base["explanation"] += " (Note: negation present in sentence, which may weaken the claim.)"

    return base

def extract_answer_blocks_from_md(md_file_path):
    with open(md_file_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
    soup = BeautifulSoup(html, "html.parser")
    q_blocks = soup.find_all("h3")
    extracted_answers = []
    for q_block in q_blocks:
        question_text = q_block.text.strip()
        content = []
        source_url = None
        for sibling in q_block.find_next_siblings():
            if sibling.name == "h3":
                break
            text = sibling.text.strip()
            if text.lower().startswith("source url:"):
                match = re.search(r"https?://\S+", text)
                if match:
                    source_url = match.group(0)
            else:
                content.append(text)
        combined = "\n".join(content).strip()
        extracted_answers.append((question_text, combined, source_url))
    return extracted_answers

def extract_policies_from_answers(answer_blocks, embedder):
    structured = {}
    for i, (question, answer_text, source_url) in enumerate(answer_blocks):
        label = QUESTION_INDEX_MAPPING.get(i)
        question_embedding = embedder.encode_batch([PREDEFINED_QUESTIONS[i]])[0]
        structured[label] = extract_structured_info(label, answer_text, question_embedding, embedder, source_url)
    return structured

def write_country_markdown(country_id, structured):
    lines = [f"# Policy Targets for {country_id.title()}\n"]
    for label, entry in structured.items():
        label_title = label.replace("_", " ").title()
        lines.append(f"## {label_title}")
        lines.append(f"- **Answer**: `{entry.get('yes_no', 'N/A')}`")
        lines.append(f"- **Explanation**: {entry.get('explanation', '')}")
        if entry.get("year"):
            lines.append(f"- **Year(s)**: {', '.join(entry['year'])}")
        if entry.get("sectors"):
            lines.append(f"- **Sector(s)**: {', '.join(entry['sectors'])}")
        lines.append(f"- **Confidence**: {entry.get('confidence', 'N/A')}")
        lines.append(f"- **Source URL**: {entry.get('source_url', 'N/A')}")
        lines.append(f"> {entry.get('quote', '').replace(chr(10), ' ')}\n")
    out_path = MARKDOWN_OUTPUT_DIR / f"{country_id.lower().replace(' ', '_')}_policy_targets.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))

def run_policy_extraction(country_filter=None):
    from climate_tracker.climate_tracker.embedding_utils import BAAIEmbedder
    embedder = BAAIEmbedder(os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3"))

    results = {}
    all_md_files = sorted(REPORT_DIR.glob("*_report.md"))
    if country_filter:
        all_md_files = [f for f in all_md_files if country_filter.lower() in f.name.lower()]

    if not all_md_files:
        logger.warning("No matching markdown files found for filtering.")
        return

    for md_file in all_md_files:
        country_id = md_file.stem.replace("_report", "")
        logger.info(f"Processing {country_id}...")
        answer_blocks = extract_answer_blocks_from_md(md_file)
        structured = extract_policies_from_answers(answer_blocks, embedder)
        results[country_id] = structured
        write_country_markdown(country_id, structured)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run refined policy extractor")
    parser.add_argument("--country", type=str, default=None, help="Run extraction on one country (partial name)")
    parser.add_argument("--save", action="store_true", help="Save output as JSON file")
    args = parser.parse_args()

    output = run_policy_extraction(country_filter=args.country)
    print(json.dumps(output, indent=2))

    if args.save:
        try:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Successfully saved output to {OUTPUT_JSON}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
