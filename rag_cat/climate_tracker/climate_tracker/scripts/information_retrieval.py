import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Path setup: START ---
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))
# --- Path setup: END ---

load_dotenv()

from climate_tracker.climate_tracker.my_logging import get_logger
from climate_tracker.climate_tracker.models import get_db_session, CountryModel, CountryPageSectionModel
from climate_tracker.climate_tracker.embedding_utils import BAAIEmbedder

logger = get_logger(__name__)

PREDEFINED_QUESTIONS = [
    "Does the country have a net zero target, and if so, what year is the target set for?",
    "Does the country have a multi-sector climate strategy that sets quantified sector-specific emission targets or projections for key sectors like Electricity, Transport, Industry, LULUCF/Agriculture, and any other fifth sector with significant emissions?",
    "Does the country have an energy efficiency law or a strategic framework for national energy efficiency, AND has it set an energy efficiency target (economy-wide or sectoral)?",
    "Has the country set a net zero electricity target aligned with 1.5\u00b0C (e.g., by 2035 for high-income, 2040 for China, 2045 for rest of world), or an equivalent economy-wide net zero commitment?",
    "Does the country have a carbon pricing mechanism in place (e.g., carbon tax or emissions trading system)?"
]

def chunk_text(text, chunk_size=3, overlap=1):
    if not text:
        return []
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return [c for c in chunks if c.strip()]

def retrieve_and_format_answers(output_dir_name="retrieved_country_reports_v2_chunked"):
    logger.info("Starting information retrieval process (chunking sections, on-the-fly chunk embeddings)...")
    
    output_path = _project_root / output_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to directory: {output_path}")

    model_path = os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3")
    embedder = BAAIEmbedder(model_path)
    logger.info(f"Initialized embedder with model: {model_path}.")

    db_session = get_db_session(os.getenv("DATABASE_URL"))
    countries = db_session.query(CountryModel).all()
    logger.info(f"Fetched {len(countries)} countries from 'countries_v2' table for processing.")

    for country_data in countries:
        if country_data.doc_id != "mexico":
            continue

        country_doc_id = country_data.doc_id
        country_name = country_data.country_name
        logger.info(f"Processing country: {country_name} ({country_doc_id})")

        country_markdown_parts = [f"## Country: {country_name}\n\n---\n"]
        
        sections = db_session.query(CountryPageSectionModel).filter(
            CountryPageSectionModel.country_doc_id == country_doc_id
        ).all()

        valid_chunks, urls, titles = [], [], []

        for section in sections:
            if not section.text_content:
                continue
            chunks = chunk_text(section.text_content)
            for c in chunks:
                valid_chunks.append(c)
                urls.append(section.section_url or country_data.country_url)
                titles.append(section.section_title)

        if not valid_chunks:
            logger.warning(f"No valid chunks for {country_name}. Skipping.")
            country_markdown_parts.append("\n*No text chunks could be generated.*\n\n---\n")
            with open(output_path / f"{country_doc_id}_report.md", "w") as f:
                f.write("".join(country_markdown_parts))
            continue

        logger.info(f"Embedding {len(valid_chunks)} chunks for {country_name}...")

        embeddings = []
        for chunk in valid_chunks:
            try:
                emb = embedder.encode_batch([chunk])[0]
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Skipping chunk due to error: {e}")

        if not embeddings:
            logger.warning(f"No embeddings for {country_name}.")
            country_markdown_parts.append("\n*Failed to generate embeddings.*\n\n---\n")
            with open(output_path / f"{country_doc_id}_report.md", "w") as f:
                f.write("".join(country_markdown_parts))
            continue

        embeddings_np = np.array(embeddings)
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        for i, question in enumerate(PREDEFINED_QUESTIONS):
            country_markdown_parts.append(f"### Question {i+1}: {question}\n\n")
            try:
                q_emb = embedder.encode_batch([question])[0]
                sims = cosine_similarity(q_emb.reshape(1, -1), embeddings_np)
                idx = np.argmax(sims)
                score = sims[0, idx]
                text = valid_chunks[idx].replace('`', '\\`').replace('\n', ' ')
                url = urls[idx]
                
                country_markdown_parts.append(f"**Answer/Evidence (Similarity: {score:.4f}):**\n")
                country_markdown_parts.append(f"> {text}\n\n")
                country_markdown_parts.append(f"**Source URL:** [{url}]({url})\n\n---\n")
            except Exception as e:
                logger.warning(f"Embedding question failed: {e}")
                country_markdown_parts.append(f"\n*Failed to process question.*\n\n---\n")

        with open(output_path / f"{country_doc_id}_report.md", "w") as f:
            f.write("".join(country_markdown_parts))

    db_session.close()
    logger.info(f"✅ Information retrieval process complete. Reports saved in {output_path}")

if __name__ == "__main__":
    retrieve_and_format_answers()
