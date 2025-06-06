import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text, and_
from tqdm import tqdm


# --- Path setup: START ---
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root))
# --- Path setup: END ---

load_dotenv()

from climate_tracker.climate_tracker.my_logging import get_logger
logger = get_logger(__name__)
from climate_tracker.climate_tracker.models import Base, get_db_session, CountryPageSectionModel
from climate_tracker.climate_tracker.embedding_utils import BAAIEmbedder

print("🟢 Script is running...")


# Define a zero-vector matching your embedding dimension (e.g. 1024)
ZERO_VECTOR = [0.0] * 1024

def generate_embeddings(batch_size=5, max_chars=30000, only_country=None):
    default_model_path = "BAAI/bge-m3"
    model_path = os.getenv("BGE_MODEL_PATH", default_model_path)
    logger.info(f"Using embedding model path: {model_path}")

    session = get_db_session(os.getenv("DATABASE_URL"))
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()

    embedder = BAAIEmbedder(model_path)

    logger.info("Fetching sections without embeddings...")

    base_filter = [CountryPageSectionModel.embedding == None]
    if only_country:
        base_filter.append(CountryPageSectionModel.country_doc_id == only_country)

    total_to_process = session.query(CountryPageSectionModel).filter(and_(*base_filter)).count()
    logger.info(f"Total sections to embed: {total_to_process}")

    processed = 0

    while True:
        sections_to_embed = session.query(CountryPageSectionModel)\
            .filter(and_(*base_filter))\
            .limit(batch_size).all()

        if not sections_to_embed:
            break

        texts, valid_sections = [], []

        for section_obj in sections_to_embed:
            if not section_obj.text_content:
                logger.warning(f"Section ID {section_obj.id} (URL: {section_obj.section_url}) has no text_content. Skipping.")
                section_obj.embedding = ZERO_VECTOR  # Mark as processed with zero-vector
                session.add(section_obj)
                continue

            trimmed_text = section_obj.text_content.strip()[:max_chars]
            if len(trimmed_text) < 20:
                logger.warning(f"Section ID {section_obj.id} (URL: {section_obj.section_url}) text_content too short after trimming. Skipping. Content: '{trimmed_text[:50]}...'")
                section_obj.embedding = ZERO_VECTOR  # Mark as processed with zero-vector
                session.add(section_obj)
                continue

            texts.append(trimmed_text)
            valid_sections.append(section_obj)

        # Commit skipped sections
        session.commit()

        if not texts:
            logger.info("No valid texts to embed in the current batch. Checking if more sections are pending...")

            # NEW SAFETY CHECK — if nothing is valid for this country, exit
            if only_country:
                logger.info(f"No valid sections with text found for country: {only_country}. Exiting early.")
                break

            # For general case, continue looping in case next batch has valid sections
            if not session.query(CountryPageSectionModel).filter(and_(*base_filter)).first():
                logger.info("No more sections found to process at all.")
                break
            continue

        logger.info(f"Embedding batch of {len(texts)} sections...")
        embeddings = []
        for i, txt in enumerate(texts):
            try:
                vec = embedder.encode_batch([txt])[0]
                embeddings.append(vec)
            except Exception as e:
                logger.warning(f"⚠ Failed to embed section index {i}: {e}")
                embeddings.append(None)

        for i, section_obj in enumerate(valid_sections):
            if embeddings[i] is None:
                logger.warning(f"Skipping section ID {section_obj.id} (URL: {section_obj.section_url}) due to embedding failure.")
                continue
            section_obj.embedding = embeddings[i].tolist()
            logger.debug(f"Generated embedding for section ID {section_obj.id} (URL: {section_obj.section_url})")

        try:
            session.commit()
            processed += len(valid_sections)
            logger.info(f"Committed embeddings for {len(valid_sections)} sections. Total processed so far: {processed}/{total_to_process}")
        except Exception as e:
            logger.error(f"Failed to commit batch of embeddings: {e}")
            session.rollback()
            break

    logger.info("✅ Embedding generation complete.")
    session.close()
