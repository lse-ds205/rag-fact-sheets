#!/usr/bin/env python3
"""
Import structured JSON text files (from Scrapy) into the PostgreSQL database.
"""
import os
import json
import logging
import psycopg2
from pathlib import Path
from psycopg2.extras import Json
from dotenv import load_dotenv

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = PROJECT_ROOT / 'data' / 'full_text' / 'structured'

# DB Connection
def get_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME", "climate"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return None

# Load and combine text
def load_combined_text(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or "sections" not in data:
            logger.warning(f"Unexpected JSON format or no 'sections' in {json_path}")
            return None

        all_text = ""
        for section, section_data in data["sections"].items():
            content = section_data.get("content", [])
            if isinstance(content, list):
                section_text = "\n".join(content)
                all_text += f"\n## {section}\n{section_text}\n"

        return all_text.strip() if all_text.strip() else None

    except Exception as e:
        logger.warning(f"Failed to parse {json_path}: {e}")
        return None

def ensure_country_exists(doc_id, country_name, conn):
    # Generate fallback URL using doc_id
    country_url = f"https://climateactiontracker.org/countries/{doc_id}/"

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO countries_v2 (doc_id, country_name, country_url)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id) DO NOTHING
            """, (doc_id, country_name, country_url))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error inserting into countries_v2 for {doc_id}: {e}")
        conn.rollback()
        return False
# Update DB
def store_sections(doc_id, country_name, data, conn):
    try:
        with conn.cursor() as cur:
            for section_title, section_data in data.get("sections", {}).items():
                section_url = section_data.get("url")
                content_list = section_data.get("content", [])
                text_content = "\n".join(content_list).strip() if isinstance(content_list, list) else None

                if not section_url or not text_content or len(text_content) < 20:
                    logger.warning(f"Skipping section '{section_title}' for {doc_id} due to missing or short content.")
                    continue

                cur.execute("""
                    INSERT INTO country_page_sections_v2 (
                        country_doc_id, section_title, section_url, text_content, language, scraped_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (section_url) DO NOTHING
                """, (doc_id, section_title, section_url, text_content, 'en'))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error storing sections for {doc_id}: {e}")
        conn.rollback()
        return False

def main():
    logger.info(f"Looking for structured JSON files in: {STRUCTURED_DIR}")
    conn = get_connection()
    if not conn:
        return

    updated, failed = 0, 0

    for json_path in STRUCTURED_DIR.glob("*.json"):
        doc_id = json_path.stem  # remove .json extension
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            country_name = data.get("country_name") or doc_id.capitalize()
        except Exception as e:
            logger.error(f"❌ Failed to read JSON {json_path}: {e}")
            failed += 1
            continue

        if not ensure_country_exists(doc_id, country_name, conn):
            logger.error(f"✖ Failed to ensure country record for {doc_id}")
            failed += 1
            continue

        success = store_sections(doc_id, country_name, data, conn)
        if success:
            logger.info(f"✔ Stored sections for {doc_id}")
            updated += 1
        else:
            logger.error(f"✖ Failed to store sections for {doc_id}")
            failed += 1

    conn.close()
    logger.info(f"\n✅ Done. Updated {updated} documents. Failed {failed}.")

if __name__ == "__main__":
    main()