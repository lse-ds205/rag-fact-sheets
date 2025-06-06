#!/usr/bin/env python3
"""
Management script for Climate Action Tracker project.
Supports scraping and embedding generation.
"""
import os
import sys
import subprocess
from pathlib import Path
import click
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sys
from pathlib import Path
import psycopg2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from climate_tracker.climate_tracker.models import Base, get_db_session
from climate_tracker.climate_tracker.utils import now_london_time
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
# Define outside the try block
conn = None
cursor = None
try:
    conn = psycopg2.connect(
        DATABASE_URL
    )
    cursor = conn.cursor()

    create_stmt = """
    CREATE TABLE IF NOT EXISTS countries (
        doc_id TEXT PRIMARY KEY,
        country TEXT,
        language TEXT,
        text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    cursor.execute(create_stmt)
    conn.commit()
    click.echo("✅ Table `countries` created or already exists.")
except Exception as e:
    click.echo(f"❌ Failed to create table: {e}")
finally:
    if conn:
        cursor.close()
        conn.close()