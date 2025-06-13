"""
Since new countries may not upload their documents, this is a robustness check for our pipeline.
We run assuming as if the country has just uploaded their document, in this case 'Bahrain'.
"""

import sys
from pathlib import Path
from sqlalchemy import text

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from databases.auth import PostgresConnection

db = PostgresConnection()

with db.connect() as conn:
    with conn.begin():
        result = conn.execute(
            text("DELETE FROM documents WHERE country = :country"),
            {"country": "Bahrain"}
        )
        print(f"[MANUAL] Deleted {result.rowcount} document(s) for country: Bahrain")