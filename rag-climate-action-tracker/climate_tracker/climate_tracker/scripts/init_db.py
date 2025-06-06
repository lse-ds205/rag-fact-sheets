import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from climate_tracker.climate_tracker.models import Base, get_db_session
from climate_tracker.climate_tracker.utils import now_london_time
from climate_tracker.climate_tracker.utils import generate_word_embeddings, save_word2vec_model
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

def create_engine_and_extension():
    """Create the database engine and vector extension."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    return engine


"""Initialize the database with vector support."""
engine = create_engine_and_extension()
    
try:
    # Create vector extension first
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        
        # Create all tables
        Base.metadata.create_all(engine)
        
except Exception as e:
    raise