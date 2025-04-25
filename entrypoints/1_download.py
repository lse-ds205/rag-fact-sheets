import sys
from pathlib import Path
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.database import Connection, Document
from group4py.src.helpers import Logger, Test, TaskInfo

logger = logging.getLogger(__name__)