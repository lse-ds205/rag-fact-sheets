import sys
from pathlib import Path
from fastapi import FastAPI

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "api"))

from routes import register_routes
from middleware.cors import setup_cors

app = FastAPI(
    title="ds205-group4-api",
    description="API Endpoint for DS205 Group 4",
    version="1.0.0"
)

setup_cors(app)
register_routes(app)