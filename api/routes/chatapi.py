from pathlib import Path
import sys
from fastapi import APIRouter, HTTPException

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
import entrypoints

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/chat")
async def some_function():
    pass