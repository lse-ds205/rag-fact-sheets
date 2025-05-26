from pathlib import Path
import sys
from fastapi import APIRouter, HTTPException

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from entrypoints import run_scrape

router = APIRouter(prefix="/scrape", tags=["scrape"])

@router.post("/run")
async def run_scrape_endpoint():
    """Execute the NDC document scraping workflow."""
    try:
        result = run_scrape()
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))