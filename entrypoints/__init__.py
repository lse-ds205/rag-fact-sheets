"""
Entrypoints module - exposes all run_script functions for API consumption.

This module provides clean imports for all entrypoint functionality,
making it easy for API routes to access individual pipeline stages.
"""

import importlib

# Import modules (required for numeric prefixes)
scrape_module = importlib.import_module('entrypoints.2_scrape')
chunk_module = importlib.import_module('entrypoints.2_chunk')
embed_module = importlib.import_module('entrypoints.3_embed')
retrieve_module = importlib.import_module('entrypoints.4_retrieve')
llm_module = importlib.import_module('entrypoints.5_llm_response')
download_module = importlib.import_module('entrypoints.1_download')
output_module = importlib.import_module('entrypoints.6_output')

# Clean wrapper functions
def run_scrape():
    """Run the scraping/detection process."""
    return scrape_module.run_script()

async def run_chunk(force_reprocess: bool = False):
    """Run the document chunking pipeline."""
    return await chunk_module.run_script(force_reprocess=force_reprocess)

async def run_embed(force_reembed: bool = False):
    """Run the embedding generation pipeline."""
    return await embed_module.run_script(force_reembed=force_reembed)

def run_retrieve(prompt: str):
    """Run the retrieval process to find relevant chunks."""
    return retrieve_module.run_script(prompt=prompt)

def run_llm_response(top_selected_chunks, prompt: str):
    """Generate LLM response based on chunks and prompt."""
    return llm_module.run_script(top_selected_chunks=top_selected_chunks, prompt=prompt)

def run_download(db_url: str, download_dir: str = "data", limit=None):
    """Run the download process to fetch PDFs."""
    return download_module.process_downloads(db_url=db_url, download_dir=download_dir, limit=limit)

def run_output():
    """Run the output processing."""
    return output_module.main()



# Expose all functions
__all__ = [
    'run_scrape',
    'run_chunk',
    'run_embed', 
    'run_retrieve',
    'run_llm_response',
    'run_download',
    'run_output'
]