from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test, TaskInfo

import os
import json
import nltk
from typing import List, Dict, Any, Optional
import logging  # Add explicit logging import
from tqdm import tqdm  # Import tqdm for progress bars

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize
from unstructured.partition.pdf import partition_pdf

# Get a logger for this module
logger = logging.getLogger(__name__)

@Logger.debug_log()
def extract_text_from_pdf(
    pdf_path: str, 
    strategy: str = "fast", 
    extract_images: bool = False,
    infer_table_structure: bool = True,
    languages: str = "eng",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file using the unstructured library.
    
    This function uses the unstructured library to extract text and metadata
    from a PDF file. The extracted elements include text, page numbers,
    paragraph information, and other metadata that can be used for document analysis.
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Strategy for extraction:
            - "fast": Fast extraction without OCR (default)
            - "ocr_only": Use only OCR for text extraction
            - "auto": Automatically determine best extraction method
        extract_images: Whether to extract images from the PDF
        infer_table_structure: Whether to infer table structure in the PDF
        languages: Languages to use for OCR (if applicable)
        **kwargs: Additional parameters to pass to partition_pdf
        
    Returns:
        List of extracted elements with their metadata
    """
    # Validate and map strategy parameter
    # Valid strategies for partition_pdf are "fast", "ocr_only", and "auto"
    valid_strategies = ["fast", "ocr_only", "auto"]
                
    if strategy not in valid_strategies:
        logger.warning(f"Invalid strategy: {strategy}. Defaulting to 'fast'")
        strategy = "fast"

    try:
        filename = os.path.basename(pdf_path)
        logger.info(f"Extracting text from PDF: {pdf_path} using strategy: {strategy}")
        
       
        languages_list = [lang.strip() for lang in languages.split(',')] if languages else ['eng']
        logger.debug(f"Using languages: {languages_list}")
        
        # Extract elements from the PDF - removed chunking_strategy parameter
        logger.debug(f"Starting PDF extraction with extract_images={extract_images}, infer_table_structure={infer_table_structure}")
        elements = partition_pdf(
            filename=pdf_path, 
            strategy=strategy,
            extract_images_in_pdf=extract_images,
            infer_table_structure=infer_table_structure,
            languages=languages_list,
            **kwargs
        )
        
        logger.info(f"Extracted {len(elements)} elements from PDF")
        
        # Initialize paragraph tracking
        paragraphs_by_page = {}  # Dict to track paragraph numbers by page
        last_element_type = None
        last_page_number = None
        global_paragraph_count = 0  # Track paragraphs across the entire document
        
        # Convert unstructured elements to a more usable dictionary format
        result = []
        
        # Use tqdm to track element processing
        for i, element in enumerate(tqdm(elements, desc=f"Processing {filename}", unit="element", leave=False)):
            # Skip empty elements
            if not hasattr(element, 'text') or not element.text.strip():
                continue
                
            # Create a dictionary with element information
            element_dict = {
                "id": f"element_{i}",
                "type": element.category if hasattr(element, 'category') else type(element).__name__,
                "text": element.text if hasattr(element, 'text') else str(element),
                "metadata": {
                    "filename": filename
                }
            }
            
            # Get page number if available
            page_number = None
            if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
                page_number = element.metadata.page_number
            elif hasattr(element, "page_number"):
                page_number = element.page_number
            else:
                page_number = 0
                
            element_dict["metadata"]["page_number"] = page_number
            
            # Initialize or get paragraph counter for this page
            if page_number not in paragraphs_by_page:
                paragraphs_by_page[page_number] = 0
                
            # Determine if this is a new paragraph based on element type and context
            is_new_paragraph = False
            if page_number != last_page_number:  # New page = new paragraph
                is_new_paragraph = True
            elif element.category != last_element_type if hasattr(element, 'category') else True:  # Type change = new paragraph
                is_new_paragraph = True
            elif (hasattr(element, 'category') and 
                  element.category == "NarrativeText" and last_element_type in ["Title", "ListItem"]):
                is_new_paragraph = True
                
            if is_new_paragraph:
                paragraphs_by_page[page_number] += 1
                global_paragraph_count += 1  # Increment global paragraph counter
                
            # Update tracking variables
            last_element_type = element.category if hasattr(element, 'category') else None
            last_page_number = page_number
            
            # Add paragraph information to metadata
            paragraph_id = f"p{page_number}_para{paragraphs_by_page[page_number]}"
            element_dict["metadata"]["paragraph_id"] = paragraph_id
            element_dict["metadata"]["paragraph_number"] = paragraphs_by_page[page_number]  # Add explicit paragraph number
            element_dict["metadata"]["global_paragraph_number"] = global_paragraph_count  # Add global paragraph number
            
            # Add coordinates if available
            if hasattr(element, "coordinates"):
                element_dict["metadata"]["coordinates"] = element.coordinates
                
            # Add any other metadata that might be useful
            if hasattr(element, "metadata"):
                for key, value in vars(element.metadata).items():
                    if key not in ["page_number"]:  # Skip already added metadata
                        element_dict["metadata"][key] = value
                        
            result.append(element_dict)
        
        logger.info(f"Successfully processed {len(result)} non-empty elements")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}", exc_info=True)
        return []
    
@Logger.debug_log()    
def extract_text_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX file.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        A list of dictionaries with text and metadata
    """
    logger.info(f"Extracting text from DOCX: {docx_path}")
    try:
        # Only import docx when needed
        from docx import Document
        doc = Document(docx_path)
        logger.debug(f"Successfully opened DOCX document")
    except ImportError:
        logger.error("python-docx package is not installed. Cannot extract text from DOCX file.")
        return [{'text': f"ERROR: Could not extract text from {docx_path}. python-docx package is not installed.", 'type': 'error'}]
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {docx_path}: {str(e)}", exc_info=True)
        return [{'text': f"ERROR: Could not extract text from {docx_path}. {str(e)}", 'type': 'error'}]
    
    # Get the filename for metadata
    filename = os.path.basename(docx_path)
    elements = []
    page_number = 1  # DOCX doesn't have a concept of pages, so we simulate it
    paragraph_count = 0
    
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if text:
            paragraph_count += 1
            # Every 20 paragraphs, we simulate a new page
            if paragraph_count > 20:
                page_number += 1
                paragraph_count = 1
                
            # Determine element type based on formatting
            element_type = "paragraph"
            if paragraph.style.name.startswith("Heading"):
                element_type = "Heading"
            elif paragraph.style.name == "Title":
                element_type = "Title"
                
            elements.append({
                'id': f"element_{i}",
                'type': element_type,
                'text': text,
                'metadata': {
                    'page_number': page_number,
                    'filename': filename,
                    'paragraph_id': f"p{page_number}_para{paragraph_count}",
                    'style': paragraph.style.name
                }
            })
    
    logger.info(f"Successfully extracted {len(elements)} elements from DOCX")
    return elements
