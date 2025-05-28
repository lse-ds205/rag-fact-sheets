from pathlib import Path
import sys
import os
import json
import nltk
import traceback
import logging
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from helpers.internal import Logger, Test, TaskInfo

# Get a logger for this module
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

def _extract_country_from_filename(pdf_path: str) -> str:
    """
    Extract country name from PDF filename.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Country name or 'Unknown'
    """
    filename = os.path.basename(pdf_path)
    filename_lower = filename.lower()
    
    # Common country name patterns in filenames
    country_patterns = {
        'afghanistan': 'Afghanistan',
        'albania': 'Albania',
        'algeria': 'Algeria',
        'argentina': 'Argentina',
        'australia': 'Australia',
        'austria': 'Austria',
        'bangladesh': 'Bangladesh',
        'belgium': 'Belgium',
        'brazil': 'Brazil',
        'canada': 'Canada',
        'chile': 'Chile',
        'china': 'China',
        'colombia': 'Colombia',
        'costa_rica': 'Costa Rica',
        'egypt': 'Egypt',
        'ethiopia': 'Ethiopia',
        'france': 'France',
        'germany': 'Germany',
        'ghana': 'Ghana',
        'india': 'India',
        'indonesia': 'Indonesia',
        'iran': 'Iran',
        'iraq': 'Iraq',
        'italy': 'Italy',
        'japan': 'Japan',
        'kenya': 'Kenya',
        'madagascar': 'Madagascar',
        'malaysia': 'Malaysia',
        'mexico': 'Mexico',
        'morocco': 'Morocco',
        'nigeria': 'Nigeria',
        'pakistan': 'Pakistan',
        'peru': 'Peru',
        'philippines': 'Philippines',
        'poland': 'Poland',
        'russia': 'Russia',
        'saudi_arabia': 'Saudi Arabia',
        'singapore': 'Singapore',
        'south_africa': 'South Africa',
        'south_korea': 'South Korea',
        'spain': 'Spain',
        'thailand': 'Thailand',
        'turkey': 'Turkey',
        'ukraine': 'Ukraine',
        'united_kingdom': 'United Kingdom',
        'uk': 'United Kingdom',
        'united_states': 'United States',
        'usa': 'United States',
        'vietnam': 'Vietnam',
    }
    
    # Try to find country in filename
    for pattern, country_name in country_patterns.items():
        if pattern in filename_lower:
            return country_name
    
    # If no pattern found, try to extract from first part of filename
    # Many files are named like "country_language_date.pdf"
    filename_parts = filename.replace('.pdf', '').split('_')
    if filename_parts:
        first_part = filename_parts[0].lower()
        # Try exact match first
        for pattern, country_name in country_patterns.items():
            if first_part == pattern:
                return country_name
        
        # If still no match, capitalize the first part as a fallback
        if len(first_part) > 2:
            return first_part.replace('_', ' ').title()

    return 'Unknown'

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
    valid_strategies = ["fast", "ocr_only", "auto"]
                
    if strategy not in valid_strategies:
        logger.warning(f"Invalid strategy '{strategy}', using 'fast' instead")
        strategy = "fast"

    try:
        filename = os.path.basename(pdf_path)
        country = _extract_country_from_filename(pdf_path)
        logger.info(f"Extracting text from PDF: {pdf_path} using strategy: {strategy}")
        logger.info(f"Detected country: {country}")
        
        languages_list = [lang.strip() for lang in languages.split(',')] if languages else ['eng']
        logger.debug(f"Using languages: {languages_list}")
        
        # Extract elements from the PDF
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
          # Process and filter elements
        processed_elements = []
        has_corruption = False
        
        # Track page-relative paragraph numbering
        current_page = 0
        page_paragraph_count = 0
        
        for i, element in enumerate(elements):
            # Get text content
            text = str(element).strip() if element else ""
            
            # Check for PDF corruption/encoding issues
            if _has_character_corruption(text):
                has_corruption = True
                logger.debug(f"Detected character corruption in element {i}: {text[:100]}...")
                continue  # Skip corrupted elements
              # Skip very short or empty text
            if len(text) < 10:
                continue
            
            # Skip common PDF artifacts
            skip_patterns = [
                r'^\d+$',  # Just page numbers
                r'^[ivxlcdm]+$',  # Roman numerals only
                r'^[\s\-_=\.]+$',  # Just punctuation/whitespace
                r'^(page|pg)\s*\d+$',  # Page indicators
            ]
            
            if any(re.match(pattern, text.lower()) for pattern in skip_patterns):
                continue

            # Extract metadata - handle both dict-like and object-like structures
            metadata = {
                'element_index': i,
                'element_type': type(element).__name__,
                'page_number': 0,  # Default value, will be updated below if available
                'paragraph_number': None,
                'extraction_strategy': strategy,
                'filename': filename,
                'country': country,
                'document_title': f"{country} NDC"
            }
            
            # Handle metadata extraction from unstructured elements
            try:
                if hasattr(element, 'metadata') and element.metadata:
                    element_metadata = element.metadata
                    # Handle different metadata types
                    if hasattr(element_metadata, 'page_number'):
                        metadata['page_number'] = element_metadata.page_number
                    elif hasattr(element_metadata, 'to_dict'):
                        # Convert object to dict and extract page_number
                        meta_dict = element_metadata.to_dict()
                        metadata['page_number'] = meta_dict.get('page_number', 0)
                      # Extract coordinates if available
                    if hasattr(element_metadata, 'coordinates'):
                        metadata['coordinates'] = str(element_metadata.coordinates)
                    elif hasattr(element_metadata, 'to_dict'):
                        meta_dict = element_metadata.to_dict()
                        if 'coordinates' in meta_dict:
                            metadata['coordinates'] = str(meta_dict['coordinates'])
                      # Extract filename if available
                    if hasattr(element_metadata, 'filename'):
                        metadata['source_file'] = element_metadata.filename
                    elif hasattr(element_metadata, 'to_dict'):
                        meta_dict = element_metadata.to_dict()
                        if 'filename' in meta_dict:
                            metadata['source_file'] = meta_dict['filename']
                      # Try to extract paragraph number from coordinates or other metadata
                    if hasattr(element_metadata, 'to_dict'):
                        meta_dict = element_metadata.to_dict()
                        # Look for any field that might indicate paragraph/element order
                        for key in ['paragraph', 'para', 'element_id', 'order']:
                            if key in meta_dict and isinstance(meta_dict[key], (int, str)):
                                try:
                                    metadata['paragraph_number'] = int(meta_dict[key])
                                    break
                                except (ValueError, TypeError):                                    continue
                                    
            except Exception as meta_error:
                logger.warning(f"Error extracting metadata from element {i}: {meta_error}")
                # Keep default metadata values
            
            # Check if we moved to a new page and reset paragraph counter
            if metadata['page_number'] != current_page:
                current_page = metadata['page_number']
                page_paragraph_count = 1
            else:
                page_paragraph_count += 1
            
            # If no paragraph number found, use page-relative numbering
            if metadata['paragraph_number'] is None:
                metadata['paragraph_number'] = page_paragraph_count
            
            processed_elements.append({
                'text': text,
                'metadata': metadata
            })

        # If we detected corruption and got poor results, try OCR
        if has_corruption and len(processed_elements) < 5:
            logger.warning(f"Detected significant character corruption, attempting OCR extraction")
            return _retry_with_ocr(pdf_path, languages_list)
        
        logger.info(f"Successfully processed {len(processed_elements)} non-empty elements")
        
        # If we got no valid elements, try to extract any text we can find
        if not processed_elements:
            logger.warning(f"No valid elements found with strategy {strategy}, attempting OCR fallback")
            return _retry_with_ocr(pdf_path, languages_list)

        return processed_elements
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Try OCR as last resort
        try:
            logger.info("Attempting OCR as last resort for failed extraction")
            return _retry_with_ocr(pdf_path, ['eng'])
        except:
            return []

def _has_character_corruption(text: str) -> bool:
    """
    Detect if text contains character encoding corruption typical of PDF extraction issues.
    Improved to reduce false positives with legitimate formatted text.
    
    Args:
        text: Text to check for corruption
        
    Returns:
        True if corruption is detected
    """
    if not text:
        return False
    
    # Check for CID (Character ID) corruption - this is the main indicator
    cid_pattern = r'\(cid:\d+\)'
    cid_matches = len(re.findall(cid_pattern, text))
    
    # If more than 10% of the text appears to be CID references, it's corrupted
    if cid_matches > 0 and (cid_matches * 10) > len(text.split()):
        return True
    
    return False

def _retry_with_ocr(pdf_path: str, languages_list: list) -> List[Dict[str, Any]]:
    """
    Retry PDF extraction using OCR when standard extraction fails or produces corruption.
    
    Args:
        pdf_path: Path to PDF file
        languages_list: List of languages for OCR
        
    Returns:
        List of extracted elements using OCR
    """
    try:
        logger.info(f"Attempting OCR extraction for {pdf_path}")
        filename = os.path.basename(pdf_path)
        country = _extract_country_from_filename(pdf_path)
        
        # Force OCR extraction
        elements = partition_pdf(
            filename=pdf_path, 
            strategy="ocr_only",
            extract_images_in_pdf=False,
            infer_table_structure=True,
            languages=languages_list        )
        
        processed_elements = []
        
        # Track page-relative paragraph numbering for OCR
        current_page = 0
        page_paragraph_count = 0
        
        for i, element in enumerate(elements):
            text = str(element).strip() if element else ""
            
            # Still check for corruption in OCR results
            if _has_character_corruption(text):
                logger.debug(f"OCR element {i} still has corruption, skipping")
                continue
            
            if len(text) < 10:
                continue
                
            # Extract metadata
            metadata = {
                'element_index': i,
                'element_type': type(element).__name__,
                'page_number': 0,
                'paragraph_number': None,
                'extraction_strategy': 'ocr_fallback',
                'filename': filename,
                'country': country,
                'document_title': f"{country} NDC"
            }
            
            # Handle metadata extraction from unstructured elements
            try:
                if hasattr(element, 'metadata') and element.metadata:
                    element_metadata = element.metadata
                    
                    if hasattr(element_metadata, 'page_number'):
                        metadata['page_number'] = element_metadata.page_number
                    elif hasattr(element_metadata, 'to_dict'):
                        meta_dict = element_metadata.to_dict()
                        metadata['page_number'] = meta_dict.get('page_number', 0)
                        
                        # Try to extract paragraph number
                        for key in ['paragraph', 'para', 'element_id', 'order']:
                            if key in meta_dict and isinstance(meta_dict[key], (int, str)):
                                try:
                                    metadata['paragraph_number'] = int(meta_dict[key])
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
            except Exception as meta_error:
                logger.warning(f"Error extracting OCR metadata from element {i}: {meta_error}")
            
            # Check if we moved to a new page and reset paragraph counter
            if metadata['page_number'] != current_page:
                current_page = metadata['page_number']
                page_paragraph_count = 1
            else:
                page_paragraph_count += 1
            
            # If no paragraph number found, use page-relative numbering
            if metadata['paragraph_number'] is None:
                metadata['paragraph_number'] = page_paragraph_count
            
            processed_elements.append({
                'text': text,
                'metadata': metadata
            })

        logger.info(f"OCR extraction completed with {len(processed_elements)} elements")
        return processed_elements
        
    except Exception as e:
        logger.error(f"OCR extraction failed for {pdf_path}: {e}")
        return []

@Logger.debug_log()    
def extract_text_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX file using python-docx.
    
    Args:
        docx_path: Path to the DOCX file
        
    Returns:
        List of extracted text elements with metadata
    """
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
    country = _extract_country_from_filename(docx_path)
    elements = []
    page_number = 0  # DOCX doesn't have a concept of pages, so we simulate it
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
                    'paragraph_number': paragraph_count,
                    'filename': filename,
                    'country': country,
                    'document_title': f"{country} Document",
                    'paragraph_id': f"p{page_number}_para{paragraph_count}",
                    'style': paragraph.style.name
                }
            })
    
    logger.info(f"Extracted {len(elements)} elements from DOCX file")
    return elements
