import sys
from pathlib import Path
from typing import List, Dict, Any
import json
import os
import re
from nltk.tokenize import sent_tokenize
import logging
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from exceptions import ChunkingError, ChunkProcessingError, ChunkCleaningError
from group4py.src.chunk.cleaner import merge_short_chunks, split_long_chunks, remove_gibberish

logger = logging.getLogger(__name__)


class DocChunker:
    """
    Document chunking class. General methodology:
        Input: Some form of documents (PDFs, etc.)
        Output: A list of chunks
    """
    
    @staticmethod
    def chunk_document_by_sentences(elements: list, max_chunk_size: int = 512, overlap: int = 2) -> list:
        """
        Chunk a document into sentences with context for better semantic meaning.
        
        Args:
            elements: List of document elements from unstructured
            max_chunk_size: Maximum size of each chunk in characters (approx)
            overlap: Number of sentences to overlap between chunks
            
        Returns:
            List of chunks with their metadata (page number, etc.)
            
        Raises:
            ChunkProcessingError: If there is an error processing the chunks
        """
        try:
            # Initialize variables
            chunks = []
            current_chunk_text = ""
            current_chunk_sentences = []
            current_chunk_metadata = {}
            chunk_id = 0
            current_element_types = set()
            current_paragraph_numbers = set()
            
            # Process each element
            for element in elements:
                element_type = element.get('type', 'Text')
                element_text = element.get('text', '')
                element_metadata = element.get('metadata', {})
                
                # Skip empty text
                if not element_text.strip():
                    continue
                    
                # Special handling for titles, headings, etc. - keep them as standalone chunks
                if element_type in ['Title', 'Heading', 'Header', 'SubHeading']:
                    # First finish any in-progress chunk
                    if current_chunk_sentences:
                        chunks.append({
                            "id": f"chunk_{chunk_id}",
                            "text": current_chunk_text.strip(),
                            "sentences": current_chunk_sentences,
                            "metadata": current_chunk_metadata
                        })
                        chunk_id += 1
                        
                    # Add the title/heading as its own chunk
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": element_text.strip(),
                        "sentences": [element_text.strip()],
                        "metadata": {
                            "element_types": [element_type],
                            "page_number": element_metadata.get('page_number', 0),
                            "paragraph_number": element_metadata.get('paragraph_number'),
                            "global_paragraph_number": element_metadata.get('global_paragraph_number'),
                            "paragraph_id": element_metadata.get('paragraph_id'),
                            "filename": element_metadata.get('filename', ''),
                            "country": element_metadata.get('country', ''),
                            "document_title": element_metadata.get('document_title', ''),
                            "submission_date": element_metadata.get('submission_date', '')
                        }
                    })
                    chunk_id += 1
                    
                    # Reset tracking variables
                    current_chunk_text = ""
                    current_chunk_sentences = []
                    current_chunk_metadata = {}
                    current_element_types = set()
                    current_paragraph_numbers = set()
                    continue
                
                # For regular text elements, split into sentences
                sentences = sent_tokenize(element_text)
                
                # Start a new chunk if we don't have one yet
                if not current_chunk_sentences:
                    current_chunk_metadata = {
                        "element_types": [element_type],
                        "page_number": element_metadata.get('page_number', 0),
                        "paragraph_numbers": [element_metadata.get('paragraph_number')] if element_metadata.get('paragraph_number') is not None else [],
                        "paragraph_ids": [element_metadata.get('paragraph_id')] if element_metadata.get('paragraph_id') is not None else [],
                        "global_paragraph_numbers": [element_metadata.get('global_paragraph_number')] if element_metadata.get('global_paragraph_number') is not None else [],
                        "filename": element_metadata.get('filename', ''),
                        "country": element_metadata.get('country', ''),
                        "document_title": element_metadata.get('document_title', ''),
                        "submission_date": element_metadata.get('submission_date', '')
                    }
                    current_element_types = {element_type}
                    if element_metadata.get('paragraph_number') is not None:
                        current_paragraph_numbers = {element_metadata.get('paragraph_number')}
                else:                # Track element types and paragraph numbers
                    current_element_types.add(element_type)
                    if element_metadata.get('paragraph_number') is not None:
                        current_paragraph_numbers.add(element_metadata.get('paragraph_number'))
                        if element_metadata.get('paragraph_number') not in current_chunk_metadata['paragraph_numbers']:
                            current_chunk_metadata['paragraph_numbers'].append(element_metadata.get('paragraph_number'))
                    
                    if element_metadata.get('paragraph_id') is not None and element_metadata.get('paragraph_id') not in current_chunk_metadata['paragraph_ids']:
                        current_chunk_metadata['paragraph_ids'].append(element_metadata.get('paragraph_id'))
                    
                    if element_metadata.get('global_paragraph_number') is not None and element_metadata.get('global_paragraph_number') not in current_chunk_metadata['global_paragraph_numbers']:
                        current_chunk_metadata['global_paragraph_numbers'].append(element_metadata.get('global_paragraph_number'))
                        
                    # Add element type if it's a new one
                    if element_type not in current_chunk_metadata['element_types']:
                        current_chunk_metadata['element_types'].append(element_type)
                    
                    # Update country if not set yet but available in this element
                    if not current_chunk_metadata.get('country') and element_metadata.get('country'):
                        current_chunk_metadata['country'] = element_metadata.get('country')
                        
                    # Update document_title if not set yet but available in this element
                    if not current_chunk_metadata.get('document_title') and element_metadata.get('document_title'):
                        current_chunk_metadata['document_title'] = element_metadata.get('document_title')
                        
                    # Update submission_date if not set yet but available in this element
                    if not current_chunk_metadata.get('submission_date') and element_metadata.get('submission_date'):
                        current_chunk_metadata['submission_date'] = element_metadata.get('submission_date')
                
                # Process each sentence
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += " " + sentence
                    else:
                        current_chunk_text = sentence
                        
                    current_chunk_sentences.append(sentence)
                    
                    # Check if we should start a new chunk
                    if len(current_chunk_text) >= max_chunk_size and len(current_chunk_sentences) > overlap + 1:
                        # Create a chunk with the accumulated text
                        chunks.append({
                            "id": f"chunk_{chunk_id}",
                            "text": current_chunk_text.strip(),
                            "sentences": current_chunk_sentences,
                            "metadata": current_chunk_metadata
                        })
                        chunk_id += 1                    # Start new chunk with overlap
                        current_chunk_sentences = current_chunk_sentences[-overlap:] if overlap > 0 else []
                        current_chunk_text = " ".join(current_chunk_sentences)
                        # Reset metadata for the new chunk but keep document-level info
                        current_chunk_metadata = {
                            "element_types": [element_type],  # Start fresh with current element type
                            "page_number": element_metadata.get('page_number', current_chunk_metadata.get('page_number', 0)),
                            "paragraph_numbers": [element_metadata.get('paragraph_number')] if element_metadata.get('paragraph_number') is not None else [],  # Start fresh
                            "paragraph_ids": [element_metadata.get('paragraph_id')] if element_metadata.get('paragraph_id') is not None else [],  # Start fresh
                            "global_paragraph_numbers": [element_metadata.get('global_paragraph_number')] if element_metadata.get('global_paragraph_number') is not None else [],  # Start fresh
                            "filename": current_chunk_metadata.get('filename', ''),  # Keep document info
                            "country": current_chunk_metadata.get('country', ''),  # Keep document info
                            "document_title": current_chunk_metadata.get('document_title', ''),  # Keep document info
                            "submission_date": current_chunk_metadata.get('submission_date', '')  # Keep document info
                        }
                        # Reset tracking sets
                        current_element_types = {element_type}
                        current_paragraph_numbers = {element_metadata.get('paragraph_number')} if element_metadata.get('paragraph_number') is not None else set()
            
            # Add the last chunk if it has content
            if current_chunk_sentences:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": current_chunk_text.strip(),
                    "sentences": current_chunk_sentences,
                    "metadata": current_chunk_metadata
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error in chunk_document_by_sentences: {e}")
            raise ChunkProcessingError(f"Failed to chunk document by sentences: {str(e)}") from e

    @staticmethod
    def cleaning_function(elements: List[Dict[str, Any]], min_length: int = 20, max_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Clean and process document chunks from extract_document.py output.
        
        This function applies multiple cleaning strategies:
        1. Merges chunks that are too short
        2. Splits chunks that are too long
        3. Applies additional cleaning as needed
        
        Args:
            elements: List of document elements from extract_document.py
            min_length: Minimum length threshold for a chunk (characters)
            max_length: Maximum length threshold for a chunk (characters)
            
        Returns:
            List of cleaned and processed chunks
            
        Raises:
            ChunkCleaningError: If there is an error cleaning the chunks
        """
        try:
            logger.info(f"Cleaning {len(elements)} document chunks")        # Apply cleaning strategies in sequence using proper static method references
            cleaned_elements = merge_short_chunks(elements, min_length)
            cleaned_elements = split_long_chunks(cleaned_elements, max_length)
            cleaned_elements = remove_gibberish(cleaned_elements)
            
            logger.info(f"Cleaning complete. Produced {len(cleaned_elements)} chunks")
            return cleaned_elements
        except Exception as e:
            logger.error(f"Error in cleaning_function: {e}")
            raise ChunkCleaningError(f"Failed to clean document chunks: {str(e)}") from e

    @staticmethod
    def chunk_json(chunks: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        Generate a JSON file from document chunks to be saved in the data/processed folder.
        
        Args:
            chunks: List of document chunks to convert to JSON
            output_path: Optional custom output path. If None, will use default path based on metadata
            
        Returns:
            String path to the generated JSON file
            
        Raises:
            ChunkingError: If there is an error generating the JSON file
        """
        if not chunks:
            logger.warning("No chunks provided to generate JSON")
            return ""
            
        # Create a metadata object with information about the chunks
        output_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                "source": "chunking.py"
            },
            "chunks": chunks
        }
        
        # Determine the output path
        if not output_path:
            # Try to construct output path from the first chunk's metadata
            first_chunk = chunks[0]
            metadata = first_chunk.get('metadata', {})
            country = metadata.get('country', 'unknown')
            date = metadata.get('submission_date', datetime.now().strftime('%Y%m%d'))
            
            # Clean country name for filename (remove special chars, lowercase)
            clean_country = re.sub(r'[^\w]', '_', country.lower())
            
            # Create the output directory if it doesn't exist
            output_dir = project_root / "data" / "processed"
            os.makedirs(output_dir, exist_ok=True)
            
            # Construct filename
            filename = f"{clean_country}_{date}_chunks.json"
            output_path = output_dir / filename
        
        try:
            # Convert to JSON and save to file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(output_data, file, indent=2, cls=ExtractedDataEncoder)
            logger.info(f"Successfully generated JSON with {len(chunks)} chunks at {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating JSON file: {e}")
            raise ChunkingError(f"Failed to generate JSON file: {str(e)}") from e
            

class ExtractedDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special data types."""
    def default(self, obj):
        # Handle CoordinatesMetadata specifically
        if obj.__class__.__name__ == 'CoordinatesMetadata':
            return {
                'points': getattr(obj, 'points', []),
                'system': getattr(obj, 'system', ''),
                'x1': getattr(obj, 'x1', 0),
                'y1': getattr(obj, 'y1', 0),
                'x2': getattr(obj, 'x2', 0),
                'width': getattr(obj, 'width', 0),
                'height': getattr(obj, 'height', 0)
            }
        
        # Handle other types
        if hasattr(obj, '__dict__'): 
            return obj.__dict__
        try: 
            return dict(obj)
        except: 
            pass
        try: 
            return list(obj)
        except: 
            pass
        return str(obj)