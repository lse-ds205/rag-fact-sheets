import sys
import os
import json
import re
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers import Logger, Test, TaskInfo

# Create a standard logger instead of a Logger instance
logger = logging.getLogger(__name__)

# Add imports for model loading
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    logger.warning("Transformers library not installed. Using dummy embedding functionality.")

# Model directories
ENGLISH_MODEL_DIR = project_root / "models" / "english"
MULTILINGUAL_MODEL_DIR = project_root / "models" / "multilingual"

COUNTRY_LANG_MAP = {
    'france': 'fr',
    'french': 'fr',
    'germany': 'de',
    'german': 'de',
    'spain': 'es',
    'spanish': 'es',
    'italy': 'it',
    'italian': 'it',
    'china': 'zh',
    'chinese': 'zh',
    'japan': 'ja',
    'japanese': 'ja',
    'russia': 'ru',
    'russian': 'ru',
    'brazil': 'pt',
    'portuguese': 'pt',
    'india': 'en',  # India often uses English for official documents
    'usa': 'en',
    'uk': 'en',
    'english': 'en',
    'mexico': 'es',
    'argentina': 'es',
    'colombia': 'es',
    'peru': 'es',
    'chile': 'es',
    'venezuela': 'es',
}

class DocChunk:
    """
    Chunk class. General methodology:
        Input: Some form of documents (PDFs, etc.)
        Output: A list of chunks
    """
    def __init__(self):
        pass

    @Logger.debug_log()     
    def chunk_document_by_sentences(self, elements: list, max_chunk_size: int = 512, overlap: int = 2) -> list:
        """
        Chunk a document into sentences with context for better semantic meaning.
        
        Args:
            elements: List of document elements from unstructured
            max_chunk_size: Maximum size of each chunk in characters (approx)
            overlap: Number of sentences to overlap between chunks
            
        Returns:
            List of chunks with their metadata (page number, etc.)
        """
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
            else:
                # Track element types and paragraph numbers
                current_element_types.add(element_type)
                if element_metadata.get('paragraph_number') is not None:
                    current_paragraph_numbers.add(element_metadata.get('paragraph_number'))
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
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    current_chunk_sentences = current_chunk_sentences[-overlap:] if overlap > 0 else []
                    current_chunk_text = " ".join(current_chunk_sentences)
                    # Keep the same metadata since we're still in the same element
        
        # Add the last chunk if it has content
        if current_chunk_sentences:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": current_chunk_text.strip(),
                "sentences": current_chunk_sentences,
                "metadata": current_chunk_metadata
            })
        
        return chunks

    @Logger.debug_log()
    def cleaning_function(self, elements: List[Dict[str, Any]], min_length: int = 20, max_length: int = 1000) -> List[Dict[str, Any]]:
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
        """
        # Use logging.getLogger() directly instead of the logger variable
        logging.getLogger(__name__).info(f"Cleaning {len(elements)} document chunks")
        
        # Apply cleaning strategies in sequence using proper static method references
        cleaned_elements = DocChunk._DocChunk.merge_short_chunks(elements, min_length)
        cleaned_elements = DocChunk._DocChunk.split_long_chunks(cleaned_elements, max_length)
        cleaned_elements = DocChunk._DocChunk.remove_gibberish(cleaned_elements)
        
        logging.getLogger(__name__).info(f"Cleaning complete. Produced {len(cleaned_elements)} chunks")
        return cleaned_elements

    @Logger.debug_log()
    def chunk_json(self, chunks: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        Generate a JSON file from document chunks to be saved in the data/processed folder.
        
        Args:
            chunks: List of document chunks to convert to JSON
            output_path: Optional custom output path. If None, will use default path based on metadata
            
        Returns:
            String path to the generated JSON file
        """
        if not chunks:
            logger.warning("No chunks provided to generate JSON")
            return ""
            
        # Create a metadata object with information about the chunks
        output_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "timestamp": TaskInfo.get_timestamp(),
                "version": TaskInfo.get_version(),
                "source": "chunk_embed.py"
            },
            "chunks": chunks
        }
        
        # Determine the output path
        if not output_path:
            # Try to construct output path from the first chunk's metadata
            first_chunk = chunks[0]
            metadata = first_chunk.get('metadata', {})
            country = metadata.get('country', 'unknown')
            date = metadata.get('submission_date', TaskInfo.get_date_string())
            
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
            return ""

    class _DocChunk:
        """
        Internal chunk class with various cleaning strategies.
        """
        @staticmethod
        def merge_short_chunks(elements: List[Dict[str, Any]], min_length: int = 20) -> List[Dict[str, Any]]:
            """
            Merge chunks that are shorter than the minimum length threshold.
            
            Args:
                elements: List of document elements
                min_length: Minimum length threshold for a chunk
                
            Returns:
                List of merged chunks
            """
            if not elements:
                return elements
            
            merged_elements = []
            current_chunk = None
            
            for element in elements:
                text = element.get('text', '')
                
                if current_chunk is None:
                    current_chunk = element
                elif len(current_chunk.get('text', '')) < min_length:
                    # Merge with current chunk
                    current_chunk['text'] = current_chunk.get('text', '') + ' ' + text
                    
                    # Merge metadata where appropriate
                    if 'metadata' in element and 'metadata' in current_chunk:
                        # Merge element types
                        if 'element_types' in element['metadata'] and 'element_types' in current_chunk['metadata']:
                            for et in element['metadata']['element_types']:
                                if et not in current_chunk['metadata']['element_types']:
                                    current_chunk['metadata']['element_types'].append(et)
                        
                        # Merge paragraph numbers if they exist
                        if 'paragraph_numbers' in element['metadata'] and 'paragraph_numbers' in current_chunk['metadata']:
                            for pn in element['metadata']['paragraph_numbers']:
                                if pn not in current_chunk['metadata']['paragraph_numbers']:
                                    current_chunk['metadata']['paragraph_numbers'].append(pn)
                        
                        # Merge paragraph IDs if they exist
                        if 'paragraph_ids' in element['metadata'] and 'paragraph_ids' in current_chunk['metadata']:
                            for pid in element['metadata']['paragraph_ids']:
                                if pid not in current_chunk['metadata']['paragraph_ids']:
                                    current_chunk['metadata']['paragraph_ids'].append(pid)
                else:
                    # Current chunk is long enough, add to results
                    merged_elements.append(current_chunk)
                    current_chunk = element
            
            # Don't forget the last chunk
            if current_chunk is not None:
                merged_elements.append(current_chunk)
            
            return merged_elements
            
        @staticmethod
        def split_long_chunks(elements: List[Dict[str, Any]], max_length: int = 1000) -> List[Dict[str, Any]]:
            """
            Split chunks that are longer than the maximum length threshold.
            
            Args:
                elements: List of document elements
                max_length: Maximum length threshold for a chunk
                
            Returns:
                List of split chunks
            """
            if not elements:
                return elements
                
            result = []
            chunk_id_counter = 0
            
            for element in elements:
                text = element.get('text', '')
                
                # If the element is short enough, keep it as is
                if len(text) <= max_length:
                    result.append(element)
                    continue
                
                # If the element has sentences, use those for splitting
                sentences = element.get('sentences', [])
                if not sentences and text:
                    # If no sentences are provided but we have text, tokenize it
                    sentences = sent_tokenize(text)
                
                if not sentences:
                    # If still no sentences, just add the element as is
                    result.append(element)
                    continue
                
                # Split into multiple chunks
                current_chunk_text = ""
                current_sentences = []
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed the max length
                    if len(current_chunk_text) + len(sentence) + 1 > max_length and current_chunk_text:
                        # Create a new chunk with current content
                        new_chunk = {
                            "id": f"chunk_{chunk_id_counter}",
                            "text": current_chunk_text.strip(),
                            "sentences": current_sentences,
                            "metadata": element.get('metadata', {}).copy()  # Copy metadata
                        }
                        result.append(new_chunk)
                        chunk_id_counter += 1
                        
                        # Reset for next chunk
                        current_chunk_text = sentence
                        current_sentences = [sentence]
                    else:
                        # Add to current chunk
                        if current_chunk_text:
                            current_chunk_text += " " + sentence
                        else:
                            current_chunk_text = sentence
                        current_sentences.append(sentence)
                
                # Add the last chunk if there's content
                if current_chunk_text:
                    new_chunk = {
                        "id": f"chunk_{chunk_id_counter}",
                        "text": current_chunk_text.strip(),
                        "sentences": current_sentences,
                        "metadata": element.get('metadata', {}).copy()
                    }
                    result.append(new_chunk)
                    chunk_id_counter += 1
            
            return result
            
        @staticmethod
        def remove_gibberish(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """
            Remove gibberish text from chunks.
            
            This function removes common OCR artifacts and formatting issues.
            
            Args:
                elements: List of document elements
                
            Returns:
                List of cleaned chunks
            """
            if not elements:
                return elements
                
            cleaned_elements = []
            
            for element in elements:
                if 'text' not in element:
                    continue
                
                text = element['text']
                
                # Check for severe corruption that should cause chunk rejection
                if _is_severely_corrupted(text):
                    logger.warning(f"Rejecting severely corrupted chunk: {text[:50]}...")
                    continue
                
                # Clean up the text
                text = _clean_corrupted_text(text)
                
                # Replace multiple spaces with a single space
                text = ' '.join(text.split())
                
                # Remove common OCR artifacts
                text = text.replace('•', '').replace('|', '').replace('¦', '')
                
                # Remove excessive punctuation
                for char in '.,;:!?':
                    text = text.replace(f'{char}{char}{char}', f'{char}')
                    text = text.replace(f'{char} {char}', f'{char}')
                
                # Only include elements that have meaningful content
                if len(text.strip()) > 5:  # Arbitrary threshold to filter out tiny fragments
                    element['text'] = text.strip()
                    cleaned_elements.append(element)
            
            return cleaned_elements

def _is_severely_corrupted(text: str) -> bool:
    """
    Check if text is so corrupted it should be completely rejected.
    Updated to be less aggressive with legitimate formatted text.
    
    Args:
        text: Text to check
        
    Returns:
        True if text should be rejected
    """
    if not text or len(text) < 10:
        return True
    
    # Count CID references - this is the primary corruption indicator
    cid_count = len(re.findall(r'\(cid:\d+\)', text))
    word_count = len(text.split())
    
    # If more than 30% of "words" are CID references, reject (was 50%)
    if word_count > 0 and (cid_count / word_count) > 0.3:
        return True
    
    # Check for strings that are mostly non-alphabetic, but be more lenient
    # Include accented characters and common punctuation
    legit_char_pattern = r'[a-zA-Z0-9àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß\s.,;:!?()\-\'"]'
    legit_chars = len(re.findall(legit_char_pattern, text))
    
    if len(text) > 20 and (legit_chars / len(text)) < 0.4:  # Lowered from 0.7 to 0.4
        return True
    
    # Check for excessive Unicode replacement characters
    if text.count('�') + text.count('\ufffd') > 5:
        return True
    
    return False

def _clean_corrupted_text(text: str) -> str:
    """
    Clean text with character corruption issues.
    Updated to preserve legitimate formatting.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove CID references
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    
    # Remove excessive repeated characters, but preserve table of contents dots
    # Only replace if it's not a table of contents pattern
    if not re.match(r'^[^.]*\.{10,}[^.]*$', text):  # Not a table of contents
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)  # Reduce excessive repetition
    
    # Remove Unicode replacement characters
    text = text.replace('�', ' ').replace('\ufffd', ' ')
    
    # Remove problematic control characters but preserve normal spaces and newlines
    text = re.sub(r'[\x00-\x08\x0E-\x1F\x7F]', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
    
class Embedding:

    """
    Embedding class. General methodology:
        Input: DocChunks, in the form of strings
        Output: Embeddings, in the form of a list of floats
    """
    def __init__(self):
        self.english_model = None
        self.english_tokenizer = None
        self.multilingual_model = None
        self.multilingual_tokenizer = None
        self.models_loaded = False

    @Logger.debug_log()
    def load_models(self):
        """Load embedding models if they haven't been loaded yet."""
        if not self.models_loaded:
            try:
                self.english_tokenizer, self.english_model = self._Embedding.load_english_model()
                self.multilingual_tokenizer, self.multilingual_model = self._Embedding.load_multilingual_model()
                self.models_loaded = True
                logger.info("Successfully loaded embedding models")
            except Exception as e:
                logger.error(f"Failed to load embedding models: {e}")
                # Continue with dummy embeddings
    
    @Logger.debug_log()
    def embed_many(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embeds multiple chunks using appropriate language models.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        # Try to load models if not already loaded
        if not self.models_loaded:
            self.load_models()
        
        embedded_chunks = []
        for chunk in chunks:
            # Determine language for this chunk
            filename = chunk.get('metadata', {}).get('filename', '')
            metadata = chunk.get('metadata', {})
            language = self._Embedding.determine_language(filename, metadata)
            
            # Add language to metadata if not already present
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['language'] = language
            
            # Get embedding for this chunk
            embedding = self.embed_transformer(chunk)
            
            # Add embedding to chunk
            chunk['embedding'] = embedding
            embedded_chunks.append(chunk)
            
        return embedded_chunks

    @Logger.debug_log()
    def embed_transformer(self, input_data) -> List[float]:
        """
        Generate an embedding for a single chunk.
        
        Args:
            input_data: Either a string of text or a chunk dictionary
            
        Returns:
            List of embedding values (vector)
        """
        # Handle empty input
        if not input_data:
            return []
        
        # Extract text and language based on input type
        if isinstance(input_data, str):
            text = input_data
            language = 'en'  # Default to English for string input
        else:
            # Assume it's a dictionary chunk
            text = input_data.get('text', '')
            language = input_data.get('metadata', {}).get('language', 'en')
    
        # Ensure models are loaded
        if not self.models_loaded:
            self.load_models()
    
        # Use appropriate model based on language
        if language == 'en' and self.english_model is not None:
            embedding = self._Embedding._embed_strategy_one(text, self.english_tokenizer, self.english_model)
        elif self.multilingual_model is not None:
            embedding = self._Embedding._embed_strategy_two(text, self.multilingual_tokenizer, self.multilingual_model)
        else:
            # If models failed to load, return dummy embedding
            logger.warning("No embedding models available, returning empty embedding")
            embedding = []
            
        return embedding

    class _Embedding:
        """
        Internal embedding class.
        """
        @staticmethod
        def _embed_strategy_one(text, tokenizer, model):
            """
            Use English model to embed text.
            
            Args:
                text: The text to embed
                tokenizer: The tokenizer to use
                model: The model to use
                
            Returns:
                List of embedding values
            """
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use CLS token as the sentence embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error in English embedding: {e}")
                return []

        @staticmethod
        def _embed_strategy_two(text, tokenizer, model):
            """
            Use multilingual model to embed text.
            
            Args:
                text: The text to embed
                tokenizer: The tokenizer to use
                model: The model to use
                
            Returns:
                List of embedding values
            """
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use CLS token as the sentence embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error in multilingual embedding: {e}")
                return []
                
        @staticmethod
        def determine_language(filename, metadata=None):
            """
            Determine the language from the filename or metadata.
            Falls back to language detection only when needed.
            
            Args:
                filename (str): The filename that might contain language information
                metadata (dict): Optional metadata that might contain language information
                
            Returns:
                str: Two-letter language code (defaults to 'en' when uncertain)
            """
            # First check if metadata already contains language information
            if metadata and 'language' in metadata and metadata['language']:
                return metadata['language'].lower()
                
            # Check for language code in filename (like "document_fr.pdf" or "fr_document.pdf")
            lang_pattern = r'[_\-\s\.](fr|es|de|it|zh|ja|ru|pt|en)[_\-\s\.]'
            lang_match = re.search(lang_pattern, filename.lower())
            if lang_match:
                return lang_match.group(1).lower()
                
            # Look for country names in filename or metadata that can indicate language
            if metadata and 'country' in metadata and metadata['country']:
                country = metadata['country'].lower()
                for country_key, lang_code in COUNTRY_LANG_MAP.items():
                    if country_key in country:
                        return lang_code

            # Check if filename contains country names
            for country_key, lang_code in COUNTRY_LANG_MAP.items():
                if country_key in filename.lower():
                    return lang_code
                    
            # If we still don't have a language, try to detect it from content
            # but we'll do this in the process_document_chunks function
            # to avoid circular imports
            
            # Default to English if we can't determine language
            return 'en'

        @staticmethod
        def load_english_model():
            """
            Load the English language model (ClimateBERT) from local directory.
            If not available, attempt to download it from Hugging Face.
            Returns tokenizer and model as a tuple.
            """
            logger.info(f"Loading English model from {ENGLISH_MODEL_DIR}")
            try:
                # Check if local model exists in proper directory structure
                local_model_path = project_root / "local_models" / "distilroberta-base"
                
                if os.path.exists(local_model_path):
                    logger.info(f"Found local English model at {local_model_path}")
                    try:
                        # Attempt to load directly from the local path
                        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
                        model = AutoModel.from_pretrained("distilroberta-base")
                        logger.info("Successfully loaded English model from HuggingFace")
                        return tokenizer, model
                    except Exception as local_e:
                        logger.warning(f"Failed to load from local path: {local_e}")
                
                # If local loading fails or model doesn't exist, download from Hugging Face
                logger.info("Downloading English model from Hugging Face")
                model_name = "distilroberta-base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                logger.info(f"Successfully loaded English model from Hugging Face")
                return tokenizer, model
            except Exception as e:
                logger.error(f"Could not load or download English model: {e}")
                raise

        @staticmethod
        def load_multilingual_model():
            """
            Load the multilingual model (XLM-RoBERTa) from local directory.
            If not available, attempt to download it from Hugging Face.
            Returns tokenizer and model as a tuple.
            """
            logger.info(f"Loading multilingual model from {MULTILINGUAL_MODEL_DIR}")
            try:
                # Check if local model exists in proper directory structure
                local_model_path = project_root / "local_models" / "xlm-roberta-base"
                
                if os.path.exists(local_model_path):
                    logger.info(f"Found local multilingual model at {local_model_path}")
                    try:
                        # Attempt to load directly from the local path
                        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
                        model = AutoModel.from_pretrained("xlm-roberta-base")
                        logger.info("Successfully loaded multilingual model from HuggingFace")
                        return tokenizer, model
                    except Exception as local_e:
                        logger.warning(f"Failed to load from local path: {local_e}")
                
                # If local loading fails or model doesn't exist, download from Hugging Face
                logger.info("Downloading multilingual model from Hugging Face")
                model_name = "xlm-roberta-base"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                logger.info(f"Successfully loaded multilingual model from Hugging Face")
                return tokenizer, model
            except Exception as e:
                logger.error(f"Could not load or download multilingual model: {e}")
                raise

    @Logger.debug_log()
    def word2vec_embedding(self, chunks: List[Dict[str, Any]], vector_size=300, window=10, min_count=2, workers=4, sg=1) -> List[Dict[str, Any]]:
        """
        Generate Word2Vec embeddings for a list of document chunks.
        
        Args:
            chunks: List of chunks to embed
            vector_size: Dimensionality of the word vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of CPU cores to use
            sg: Training algorithm: 1 for skip-gram; 0 for CBOW
            
        Returns:
            List of chunks with Word2Vec embeddings added
        """
        if not chunks:
            logger.warning("No chunks provided for Word2Vec embedding")
            return []
            
        logger.info(f"Generating Word2Vec embeddings for {len(chunks)} chunks")
        
        # Preprocess all texts and prepare for training
        texts = [chunk.get('text', '') for chunk in chunks]
        tokenized_docs = [self._preprocess_text(text) for text in texts]
        
        # Filter out empty documents
        valid_indices = [i for i, tokens in enumerate(tokenized_docs) if tokens]
        if not valid_indices:    
            logger.warning("No valid documents for Word2Vec training after preprocessing")
            return chunks
        
        valid_tokenized_docs = [tokenized_docs[i] for i in valid_indices]
        
        # Train Word2Vec model on tokenized documentswv)}")
        logger.info(f"Training Word2Vec model with vector_size={vector_size}, window={window}, min_count={min_count}")
        w2v_model = self._train_word2vec(valid_tokenized_docs, vector_size, window, min_count, workers, sg)
        logger.info(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
        
        # Generate embeddings for each chunk
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            embedding = self._create_document_embedding(text, w2v_model)
            
            # Add to chunkunks")
            chunk['w2v_embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
        logger.info(f"Successfully generated Word2Vec embeddings for {len(chunks)} chunks")
        return chunks
        
    def _preprocess_text(self, text):
        """Preprocess text for Word2Vec - simplified version using whole text"""
        if not isinstance(text, str):
            return []
        
        # Simple whitespace tokenization with lowercase conversion
        tokens = text.lower().split()
        
        # Basic filtering for non-words and very short words
        tokens = [token.strip('.,;:!?()[]{}"\'-') for token in tokens]
        tokens = [token for token in tokens if len(token) > 1 and any(c.isalpha() for c in token)]
        
        return tokens

    def _create_document_embedding(self, text, model):
        """Create document embedding directly from text"""
        # Preprocess the text to get tokens
        tokens = self._preprocess_text(text)
        
        if not tokens:
            return np.zeros(model.vector_size)
        
        # Get embeddings for valid tokens
        token_embeddings = [model.wv[token] for token in tokens if token in model.wv]
        
        if not token_embeddings:
            return np.zeros(model.vector_size)
        
        # Average the embeddings
        return np.mean(token_embeddings, axis=0)

    def _train_word2vec(self, tokenized_docs, vector_size=100, window=5, min_count=1, workers=4, sg=1):
        """
        Train a Word2Vec model on a list of tokenized documents.
        
        Args:
            tokenized_docs: List of lists, where each inner list contains tokens from a document
            vector_size: Dimensionality of the word vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of CPU cores to use
            sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        
        Returns:
            Trained Word2Vec model
        """
        model = Word2Vec(sentences=tokenized_docs, 
                         vector_size=vector_size, 
                         window=window, 
                         min_count=min_count, 
                         workers=workers,
                         sg=sg)
        return model

class ExtractedDataEncoder(json.JSONEncoder):
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
        if hasattr(obj, '__dict__'): return obj.__dict__
        try: return dict(obj)
        except: pass
        try: return list(obj)
        except: pass
        return str(obj)