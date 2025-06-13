import sys
from pathlib import Path
from typing import List, Dict, Any
import re
import logging
import torch
from transformers import AutoTokenizer, AutoModel

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from exceptions import ModelLoadError, InvalidInputError
from constants.settings import COUNTRY_LANG_MAP

logger = logging.getLogger(__name__)


class TransformerEmbedding:
    """
    Transformer-based embedding class using BERT/RoBERTa models.
    """
    
    def __init__(self):
        self.english_model = None
        self.english_tokenizer = None
        self.multilingual_model = None
        self.multilingual_tokenizer = None
        self.models_loaded = False

    def load_models(self):
        """Load embedding models if they haven't been loaded yet."""
        if not self.models_loaded:
            try:
                self.english_tokenizer, self.english_model = self._load_english_model()
                self.multilingual_tokenizer, self.multilingual_model = self._load_multilingual_model()
                self.models_loaded = True
                logger.info("Successfully loaded transformer embedding models")
            except Exception as e:
                logger.error(f"Failed to load transformer embedding models: {e}")
                raise ModelLoadError(f"Failed to load transformer embedding models: {e}")
  
    def embed_many(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embeds multiple chunks using appropriate language models.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            raise InvalidInputError("No chunks provided for embedding")

        if not self.models_loaded:
            self.load_models()
        
        embedded_chunks = []
        for chunk in chunks:
            filename = chunk.get('metadata', {}).get('filename', '')
            metadata = chunk.get('metadata', {})
            language = self._determine_language(filename, metadata)
            
            # Add language to metadata if not already present
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['language'] = language
            
            # Get embedding for this chunk
            embedding = self.embed_transformer(chunk)
            chunk['embedding'] = embedding
            embedded_chunks.append(chunk)
            
        return embedded_chunks

    def embed_transformer(self, input_data) -> List[float]:
        """
        Generate an embedding for a single chunk or text.
        
        Args:
            input_data: Either a string of text or a chunk dictionary
            
        Returns:
            List of embedding values (vector)
        """

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
            embedding = self._embed_strategy_english(text, self.english_tokenizer, self.english_model)
        elif self.multilingual_model is not None:
            embedding = self._embed_strategy_multilingual(text, self.multilingual_tokenizer, self.multilingual_model)
        else:
            # If models failed to load, return empty embedding
            logger.warning("No transformer embedding models available, returning empty embedding")
            embedding = []
            
        return embedding

    @staticmethod
    def _embed_strategy_english(text, tokenizer, model):
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
            logger.error(f"Error in English transformer embedding: {e}")
            return []

    @staticmethod
    def _embed_strategy_multilingual(text, tokenizer, model):
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
            logger.error(f"Error in multilingual transformer embedding: {e}")
            return []
            
    @staticmethod
    def _determine_language(filename, metadata=None):
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
                
        # Default to English if we can't determine language
        return 'en'

    @staticmethod
    def _load_english_model():
        """
        Load the English language model (DistilRoBERTa) from local directory or download.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        logger.info("Loading English transformer model...")
        try:
            # Check if local model exists in proper directory structure
            local_model_path = project_root / "local_models" / "distilroberta-base"
            
            if local_model_path.exists():
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
            
            logger.info("Successfully loaded English model from Hugging Face")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Could not load or download English model: {e}")
            raise ModelLoadError(f"Could not load or download English model: {e}")

    @staticmethod
    def _load_multilingual_model():
        """
        Load the multilingual model (XLM-RoBERTa) from local directory or download.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        logger.info("Loading multilingual transformer model...")
        try:
            # Check if local model exists in proper directory structure
            local_model_path = project_root / "local_models" / "xlm-roberta-base"
            
            if local_model_path.exists():
                logger.info(f"Found local multilingual model at {local_model_path}")
                try:
                    # Attempt to load directly from the local path
                    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
                    model = AutoModel.from_pretrained("xlm-roberta-base")
                    logger.info("Successfully loaded multilingual model from local path")
                    return tokenizer, model
                except Exception as local_e:
                    logger.warning(f"Failed to load from local path: {local_e}")
            
            # If local loading fails or model doesn't exist, download from Hugging Face
            logger.info("Downloading multilingual model from Hugging Face")
            model_name = "xlm-roberta-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            logger.info("Successfully loaded multilingual model from Hugging Face")
            return tokenizer, model
        except Exception as e:
            logger.error(f"Could not load or download multilingual model: {e}")
            raise ModelLoadError(f"Could not load or download multilingual model: {e}")
