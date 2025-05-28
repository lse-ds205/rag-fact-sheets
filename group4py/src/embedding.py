import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import torch
import numpy as np
from gensim.models import Word2Vec
import logging
from nltk.tokenize import word_tokenize

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
import group4py
from helpers.internal import Logger, Test, TaskInfo

# Create a standard logger
logger = logging.getLogger(__name__)

# Add imports for model loading
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    logger.warning("Transformers library not installed. Using dummy embedding functionality.")

# Model directories
ENGLISH_MODEL_DIR = project_root / "models" / "english"
MULTILINGUAL_MODEL_DIR = project_root / "models" / "multilingual"

# Country to language mapping for determining appropriate embedding model
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

    @Logger.debug_log()
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
            language = self._determine_language(filename, metadata)
            
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
        Generate an embedding for a single chunk or text.
        
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
            raise

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
                    logger.info("Successfully loaded multilingual model from HuggingFace")
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
            raise


class Word2VecEmbedding:
    """
    Word2Vec embedding class with global model training and usage.
    """
    
    def __init__(self):
        self.global_model = None
        self.model_loaded = False

    @Logger.debug_log()
    def train_global_model(self, texts: List[str], model_path: Optional[str] = None, 
                          vector_size: int = 300, window: int = 10, min_count: int = 2, 
                          workers: int = 4, sg: int = 1, epochs: int = 10) -> Optional[Word2Vec]:
        """
        Train a global Word2Vec model on all provided texts.
        
        Args:
            texts: List of text strings to train on
            model_path: Path to save the trained model
            vector_size: Dimensionality of the word vectors
            window: Maximum distance between the current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of CPU cores to use
            sg: Training algorithm: 1 for skip-gram; 0 for CBOW
            epochs: Number of training epochs
            
        Returns:
            Trained Word2Vec model or None if training failed
        """
        if not texts:
            logger.error("No texts provided for Word2Vec training")
            return None
            
        logger.info(f"Training global Word2Vec model on {len(texts)} texts...")
        
        # Preprocess texts into sentences
        processed_sentences = self._preprocess_texts_for_training(texts)
        
        if not processed_sentences:
            logger.error("No valid sentences after preprocessing")
            return None
        
        try:
            # Train the model
            model = Word2Vec(
                sentences=processed_sentences,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers,
                sg=sg,
                epochs=epochs,
                alpha=0.025,
                min_alpha=0.0001
            )
            
            logger.info(f"Word2Vec training completed. Vocabulary size: {len(model.wv)}")
            
            # Save the model if path provided
            if model_path:
                # Ensure directory exists
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                model.save(model_path)
                logger.info(f"Saved Word2Vec model to {model_path}")
            
            self.global_model = model
            self.model_loaded = True
            return model
            
        except Exception as e:
            logger.error(f"Error training Word2Vec model: {e}")
            return None

    @Logger.debug_log()
    def load_global_model(self, model_path: str) -> bool:
        """
        Load a pre-trained global Word2Vec model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if loading successful, False otherwise
        """
        if not Path(model_path).exists():
            logger.warning(f"Word2Vec model not found at {model_path}")
            return False
        
        try:
            self.global_model = Word2Vec.load(model_path)
            self.model_loaded = True
            logger.info(f"Loaded Word2Vec model from {model_path}. Vocabulary size: {len(self.global_model.wv)}")
            return True
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            return False

    @Logger.debug_log()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create document embedding for a single text using the global model.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding values
        """
        if not self.model_loaded or not self.global_model:
            logger.error("Global Word2Vec model not loaded")
            return np.zeros(300)  # Default vector size
        
        return self._create_document_embedding(text, self.global_model)

    @Logger.debug_log()
    def embed_many(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate Word2Vec embeddings for a list of document chunks using the global model.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with Word2Vec embeddings added
        """
        if not self.model_loaded or not self.global_model:
            logger.error("Global Word2Vec model not loaded")
            return chunks
            
        logger.info(f"Generating Word2Vec embeddings for {len(chunks)} chunks using global model")
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            text = chunk.get('text', '')
            embedding = self._create_document_embedding(text, self.global_model)
            
            # Add to chunk
            chunk['w2v_embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
        logger.info(f"Successfully generated Word2Vec embeddings for {len(chunks)} chunks")
        return chunks

    def _preprocess_texts_for_training(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for Word2Vec training.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tokenized sentences (list of lists of tokens)
        """
        logger.info("Preprocessing texts for Word2Vec training...")
        
        processed_sentences = []
        
        for text in texts:
            if not text or not text.strip():
                continue
                
            try:
                # Clean text
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
                text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
                text = text.strip()
                
                if not text:
                    continue
                
                # Tokenize
                tokens = word_tokenize(text)
                
                # Filter tokens (remove very short words, numbers only, etc.)
                tokens = [token.strip() for token in tokens if token.strip()]
                tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
                
                if len(tokens) > 3:  # Only keep sentences with enough content
                    processed_sentences.append(tokens)
                    
            except Exception as e:
                logger.warning(f"Error preprocessing text: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed_sentences)} sentences for Word2Vec training")
        return processed_sentences

    def _create_document_embedding(self, text: str, model: Word2Vec) -> np.ndarray:
        """
        Create document embedding using the Word2Vec model.
        
        Args:
            text: Text to embed
            model: Word2Vec model to use
            
        Returns:
            Numpy array of embedding values
        """
        if not text or not model:
            return np.zeros(model.vector_size if model else 300)
        
        # Preprocess text (same as training preprocessing)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return np.zeros(model.vector_size)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
            tokens = [token.strip() for token in tokens if token.strip()]
            tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
        except:
            return np.zeros(model.vector_size)
        
        if not tokens:
            return np.zeros(model.vector_size)
        
        # Get embeddings for valid tokens
        token_embeddings = []
        for token in tokens:
            if token in model.wv:
                token_embeddings.append(model.wv[token])
        
        if not token_embeddings:
            return np.zeros(model.vector_size)
        
        # Average the embeddings
        return np.mean(token_embeddings, axis=0)


class CombinedEmbedding:
    """
    Combined embedding class that uses both transformer and Word2Vec embeddings.
    """
    
    def __init__(self):
        self.transformer_embedder = TransformerEmbedding()
        self.word2vec_embedder = Word2VecEmbedding()

    @Logger.debug_log()
    def load_models(self, word2vec_model_path: Optional[str] = None):
        """
        Load both transformer and Word2Vec models.
        
        Args:
            word2vec_model_path: Path to Word2Vec model (optional)
        """
        # Load transformer models
        self.transformer_embedder.load_models()
        
        # Load Word2Vec model if path provided
        if word2vec_model_path:
            self.word2vec_embedder.load_global_model(word2vec_model_path)

    @Logger.debug_log()
    def train_word2vec_on_texts(self, texts: List[str], model_path: Optional[str] = None) -> bool:
        """
        Train Word2Vec model on provided texts.
        
        Args:
            texts: List of texts to train on
            model_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        model = self.word2vec_embedder.train_global_model(texts, model_path)
        return model is not None

    @Logger.debug_log()
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate both transformer and Word2Vec embeddings for chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with both types of embeddings
        """
        logger.info(f"Generating combined embeddings for {len(chunks)} chunks")
        
        # Generate transformer embeddings
        if self.transformer_embedder.models_loaded:
            chunks = self.transformer_embedder.embed_many(chunks)
            logger.info("Generated transformer embeddings")
        else:
            logger.warning("Transformer models not loaded, skipping transformer embeddings")
        
        # Generate Word2Vec embeddings
        if self.word2vec_embedder.model_loaded:
            chunks = self.word2vec_embedder.embed_many(chunks)
            logger.info("Generated Word2Vec embeddings")
        else:
            logger.warning("Word2Vec model not loaded, skipping Word2Vec embeddings")
        
        return chunks

    @property
    def models_ready(self) -> bool:
        """Check if at least one embedding model is ready."""
        return self.transformer_embedder.models_loaded or self.word2vec_embedder.model_loaded