from pathlib import Path
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Any
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from exceptions import ModelLoadError, ModelNotLoadedError, EmbeddingGenerationError, InvalidInputError

logger = logging.getLogger(__name__)


class Word2VecEmbedding:
    """
    Word2Vec embedding class with global model training and usage.
    """
    
    def __init__(self):
        self.global_model = None
        self.model_loaded = False

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
            raise InvalidInputError("No texts provided for Word2Vec training")
            
        logger.info(f"Training global Word2Vec model on {len(texts)} texts...")
        
        # Preprocess texts into sentences
        processed_sentences = self._preprocess_texts_for_training(texts)
        
        if not processed_sentences:
            logger.error("No valid sentences after preprocessing")
            raise InvalidInputError("No valid sentences after preprocessing texts")
        
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
            raise ModelLoadError(f"Error training Word2Vec model: {e}")

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
            raise ModelLoadError(f"Word2Vec model not found at {model_path}")
        
        try:
            self.global_model = Word2Vec.load(model_path)
            self.model_loaded = True
            logger.info(f"Loaded Word2Vec model from {model_path}. Vocabulary size: {len(self.global_model.wv)}")
            return True
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            raise ModelLoadError(f"Error loading Word2Vec model from {model_path}: {e}")

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
            raise ModelNotLoadedError("Global Word2Vec model not loaded")
        
        if not text or not isinstance(text, str):
            raise InvalidInputError("Invalid text input for embedding")
            
        return self._create_document_embedding(text, self.global_model)

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
            raise ModelNotLoadedError("Global Word2Vec model not loaded")
            
        if not chunks:
            raise InvalidInputError("No chunks provided for embedding")
            
        logger.info(f"Generating Word2Vec embeddings for {len(chunks)} chunks using global model")
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            text = chunk.get('text', '')
            try:
                embedding = self._create_document_embedding(text, self.global_model)
                
                # Add to chunk
                chunk['w2v_embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            except Exception as e:
                logger.error(f"Error generating Word2Vec embedding for chunk: {e}")
                raise EmbeddingGenerationError(f"Error generating Word2Vec embedding: {e}")
            
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