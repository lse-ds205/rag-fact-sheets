import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import group4py
from embed.transformer import TransformerEmbedding
from embed.word2vec import Word2VecEmbedding

logger = logging.getLogger(__name__)


class CombinedEmbedding:
    """
    Combined embedding class that uses both transformer and Word2Vec embeddings.
    """
    
    def __init__(self):
        self.transformer_embedder = TransformerEmbedding()
        self.word2vec_embedder = Word2VecEmbedding()

    def load_models(self, word2vec_model_path: Optional[str] = None):
        """
        Load both transformer and Word2Vec models.
        
        Args:
            word2vec_model_path: Path to Word2Vec model (optional)
        """

        self.transformer_embedder.load_models()
        
        if word2vec_model_path:
            self.word2vec_embedder.load_global_model(word2vec_model_path)

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