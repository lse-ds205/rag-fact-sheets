"""
Utility functions for the climate policy extractor.
"""
import zoneinfo
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
import os

def now_london_time():
    """Get current time in London timezone."""
    london_tz = zoneinfo.ZoneInfo('Europe/London')
    return datetime.now(london_tz)

# Word embedding utilities
def preprocess_text(text, min_word_length=3):
    """Preprocess text for word embeddings."""
    if not text:
        return []
    
    # Download required NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize text into words and filter
    words = [
        word.lower() for word in word_tokenize(text)
        if word.isalpha() and len(word) >= min_word_length and word.lower() not in stop_words
    ]
    
    return words

def generate_word_embeddings(text, vector_size=100, window=5, min_count=5, workers=4):
    """Generate word embeddings for the given text."""
    if not text:
        return None
    
    # Preprocess text
    words = preprocess_text(text)
    
    if len(words) < min_count:
        return None
    
    # Convert to format required by Word2Vec (list of lists)
    sentences = [words]
    
    # Train Word2Vec model
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers
        )
        
        # Calculate document embedding (average of all word vectors)
        words_in_model = [word for word in words if word in model.wv]
        if not words_in_model:
            return None
            
        doc_vector = np.mean([model.wv[word] for word in words_in_model], axis=0)
        
        return {
            'model': model,
            'document_vector': doc_vector.tolist() if isinstance(doc_vector, np.ndarray) else None
        }
    except Exception as e:
        print(f"Error generating word embeddings: {e}")
        return None

def save_word2vec_model(model, doc_id, model_dir):
    """Save Word2Vec model to disk."""
    if not model or not doc_id:
        return None
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{doc_id}_word2vec.model")
    
    try:
        model.save(model_path)
        return model_path
    except Exception as e:
        print(f"Error saving Word2Vec model: {e}")
        return None