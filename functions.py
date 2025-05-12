"""
PDF Processor for NB01. Includes text extraction, chunking, embeddings generation, and database storage.
"""
import logging
import os
import glob
import uuid
import torch
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm, trange
from transformers import AutoTokenizer, AutoModel


# Set up logger
logger = logging.getLogger(__name__)


def generate_embeddings_for_text(texts, model, tokenizer):
    """
    Get embeddings from a pre-trained transformer model.
    
    Args:
        texts: List of input texts
        model: Transformer model
        tokenizer: Tokenizer for the model
        
    Returns:
        NumPy array of embeddings for the input texts
    """

    # Tokenize the input texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Get model output (without gradient calculation for efficiency)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use the CLS token embedding as the sentence embedding
    # This is a simple approach - in practice, you might use more sophisticated pooling
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    
    return sentence_embeddings.numpy().flatten().tolist()



