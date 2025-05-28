import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from embeddings_utils import generate_word2vec_embedding_for_text, generate_embeddings_for_text

class EmbeddingGenerator:
    def __init__(self):
        self.engine = create_engine(os.getenv("DB_URL"))
        self.climatebert_tokenizer = AutoTokenizer.from_pretrained(os.getenv('EMBEDDING_MODEL_LOCAL_DIR'))
        self.climatebert_model = AutoModel.from_pretrained(os.getenv('EMBEDDING_MODEL_LOCAL_DIR'))
        self.custom_w2v = Word2Vec.load("./local_model/custom_word2vec_768.model")
    
    def generate_query_embeddings(self, prompt):
        """Generate embeddings for a given prompt/query"""
        # Generate Word2Vec embeddings
        prompt_w2v_embeddings = generate_word2vec_embedding_for_text(prompt, self.custom_w2v)
        
        # Generate ClimateBERT embeddings
        prompt_climatebert_embeddings = generate_embeddings_for_text(
            prompt, self.climatebert_model, self.climatebert_tokenizer
        )
        
        return {
            'w2v_embeddings': np.array(prompt_w2v_embeddings),
            'climatebert_embeddings': np.array(prompt_climatebert_embeddings)
        }
    
    def run_workflow(self, prompt):
        """Main workflow for embedding generation"""
        print(f"Generating embeddings for prompt: {prompt}")
        embeddings = self.generate_query_embeddings(prompt)
        print("Embedding generation completed.")
        return embeddings