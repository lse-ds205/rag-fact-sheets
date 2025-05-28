import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sqlalchemy import create_engine

from retrieval import boolean_search, bm25_search, fuzzy_search, vector_search, df_with_similarity_score, hybrid_scoring
from functions import generate_word2vec_embedding_for_text, generate_embeddings_for_text

load_dotenv()

class InformationRetriever:
    def __init__(self):
        """Initialize models and database connection"""
        self.engine = create_engine(os.getenv("DB_URL"))
        self.df = None
        self.climatebert_model = None
        self.climatebert_tokenizer = None
        self.custom_w2v = None
        
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """Load ClimateBERT and Word2Vec models"""
        # Load ClimateBERT
        embedding_model_dir = os.getenv('EMBEDDING_MODEL_LOCAL_DIR')
        self.climatebert_tokenizer = AutoTokenizer.from_pretrained(embedding_model_dir)
        self.climatebert_model = AutoModel.from_pretrained(embedding_model_dir)
        
        # Load Word2Vec
        self.custom_w2v = Word2Vec.load("./local_model/custom_word2vec_768.model")
    
    def _load_data(self):
        """Load document embeddings from database"""
        self.df = pd.read_sql("SELECT * FROM document_embeddings", self.engine)
    
    def expand_keywords(self, prompt, topn=5):
        """Generate similar words using Word2Vec model"""
        keywords = simple_preprocess(prompt)
        similar_words = []
        
        for keyword in keywords:
            try:
                if keyword in self.custom_w2v.wv:
                    similar = self.custom_w2v.wv.most_similar(keyword, topn=topn)
                    similar_words.extend([word for word, score in similar])
            except KeyError:
                continue
        
        all_search_terms = list(set(keywords + similar_words))
        return keywords, all_search_terms
    
    def generate_embeddings(self, prompt):
        """Generate embeddings for prompt using both models"""
        prompt_w2v_embeddings = generate_word2vec_embedding_for_text(prompt, self.custom_w2v)
        prompt_climatebert_embeddings = generate_embeddings_for_text(
            prompt, self.climatebert_model, self.climatebert_tokenizer
        )
        return prompt_w2v_embeddings, prompt_climatebert_embeddings
    
    def keyword_retrieval(self, prompt, k=25, method='bm25'):
        """Perform keyword-based retrieval"""
        _, expanded_keywords = self.expand_keywords(prompt)
        
        if method == 'boolean':
            return boolean_search(expanded_keywords, self.df, k=k)
        elif method == 'bm25':
            return bm25_search(expanded_keywords, self.df, k=k)
        elif method == 'fuzzy':
            return fuzzy_search(prompt, self.df, k=k)
        else:
            raise ValueError("Method must be 'boolean', 'bm25', or 'fuzzy'")
    
    def semantic_retrieval(self, prompt, k=25, embedding_type='climatebert'):
        """Perform semantic retrieval using embeddings"""
        prompt_w2v_embeddings, prompt_climatebert_embeddings = self.generate_embeddings(prompt)
        
        if embedding_type == 'climatebert':
            return vector_search(
                prompt_embeddings=np.array(prompt_climatebert_embeddings),
                embedding_type='climatebert',
                top_k=k
            )
        elif embedding_type == 'word2vec':
            return vector_search(
                prompt_embeddings=np.array(prompt_w2v_embeddings),
                embedding_type='word2vec',
                top_k=k
            )
        else:
            raise ValueError("embedding_type must be 'climatebert' or 'word2vec'")
    
    def hybrid_retrieval(self, prompt, k=25, alpha=0.5):
        """Perform hybrid retrieval combining keyword and semantic search"""
        # Get expanded keywords
        _, expanded_keywords = self.expand_keywords(prompt)
        
        # Generate embeddings
        prompt_w2v_embeddings, prompt_climatebert_embeddings = self.generate_embeddings(prompt)
        
        # Get similarity scores for all documents
        df_similarity = df_with_similarity_score(
            prompt_embeddings_w2v=np.array(prompt_w2v_embeddings),
            prompt_embeddings_climatebert=np.array(prompt_climatebert_embeddings),
            top_k=None
        )
        
        # Add BM25 scores
        bm25_df = bm25_search(expanded_keywords, df_similarity, k=None)
        
        # Calculate hybrid scores
        hybrid_results = hybrid_scoring(bm25_df, alpha=alpha)
        
        return hybrid_results.head(k)
    
    def retrieve(self, prompt, method='hybrid', k=25, **kwargs):
        """Main retrieval function that can use different methods"""
        if method == 'keyword':
            return self.keyword_retrieval(prompt, k=k, **kwargs)
        elif method == 'semantic':
            return self.semantic_retrieval(prompt, k=k, **kwargs)
        elif method == 'hybrid':
            return self.hybrid_retrieval(prompt, k=k, **kwargs)
        else:
            raise ValueError("Method must be 'keyword', 'semantic', or 'hybrid'")