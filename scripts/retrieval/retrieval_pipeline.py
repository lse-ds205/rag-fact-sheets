import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gensim.utils import simple_preprocess
from retrieval_support import bm25_search, vector_search, df_with_similarity_score, hybrid_scoring

class CountryFilteredRetrieval:
    def __init__(self):
        self.engine = create_engine(os.getenv("DB_URL"))
    
    def load_data_by_country(self, country_codes=None):
        """Load document embeddings filtered by country"""
        if country_codes is None:
            # Load all data if no country filter specified
            query = "SELECT * FROM document_embeddings"
        else:
            # Filter by specific countries
            if isinstance(country_codes, str):
                country_codes = [country_codes]
            
            placeholders = ','.join(['%s'] * len(country_codes))
            query = f"SELECT * FROM document_embeddings WHERE country_code IN ({placeholders})"
        
        df = pd.read_sql(query, self.engine, params=country_codes if country_codes else None)
        return df
    
    def expand_keywords(self, prompt, custom_w2v):
        """Generate similar words using word2vec model"""
        keywords = simple_preprocess(prompt)
        similar_words = []
        
        for keyword in keywords:
            try:
                if keyword in custom_w2v.wv:
                    similar = custom_w2v.wv.most_similar(keyword, topn=5)
                    similar_words.extend([word for word, score in similar])
            except KeyError:
                continue
        
        all_search_terms = list(set(keywords + similar_words))
        return all_search_terms, keywords
    
    def retrieve_chunks(self, prompt_embeddings, all_search_terms, country_codes=None, top_k=25):
        """Retrieve chunks with country filtering"""
        # Load data filtered by country
        df = self.load_data_by_country(country_codes)
        
        if df.empty:
            print(f"No data found for countries: {country_codes}")
            return pd.DataFrame()
        
        print(f"Loaded {len(df)} documents for countries: {country_codes}")
        
        # Get similarity scores for both embedding types
        df_similarity_score = self.get_similarity_scores_filtered(
            df, prompt_embeddings, country_codes
        )
        
        # Perform BM25 search on filtered data
        bm25_df = bm25_search(all_search_terms, df_similarity_score, k=None)
        
        # Apply hybrid scoring
        hybrid_results = hybrid_scoring(bm25_df, alpha=0.5)
        
        return hybrid_results.head(top_k)
    
    def get_similarity_scores_filtered(self, df, prompt_embeddings, country_codes):
        """Get similarity scores with country filtering using raw DataFrame"""
        # Since we already have filtered DataFrame, work with it directly
        w2v_embeddings = prompt_embeddings['w2v_embeddings']
        climatebert_embeddings = prompt_embeddings['climatebert_embeddings']
        
        # Calculate similarity scores manually for the filtered DataFrame
        w2v_scores = []
        climatebert_scores = []
        
        for _, row in df.iterrows():
            # Calculate cosine similarity for each embedding type
            if 'word2vec_embedding' in df.columns and row['word2vec_embedding'] is not None:
                w2v_sim = np.dot(w2v_embeddings, row['word2vec_embedding']) / (
                    np.linalg.norm(w2v_embeddings) * np.linalg.norm(row['word2vec_embedding'])
                )
                w2v_scores.append(w2v_sim)
            else:
                w2v_scores.append(0)
            
            if 'climatebert_embedding' in df.columns and row['climatebert_embedding'] is not None:
                cb_sim = np.dot(climatebert_embeddings, row['climatebert_embedding']) / (
                    np.linalg.norm(climatebert_embeddings) * np.linalg.norm(row['climatebert_embedding'])
                )
                climatebert_scores.append(cb_sim)
            else:
                climatebert_scores.append(0)
        
        # Add scores to DataFrame
        df = df.copy()
        df['w2v_score'] = w2v_scores
        df['climatebert_score'] = climatebert_scores
        
        return df
    
    def run_workflow(self, prompt_embeddings, all_search_terms, country_codes=None, top_k=25):
        """Main workflow for country-filtered chunk retrieval"""
        print(f"Starting retrieval for countries: {country_codes}")
        results = self.retrieve_chunks(
            prompt_embeddings, all_search_terms, country_codes, top_k
        )
        print(f"Retrieved {len(results)} relevant chunks.")
        return results