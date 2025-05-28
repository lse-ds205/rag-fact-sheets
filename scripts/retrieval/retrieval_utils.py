import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from retrieval_support import bm25_search, hybrid_scoring
from functions import generate_word2vec_embedding_for_text, generate_embeddings_for_text


def do_retrieval(prompt, country_codes=None, top_k=25):
    """
    Simple function that applies the complete retrieval workflow
    
    Args:
        prompt (str): Query/prompt to search for
        country_codes (list or str): Country codes to filter by (e.g., ['USA', 'CAN'] or 'USA')
        top_k (int): Number of results to return
    
    Returns:
        pd.DataFrame: Top relevant chunks with scores
    """
    # Load models and setup
    engine = create_engine(os.getenv("DB_URL"))
    climatebert_tokenizer = AutoTokenizer.from_pretrained(os.getenv('EMBEDDING_MODEL_LOCAL_DIR'))
    climatebert_model = AutoModel.from_pretrained(os.getenv('EMBEDDING_MODEL_LOCAL_DIR'))
    custom_w2v = Word2Vec.load("./local_model/custom_word2vec_768.model")
    
    # Generate embeddings for prompt
    prompt_w2v_embeddings = np.array(generate_word2vec_embedding_for_text(prompt, custom_w2v))
    prompt_climatebert_embeddings = np.array(generate_embeddings_for_text(prompt, climatebert_model, climatebert_tokenizer))
    
    # Expand keywords
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
    
    # Load data with country filter
    if country_codes is None:
        query = "SELECT * FROM document_embeddings"
        df = pd.read_sql(query, engine)
    else:
        if isinstance(country_codes, str):
            country_codes = [country_codes]
        placeholders = ','.join(['%s'] * len(country_codes))
        query = f"SELECT * FROM document_embeddings WHERE country_code IN ({placeholders})"
        df = pd.read_sql(query, engine, params=country_codes)
    
    # Apply BM25 search
    bm25_df = bm25_search(all_search_terms, df, k=None)
    
    # Apply hybrid scoring and return top results
    hybrid_results = hybrid_scoring(bm25_df, alpha=0.5)
    
    return hybrid_results.head(top_k)