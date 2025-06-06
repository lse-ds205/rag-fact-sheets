import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

from .functions import generate_embeddings_for_text, generate_word2vec_embedding_for_text
from .retrieval_support import bm25_search, hybrid_scoring, df_with_similarity_score

load_dotenv()

def generate_prompt_embeddings(prompt, climatebert_model, climatebert_tokenizer, word2vec_model):
    """
    Generate embeddings for the input prompt using both models.
    
    Args:
        prompt (str): Input query text
        climatebert_model: Pre-loaded ClimateBERT model
        climatebert_tokenizer: Pre-loaded ClimateBERT tokenizer
        word2vec_model: Pre-loaded Word2Vec model
    
    Returns:
        tuple: (climatebert_embeddings, word2vec_embeddings)
    """
    # Generate ClimateBERT embeddings
    climatebert_embeddings = generate_embeddings_for_text(
        prompt, climatebert_model, climatebert_tokenizer
    )
    
    # Generate Word2Vec embeddings
    word2vec_embeddings = generate_word2vec_embedding_for_text(
        prompt, word2vec_model
    )
    
    return climatebert_embeddings, word2vec_embeddings

def get_expanded_keywords(prompt, word2vec_model, top_similar=5):
    """
    Generate expanded keywords from prompt using Word2Vec similarity.
    Matches the keyword expansion logic from NB03.
    
    Args:
        prompt (str): Input query text
        word2vec_model: Pre-loaded Word2Vec model
        top_similar (int): Number of similar words to include per keyword
    
    Returns:
        list: Combined original and similar keywords (all_search_terms)
    """
    keywords = simple_preprocess(prompt)
    similar_words = []
    
    # For each keyword, try to find similar words
    for keyword in keywords:
        try:
            # Only get similar words if keyword exists in vocabulary
            if keyword in word2vec_model.wv:
                similar = word2vec_model.wv.most_similar(keyword, topn=top_similar)
                similar_words.extend([word for word, score in similar])
        except KeyError:
            # Skip words not in vocabulary
            continue
    
    # Combine original keywords with similar words
    all_search_terms = list(set(keywords + similar_words))
    
    return all_search_terms

def retrieve_relevant_chunks(prompt, country_code=None, top_k=25, alpha=0.5):
    """
    Main function to retrieve relevant document chunks using hybrid search.
    Follows the exact workflow from NB03.
    
    Args:
        prompt (str): Input query text
        country_code (str, optional): 3-letter country code to filter by
        top_k (int): Number of top results to return
        alpha (float): Weight for BM25 vs semantic similarity (0-1)
    
    Returns:
        pd.DataFrame: Top relevant chunks with text, page number, and document name only
    """
    # Load models    
    EMBEDDING_MODEL_LOCAL_DIR = os.getenv('EMBEDDING_MODEL_LOCAL_DIR')
    climatebert_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_LOCAL_DIR)
    climatebert_model = AutoModel.from_pretrained(EMBEDDING_MODEL_LOCAL_DIR)
    
    # Use project root path to ensure consistent file loading regardless of working directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(project_root, "local_model", "custom_word2vec_768.model")
    word2vec_model = Word2Vec.load(model_path)
    
    # Step 1: Generate embeddings for prompt (matches NB03)
    prompt_climatebert_embeddings = generate_embeddings_for_text(
        prompt, climatebert_model, climatebert_tokenizer
    )
    prompt_w2v_embeddings = generate_word2vec_embedding_for_text(
        prompt, word2vec_model
    )
    
    # Step 2: Get expanded keywords (matches NB03 all_search_terms logic)
    all_search_terms = get_expanded_keywords(prompt, word2vec_model)
    
    # Step 3: Use df_with_similarity_score exactly like NB03
    df_similarity_score = df_with_similarity_score(
        prompt_embeddings_w2v=np.array(prompt_w2v_embeddings),
        prompt_embeddings_climatebert=np.array(prompt_climatebert_embeddings),
        top_k=None  # Get all results initially
    )
    
    # Step 4: Filter by country if specified
    if country_code:
        df_similarity_score = df_similarity_score[
            df_similarity_score['country_code'] == country_code
        ]
        
        if df_similarity_score.empty:
            print(f"No documents found for country code: {country_code}")
            return pd.DataFrame(columns=['text','document_title'])
    
    # Step 5: Apply BM25 search exactly like NB03
    bm25_df = bm25_search(all_search_terms, df_similarity_score, k=None)
    
    # Step 6: Calculate hybrid scores exactly like NB03
    hybrid_results = hybrid_scoring(bm25_df, alpha=alpha)
    
    # Step 7: Return only text, page number, and document name
    result_columns = ['original_text', 'document_title']
    
    # Get top k results and rename columns for clarity
    top_results = hybrid_results[result_columns].head(top_k)
    top_results = top_results.rename(columns={
        'original_text': 'text',
        'document_title': 'document_title'
    })
    
    return top_results

def do_retrieval(prompt, country=None, k=25):
    """
    Simplified interface for document search.
    
    Args:
        prompt (str): Search query
        country (str, optional): 3-letter country code
        k (int): Number of results to return
    
    Returns:
        list: List of chunk dictionaries in expected format
    """
    # Get DataFrame results
    df_results = retrieve_relevant_chunks(prompt, country_code=country, top_k=k)
    
    # Convert to expected list format
    chunks = []
    for _, row in df_results.iterrows():
        chunk = {
            'chunk_content': row['text'],
            'document': row['document_title'],
        }
        chunks.append(chunk)
    
    return chunks
