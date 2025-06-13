
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import simple_preprocess
from sqlalchemy import create_engine, text
from rank_bm25 import BM25Okapi

        
def boolean_search(keywords, df, k=None):
    """
    Perform a boolean search on the DataFrame based on the prompt.

    Returns:
    1. The chunks that contains exact keywords in prompt
    2. Boolean score of each chunk = macthed keywords/ total keywords
    """
    # 1. Boolean matching: What is the chunks contains exact keywords in prompt?
    boolean_scores = []
    for _, row in df.iterrows():
        text = str(row['original_text']).lower()
        # Count how many keywords appear in the text
        matches = sum(1 for keyword in keywords if keyword.lower() in text)
        score = matches / len(keywords) if keywords else 0
        boolean_scores.append(score)
    
    df['boolean_score'] = boolean_scores
    
    # Get top k chunks with highest boolean_score
    top_k = df.nlargest(k, 'boolean_score')

    return top_k

def bm25_search(keywords_list, df, k=None):
    """
    Perform a BM25 search on the DataFrame based on the prompt.
    Scores are normalized between 0 and 1.

    Parameters:
    keywords_list (list): List of keywords to search for
    df (DataFrame): DataFrame containing the text corpus
    k (int): Number of top results to return

    Returns: DataFrame with normalized BM25 score of each chunk 
    """

    # Prepare corpus
    tokenized_corpus = [doc.lower().split() for doc in df['original_text']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Convert list of keywords to lowercase for consistency
    tokenized_prompt = [word.lower() for word in keywords_list]

    # BM25 scoring
    bm25_scores = bm25.get_scores(tokenized_prompt)
    
    # Normalize scores between 0 and 1
    max_score = max(bm25_scores) if len(bm25_scores) > 0 else 1
    normalized_scores = bm25_scores / max_score if max_score != 0 else bm25_scores
    
    df['bm25_score'] = normalized_scores

    # Get sorted results
    if k is None:
        # Return all results sorted by score
        return df.sort_values('bm25_score', ascending=False)
    else:
        # Return top k results
        return df.nlargest(k, 'bm25_score')

    return top_k  

def fuzzy_search(query, df, k=25):
    """
    Performs fuzzy string matching between query and text chunks with duplicate removal
    Args:
        query (str): Search query
        df (pd.DataFrame): DataFrame containing text chunks
        k (int): Number of results to return
    Returns:
        pd.DataFrame: Top k unique results with normalized fuzzy scores
    """
    # Calculate fuzzy match ratio for each chunk
    fuzzy_scores = df['original_text'].apply(lambda x: fuzz.token_set_ratio(query, str(x)) / 100.0)
    
    # Add scores to dataframe copy
    results = df
    results['fuzzy_score'] = fuzzy_scores
    
    # Remove duplicates by keeping highest score for each unique text
    results = (results.sort_values('fuzzy_score', ascending=False)
              .drop_duplicates(subset=['original_text'])
              .head(k))
    
    return results

def vector_search(prompt_embeddings, embedding_type='climatebert', top_k=25):
    """
    Vector search using cosine similarity with pgvector and pandas post-processing
    """
    load_dotenv()
    engine = create_engine(os.getenv("DB_URL"))

    # Format embeddings for PostgreSQL vector
    embeddings_str = f"'[{','.join(map(str, prompt_embeddings))}]'"
    
    # Basic SQL query to get raw scores
    sql_command = f"""
    SELECT 
        document_id,
        country_code,
        original_text,
        (1 - ({embedding_type}_embedding <=> {embeddings_str}::vector)) as similarity_score
    FROM document_embeddings
    WHERE original_text IS NOT NULL;
    """
    
    # Get raw results
    results_df = pd.read_sql(sql_command, engine)
    
    # Clean and process results using pandas
    results_df = (results_df
        # Remove rows with empty or invalid text
        .query("original_text != '' and original_text.str.len() > 10")
        # Remove numeric-only text
        .loc[~results_df['original_text'].str.match(r'^\d+(\.\d+)?$')]
        # Remove array-like text
        .loc[~results_df['original_text'].str.startswith('[')]
        # Normalize scores using min-max scaling
        .assign(
            similarity_score=lambda df: (
                (df['similarity_score'] - df['similarity_score'].min()) /
                (df['similarity_score'].max() - df['similarity_score'].min())
            )
        )
        # Remove duplicates and sort
        .drop_duplicates(subset=['original_text'])
        .sort_values('similarity_score', ascending=False)
        .head(top_k)
    )
    
    return results_df

def df_with_similarity_score(prompt_embeddings_w2v, prompt_embeddings_climatebert, top_k=25):
    """
    Vector search using cosine similarity with pgvector and pandas post-processing
    Returns extended DataFrame with both W2V and ClimateBERT similarity scores
    """
    load_dotenv()
    engine = create_engine(os.getenv("DB_URL"))

    # Format embeddings for PostgreSQL vector
    w2v_embeddings_str = f"'[{','.join(map(str, prompt_embeddings_w2v))}]'"
    climatebert_embeddings_str = f"'[{','.join(map(str, prompt_embeddings_climatebert))}]'"
    
    # Extended SQL query to get both similarity scores
    sql_command = f"""
    SELECT 
        document_id,
        country_code,
        document_title,
        original_text,
        source_hyperlink,
        (1 - (word2vec_embedding <=> {w2v_embeddings_str}::vector)) as w2v_score,
        (1 - (climatebert_embedding <=> {climatebert_embeddings_str}::vector)) as climatebert_score
    FROM document_embeddings
    WHERE original_text IS NOT NULL;
    """
    
    # Get raw results
    results_df = pd.read_sql(sql_command, engine)
    
    # Clean and process results using pandas
    results_df = (results_df
        # Remove invalid text
        .query("original_text != '' and original_text.str.len() > 10")
        .loc[~results_df['original_text'].str.match(r'^\d+(\.\d+)?$')]
        .loc[~results_df['original_text'].str.startswith('[')]
        
        # Normalize both similarity scores
        .assign(
            w2v_score=lambda df: (
                (df['w2v_score'] - df['w2v_score'].min()) /
                (df['w2v_score'].max() - df['w2v_score'].min())
            ),
            climatebert_score=lambda df: (
                (df['climatebert_score'] - df['climatebert_score'].min()) /
                (df['climatebert_score'].max() - df['climatebert_score'].min())
            )
        )
        # Remove duplicates and sort by average score
        .assign(
            avg_score=lambda df: (df['w2v_score'] + df['climatebert_score']) / 2
        )
        .drop_duplicates(subset=['original_text'])
        .sort_values('avg_score', ascending=False)
        .head(top_k)
    )
    
    return results_df

def hybrid_scoring(df, alpha=0.5):
    """
    Compute hybrid score from vector and keyword search; tune alpha
    """
    # Access columns using proper DataFrame indexing
    bm25_score = df['bm25_score'].values
    climatebert_sim_score = df['climatebert_score'].values

    # Compute hybrid score
    hybrid_score = alpha * bm25_score + (1 - alpha) * climatebert_sim_score

    # Add hybrid score to DataFrame
    df_scored = df.copy()
    df_scored['hybrid_score'] = hybrid_score

    # Rank the DataFrame based on the hybrid score
    df_scored = df_scored.sort_values(by='hybrid_score', ascending=False)

    return df_scored

