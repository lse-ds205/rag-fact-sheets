# Import main retrieval class
from .retrieval_pipeline import do_retrieval, retrieve_relevant_chunks

# Import convenience functions
from .functions import (
    store_database_batched,
    embed_and_store_all_embeddings,
    generate_word2vec_embedding_for_text,
    load_climatebert_model,
    train_custom_word2vec_from_texts,
    generate_embeddings_for_text
    
)

from .retrieval_support import (
    boolean_search, 
    bm25_search, 
    fuzzy_search, 
    vector_search, 
    df_with_similarity_score, 
    hybrid_scoring
)
