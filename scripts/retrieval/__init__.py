# Import main retrieval class
from .retrieval_pipeline import InformationRetriever

# Import convenience functions
from .retrieval_utils import (
    get_retriever,
    do_retrieval,
    compare_retrieval_methods,
    cp1a_retrieval
)

from .retrieval_support import (
    boolean_search, 
    bm25_search, 
    fuzzy_search, 
    vector_search, 
    df_with_similarity_score, 
    hybrid_scoring
)