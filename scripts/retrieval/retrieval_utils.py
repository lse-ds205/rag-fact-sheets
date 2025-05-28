from retrieval_pipeline import InformationRetriever

# Global retriever instance (initialized once)
_retriever = None

def get_retriever():
    """Get or create the global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = InformationRetriever()
    return _retriever

def do_retrieval(prompt, method='hybrid', k=25, **kwargs):
    """Quick search function for immediate use"""
    retriever = get_retriever()
    return retriever.retrieve(prompt, method=method, k=k, **kwargs)

def compare_retrieval_methods(prompt, k=25):
    """Compare different retrieval methods for the same prompt"""
    retriever = get_retriever()
    
    results = {}
    results['bm25'] = retriever.keyword_retrieval(prompt, k=k, method='bm25')
    results['climatebert'] = retriever.semantic_retrieval(prompt, k=k, embedding_type='climatebert')
    results['word2vec'] = retriever.semantic_retrieval(prompt, k=k, embedding_type='word2vec')
    results['hybrid'] = retriever.hybrid_retrieval(prompt, k=k)
    
    return results

def cp1a_retrieval(k=25):
    """Specific function for CP1a assessment retrieval"""
    prompt = "Does the country have a decarbonisation strategy to meet Paris Agreement that they are implementing or in the national legislation?"
    return do_retrieval(prompt, method='hybrid', k=k)