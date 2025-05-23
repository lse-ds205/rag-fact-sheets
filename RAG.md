Failures of current model:
1.	Multi-hop reasoning failures
a.	When you need to connect information from multiple separate chunks/passages to fully answer a question
b.	Failures occur when your RAG system cannot make these connections

Part #1: Getting the right chunks
1a Embedding strategy 
-	Word2vec
-	Transformer Embeddings (Climate Bert)
-	BM25+ Dense Embeddings (Hybrid Search)
o	BM25  This is a spare keyword search
	Traditional keyword-based ranking function 
	Scores documents based on term frequency and inverse document frequency 
	Creates sparse vectors (mostly zeros) representing keyword presence 
	Focuses on exact lexical matching
o	BM25+ Dense Emebeddings
	Combines both approaches – runs BM25 and vector search in parallel and fuses results with various fusion techniques
	Creates comprehensive ranking to capture both lexical and semantic matches
-	ColBERT (Contextualized Late Interaction over BERT)
o	Multiple token level embeddings rather than single dense vector
o	Maintains fine-grained itneractions crucial for technical terminology

1b RAG Structure
-	Non-structured RAG 
o	Simply adopts sparse or dense retrievers
-	Tree-structured RAG
o	Focuses on the hierarchical logic of passages within a single document – ignoring relations between the hierarchical structure or across documents
-	Graph structured RAG
o	Models logical relations in the most ideal form by constructing knowledge graphs to represent documents
o	Shortcoming: Their reliance on predefined schemas limits the flexible expressive capability
-	HopRAG
o	Creates a passage graph where documents are connected by logical relationships, allowing the system to hop between related passages to build comprehensive answers
1c Similarity Search Strategy
-	Hierarchical Navigable Small World (HNSW) implementation
o	multi-layered graph structures achieving 90%+ recall with 10x faster search than brute force methods
o	configure with M=16-32 edges per element and efConstruction=200-400 for optimal performance.
-	ScaNN (Scalable Nearest Neighbors)
o	Provides 2x – 3x faster performance than HNSW on datasets exceeding 10M vectors through asymmetric hashing and quantization
-	Semantic matching for paraphrasing
o	Augmented SBERT addresses paraphrasing challenge by using cross-encoders to generate training data for bi-encoders
Construct a validation score – give the LLM the answer and question and generate the score 

Part #2: Generating a response to the questions with the right chunks
Recent work by Jeong et al. (2024) introduces automatic query complexity classification that routes queries to appropriate strategies:
1.	Type A: Direct LLM generation for simple factual queries
2.	Type B: Single-step retrieval for moderate complexity
3.	Type C: Multi-step iterative retrieval for complex multi-hop reasoning
Construct a validation score – give the LLM the answer and question and generate the score 
