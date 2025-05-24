# 🔍 RAG System Documentation

## 🚨 Failures of Current Model
1. **Multi-hop reasoning failures** 🔗
   - When you need to connect information from multiple separate chunks/passages to fully answer a question
   - Failures occur when your RAG system cannot make these connections

---

## 📊 Part #1: Getting the Right Chunks

### 🧠 1a Embedding Strategy 
- **Word2vec** 📝
- **Transformer Embeddings (Climate Bert)** 🌍
  - Uses pre-trained BERT model fine-tuned on climate policy data
  - Captures contextual relationships between words and phrases
  - Generates dense vector representations of text chunks (dimension: 768)
- **BM25+ Dense Embeddings (Hybrid Search)** 🔍
  - **BM25** - This is a sparse keyword search
    - Traditional keyword-based ranking function 
    - Scores documents based on term frequency and inverse document frequency 
    - Creates sparse vectors (mostly zeros) representing keyword presence 
    - Focuses on exact lexical matching
  - **BM25+ Dense Embeddings**
    - Combines both approaches – runs BM25 and vector search in parallel and fuses results with various fusion techniques
    - Creates comprehensive ranking to capture both lexical and semantic matches
- **ColBERT (Contextualized Late Interaction over BERT)** 🤖
  - Multiple token level embeddings rather than single dense vector
  - Maintains fine-grained interactions crucial for technical terminology

### 🏗️ 1b RAG Structure
- **Non-structured RAG** 📄
  - Simply adopts sparse or dense retrievers
- **Tree-structured RAG** 🌳
  - Focuses on the hierarchical logic of passages within a single document – ignoring relations between the hierarchical structure or across documents
- **Graph structured RAG** 🕸️
  - Models logical relations in the most ideal form by constructing knowledge graphs to represent documents
  - **Shortcoming:** Their reliance on predefined schemas limits the flexible expressive capability
- **HopRAG** 🦘
  - Creates a passage graph where documents are connected by logical relationships, allowing the system to hop between related passages to build comprehensive answers

### 🎯 1c Similarity Search Strategy
- **Hierarchical Navigable Small World (HNSW) implementation** 🌐
  - Multi-layered graph structures achieving 90%+ recall with 10x faster search than brute force methods
  - Configure with M=16-32 edges per element and efConstruction=200-400 for optimal performance
- **ScaNN (Scalable Nearest Neighbors)** ⚡
  - Provides 2x – 3x faster performance than HNSW on datasets exceeding 10M vectors through asymmetric hashing and quantization
- **Semantic matching for paraphrasing** 💬
  - Augmented SBERT addresses paraphrasing challenge by using cross-encoders to generate training data for bi-encoders

**Construct a validation score** ✅ – give the LLM the answer and question and generate the score 

### **Optimal Implementation - Traditional RAG + HopRAG**

Current implementation includes both traditional RAG architecture AND HopRAG for enhanced multi-hop reasoning.

#### HopRAG Process:
1. **Vector embeddings of text chunks** 📊
   - Generate dense vector representations of document passages
2. **Logical relationship detection between chunks** 🔗
   - Identify semantic connections and dependencies between passages
3. **BFS traversal graph algorithm** 🌐
   - Build relationships using breadth-first search with O(V) complexity

#### Benefits of HopRAG:
1. **Path Diversity** 🛤️
   - HopRAG provides both cosine similarity scores and path scores for comprehensive ranking
   - Multiple pathways to relevant information increase answer completeness
2. **Transitivity Capture** ↔️
   - Captures indirect relationships where A→B and B→C implies A relates to C
   - Essential for complex reasoning chains
3. **Context Propagation** 📡
   - Information flows through the graph structure, maintaining context across hops
   - Preserves semantic coherence in multi-step reasoning
4. **Mathematical Foundation** 🔢
   - Graph theory provides the mathematical foundation for HopRAG - we have opted to use BFS algorithm
   - BFS is optimal for unweighted graphs, ensuring efficient traversal
   - Enables formal analysis of relationship strength and path optimization

Question: "How does BFS algorithm work?"
Answer: "The BFS algorithm explores all neighbors at the present depth prior to moving on to nodes at the next depth level. It uses a queue data structure to keep track of nodes to be explored, ensuring that all nodes at the current level are processed before moving deeper into the graph. This approach guarantees that the shortest path in an unweighted graph is found."

Question: "I heard that Neo4J is the up and coming graph database! Why not do a graph construction in this case?"
Answer: "Graph implementation requires a O(n^2) complexity for the construction of the graph. This is not feasible for large datasets (>100k chunks). The BFS algorithm is optimal for unweighted graphs, ensuring efficient traversal. It enables formal analysis of relationship strength and path optimization. Neo4J is a great tool, but it may not be the best fit for this specific use case due to its complexity and overhead."

---

## 💭 Part #2: Generating a Response to the Questions with the Right Chunks

Recent work by **Jeong et al. (2024)** introduces automatic query complexity classification that routes queries to appropriate strategies:

1. **Type A:** Direct LLM generation for simple factual queries 🟢
2. **Type B:** Single-step retrieval for moderate complexity 🟡
3. **Type C:** Multi-step iterative retrieval for complex multi-hop reasoning 🔴

**Construct a validation score** ✅ – give the LLM the answer and question and generate the score