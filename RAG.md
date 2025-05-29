# 🔍 RAG System Documentation

This document outlines the architecture and design decisions for the Retrieval-Augmented Generation (RAG) system used in our project. The RAG system is designed to efficiently retrieve relevant information from a large corpus of documents and generate accurate responses to user queries.

This is the compilation of research on the RAG system architecture and design that we performed for the TPI project. The goal of this document is to provide a comprehensive overview of the RAG system architecture and design, including the technical rationale behind our decisions.


## 🚨 Failures of Current Model
1. **Multi-hop reasoning failures** 🔗
   - When you need to connect information from multiple separate chunks/passages to fully answer a question
   - Failures occur when your RAG system cannot make these connections
2. **Query to Document Mapping** 📚
    - For many students, when they attempted to do a cosine similarity search, they found that the results were synonyms or paraphases of the original query, but not providing relevant information to answer the question itself.
    - We decided to use a hybrid approach of BM25 and dense embeddings to ensure that we capture both lexical and semantic matches (Regex + Cosine similarity + Fuzzy Regex Comparison).
---

## 📊 Part #1: Getting the Right Chunks

### 🧠 1a Embedding Strategy 
- **Word2vec** 📝
- **Transformer Embeddings (Climate Bert)** 🌍
  - Uses pre-trained BERT model fine-tuned on climate policy data
  - Captures contextual relationships between words and phrases
  - Generates dense vector representations of text chunks (dimension: 768)
- **BM25+ Dense Embeddings (Hybrid Search)** 🔍🔄
  - **BM25** 📊
    - Traditional keyword-based ranking function ✨
    - Scores documents based on term frequency and inverse document frequency 📈
    - Creates sparse vectors (mostly zeros) representing keyword presence 0️⃣
    - Focuses on exact lexical matching 🎯
  - **BM25+ Dense Embeddings** 🔄
    - Combines both approaches – runs BM25 and vector search in parallel and fuses results with various fusion techniques 🔀
    - Creates comprehensive ranking to capture both lexical and semantic matches 📋
- **ColBERT (Contextualized Late Interaction over BERT)** 🤖🧩
  - Multiple token level embeddings rather than single dense vector 🧠
  - Maintains fine-grained interactions crucial for technical terminology 🔬

### 🏗️ 1b RAG Structure
- **Non-structured RAG** 📄🧩
  - Combines multiple retrieval methods:
    - Vector-based similarity using embeddings 📊
    - Pattern-based matches using regex and fuzzy matching 🔍
    - Keyword extraction for specialized climate terminology 🌿
- **Tree-structured RAG** 🌳📚
  - Focuses on the hierarchical logic of passages within a single document – ignoring relations between the hierarchical structure or across documents 📑
- **Graph structured RAG** 🕸️
  - Models logical relations in the most ideal form by constructing knowledge graphs to represent documents
  - **Shortcoming:** Their reliance on predefined schemas limits the flexible expressive capability
- **HopRAG** 🦘
  - Creates a passage graph where documents are connected by logical relationships
  - Implements optimized BFS (Breadth-First Search) traversal to find relevant connected passages
  - Uses UUID consistency and memory optimization techniques for scalability
  - Combines graph analysis with semantic relevance for robust multi-hop reasoning
  - Identifies and traverses semantic connections that traditional RAG would miss
  - Crucial for climate policy context where information is often spread across multiple documents

### 🎯 1c Similarity Search Strategy
- **Vector-based similarity search** 🔢
  - **Cosine Similarity** - Calculates similarity between vectors by measuring the cosine of the angle between them
  - Primary similarity metric for both transformer and word2vec embeddings
  - Implementation uses PostgreSQL pgvector extension for efficient indexing and search
- **Regex-based pattern matching** 🔍
  - Direct keyword matching with climate-specific terminology categories
  - Numerical pattern detection for percentages, years, and emission amounts
  - Query-specific keyword matching to align with user intent
- **Fuzzy Regex Comparison** 🧩
  - Implements advanced text matching beyond exact keyword matching
  - Combines n-gram analysis, fuzzy matching, and contextual pattern recognition
  - Especially effective for detecting semantic matches with varying terminology
  - Uses configurable threshold for matching flexibility


## FINAL DECISION: TRADITIONAL RAG ARCHITECTURE + HOPRAG

**Technical Rationale**

1. Document Scope Optimization 📋
- Most queries are intra-document, making full tree or graph traversal unnecessary overhead.
- HopRAG offers targeted cross-document reasoning only when essential, unlike Tree/Graph RAGs that maintain global relationships by default.
- Enables a lightweight, efficient retrieval strategy where high-complexity methods can be involved only when needed.

2. Infrastructure Constraints 🗄️
- Uses PostgreSQL on Supabase, favoring relational data over native graph structures.
- HopRAG’s relationships can be mapped as relational joins, avoiding the need for specialized graph databases required by Graph RAG.
- Avoids complex tree encoding and traversal logic, which can be rigid and over-engineered for this use case.

3. Usage Pattern Analysis ⏱️
- With low-frequency, batch-style usage, the overhead of maintaining persistent graph or tree states is wasteful.
- HopRAG allows on-demand reasoning during execution, without maintaining a complex knowledge graph between runs.
- Better suited for cost-conscious, infrequent workloads compared to constantly maintained Graph/Tree RAGs.

4. Resource Efficiency 💰
- Combines similarity search, regex, and fuzzy regex to efficiently resolve the vast majority of queries without recursion or traversal.
- HopRAG adds multi-hop reasoning as an additional layer that can be switched on or off, preserving performance and compute resources.
- Avoids the high processing and memory costs of Tree/Graph RAG models while still offering advanced reasoning as a complementary tool.


Current implementation includes both traditional RAG architecture AND HopRAG for enhanced multi-hop reasoning.

We use a **hybrid search strategy** that combines:
1. **BM25** for keyword-based retrieval &  **Dense embeddings** for semantic similarity
2. **HopRAG** for logical relationship traversal

This approach feeds the top-ranked chunks from both methods into the LLM for response generation, ensuring comprehensive coverage of both lexical and semantic information.


### **Diving Deeper into HopRAG**

#### HopRAG Process:
1. **Vector embeddings of text chunks** 📊
   - Generate dense vector representations of document passages
2. **Logical relationship detection between chunks** 🔗
   - Identify semantic connections and dependencies between passages
3. **BFS traversal graph algorithm** 🌐
   - Build relationships using breadth-first search with O(V+E) complexity

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

Question: "How does the BFS algorithm work?"

Answer: "The BFS algorithm explores all neighbors at the present depth prior to moving on to nodes at the next depth level. It uses a queue data structure to keep track of nodes to be explored, ensuring that all nodes at the current level are processed before moving deeper into the graph. This approach guarantees that the shortest path in an unweighted graph is found."

Question: "I heard that Neo4J is the up and coming graph database! Why not do a graph construction in this case?"

Answer: "Graph implementation requires a O(n^2) complexity for the construction of the graph. This is not feasible for large datasets (>100k chunks). The BFS algorithm is optimal for unweighted graphs, ensuring efficient traversal. It enables formal analysis of relationship strength and path optimization. Neo4J is a great tool, but it may not be the best fit for this specific use case due to its complexity and overhead."

---

## 💭 Part #2: Generating a Response to the Questions with the Right Chunks

### 🧠 Prompt Engineering

- **Prompt Engineering** 📝 `src/constants/prompts.py`
  - Crafting effective prompts to guide LLMs in generating accurate responses
  - Using structured templates to ensure clarity and relevance
  - Incorporating context from retrieved chunks to enhance answer quality
- **Structured Response Generation** 🧩 `src/query.py`
  - Implements guided JSON response format to ensure consistent, parseable outputs
  - Robust fallback mechanisms for handling API limitations with explicit JSON parsing
  - Advanced citation validation that enriches references with original chunk metadata
  - Maintains citation provenance through UUID tracking for complete auditability
- **Quality Assurance Framework** ⚖️ `group4py/src/query.py`
  - Sophisticated confidence classification with high/medium/low bands based on retrieval metrics
  - Multi-factor scoring combining vector similarity, regex matches, and fuzzy pattern recognition
  - Graceful error handling with informative diagnostics for each processing stage
  - Cross-reference validation between retrieved chunks and LLM-generated citations

### Future Work

#### Dynamic Query Complexity Classification 🔄
Recent work by Jeong et al. (2024) introduces automatic query complexity classification that routes queries to appropriate strategies:

Type A: Direct LLM generation for simple factual queries 🟢
Type B: Single-step retrieval for moderate complexity 🟡
Type C: Multi-step iterative retrieval for complex multi-hop reasoning 🔴
This classification can be integrated into the RAG system to dynamically adjust retrieval strategies based on query complexity, enhancing efficiency and accuracy. This means that for different types of questions, we can use different retrieval strategies. This will be beneficial to the entire architecture because it will allow us to use the most appropriate retrieval strategy for the question at hand - therefore reducing costs and complexity.

#### Validation Score Construction 🏆

Construct a validation score ✅ – give the LLM the answer and question and generate a score + classification.

This simple methodology allows us to classify answer and responses into 3 categories: low confidence, medium confidence, and high confidence. Understanding that this pipeline will likely work together with a human TPI analyst, once the pipeline has been tried and tested and elevated to a production level, we can leverage this classification such that the TPI analyst can focus on the low confidence responses and the medium confidence responses, while the high confidence responses can be automatically sent to the user.
