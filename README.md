**Semantic Search and Embedding Analysis using Sentence Transformers**

**Problem**

Modern AI systems rely on understanding semantic similarity between text.
Traditional keyword search fails to capture meaning.

**Solution**

This project demonstrates how to:
- Convert text into embeddings using transformer models
- Measure semantic similarity between documents
- Visualise clustering of related topics
- Lay the foundation for semantic search systems


**What this shows**
- Papers from similar domains cluster together in embedding space
- Semantic similarity is captured beyond keywords
- Dimensionality reduction helps visualise high-dimensional data

**Architecture**
- Embedding model: SentenceTransformers (MiniLM)
- Dimensionality reduction: PCA
- Visualisation: Matplotlib + Plotly

**Key Concepts Demonstrated**
- Embeddings as vector representations of meaning
- Similarity via distance in vector space
- Trade-offs in dimensionality reduction
- Clustering behaviour of semantic data

**Applications**
- Semantic search
- Recommendation systems
- Document clustering
- Retrieval-augmented generation (RAG)

**Limitations**
- Small dataset
- PCA loses information
- No real-time search system implemented

**Next Steps**
- Build semantic search API
- Add vector database (FAISS / Pinecone)
- Implement query-to-document matching

<img width="1183" height="878" alt="image" src="https://github.com/user-attachments/assets/e2c401d2-b1f4-4c13-aedd-d3047e15eef0" />

<img width="1183" height="878" alt="image" src="https://github.com/user-attachments/assets/c2ae4c3c-04ea-4b0a-9a8c-7a716f5bafd5" />
