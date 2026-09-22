# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a retriever with a generative language model. Documents are split into chunks, embedded into vectors and stored in a vector database. At query time the question is embedded, the most similar chunks are retrieved and inserted into the prompt, and the language model writes an answer grounded in that context.

RAG reduces hallucination because the model can quote evidence instead of relying on parametric memory, it lets the knowledge base be updated without retraining, and it makes answers auditable through citations. It does not eliminate hallucination: a model can still ignore the context or misread it.

Chunk size is a key design choice. Small chunks give precise retrieval but may lose context; large chunks preserve context but dilute the embedding. Overlapping chunks help when a fact spans a boundary. Hybrid retrieval combines dense embeddings with lexical BM25 scores, and a cross-encoder reranker can re-score the top candidates for higher precision.

Evaluation of RAG systems typically measures retrieval quality (recall@k, precision@k, mean reciprocal rank) separately from answer quality (faithfulness to the retrieved context, answer relevance and correctness), because a wrong answer can stem from either stage.
