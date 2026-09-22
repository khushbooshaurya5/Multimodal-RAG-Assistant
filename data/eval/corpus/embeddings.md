# Text Embeddings

A text embedding maps a sentence or passage to a fixed-length dense vector such that semantically similar texts are close together. Sentence-transformers models such as all-MiniLM-L6-v2 are trained with contrastive objectives on sentence pairs and produce 384-dimensional vectors; larger models such as bge-large or e5-large produce 1024 dimensions with higher accuracy at higher cost.

Cosine similarity between embeddings is the standard relevance score. Because embedding models have a maximum input length (256 word pieces for MiniLM, 512 for many BERT-based models), long documents must be chunked before embedding.

Bi-encoders embed queries and documents independently, so document vectors can be precomputed and indexed. Cross-encoders read the query and the document together and produce a relevance score; they are more accurate but must be run for every candidate pair, so they are used only for reranking a shortlist.

Embeddings can be biased toward lexical overlap and struggle with negation, numbers and very domain-specific vocabulary. Fine-tuning on in-domain pairs usually gives the largest quality gains.
