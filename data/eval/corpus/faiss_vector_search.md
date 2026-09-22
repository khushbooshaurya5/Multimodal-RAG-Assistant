# FAISS and Vector Search

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports exact search with flat indexes and approximate nearest-neighbour search with structures such as inverted file indexes (IVF), product quantisation (PQ) and hierarchical navigable small world graphs (HNSW).

IndexFlatIP performs exhaustive inner-product search. When vectors are L2-normalised, the inner product equals cosine similarity, which is the usual choice for text embeddings. IndexFlatL2 uses Euclidean distance instead.

IndexIVFFlat partitions the vector space into Voronoi cells using k-means; at query time only the nprobe closest cells are scanned, trading a little recall for large speedups. Product quantisation compresses vectors into short codes so that billions of vectors fit in memory.

FAISS stores only vectors and integer ids. Any metadata such as the source document, page number or timestamp must be stored separately and joined by id. Indexes can be written to disk with write_index and restored with read_index, and IDMap wrappers allow user-defined ids and removal of vectors.
