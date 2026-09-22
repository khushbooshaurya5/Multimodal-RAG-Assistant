# Architecture

```
User
 │
 ▼
Input Layer                     app/ (Streamlit)  ·  src/api (FastAPI)
 ├── Text
 ├── Image
 └── Audio
 │
 ▼
Preprocessing
 ├── Whisper  → speech-to-text            src/audio
 ├── Qwen-VL  → image understanding        src/vision
 └── Text / PDF extraction + chunking      src/text
 │
 ▼
Embedding Layer                            src/embeddings
 │
 ▼
FAISS Vector Database (+ metadata store)   src/vectorstore
 │
 ▼
Retriever (top-k, threshold, filters,      src/retrieval
           dedup, optional rerank)
 │
 ▼
Context Assembly + Prompting               src/rag/context.py, src/rag/prompts.py
 │
 ▼
Qwen-VL / LLM generation                   src/rag/generator.py  (backends in src/models)
 │
 ▼
Grounded Response = Answer + Citations + Evidence indicator
```

## Layer responsibilities

| Package | Responsibility | Key types |
| --- | --- | --- |
| `src/config.py` | Single source of truth for configuration (env / `.env`). | `Settings` |
| `src/schemas.py` | Pydantic payloads exchanged between layers. | `Document`, `Chunk`, `RetrievedChunk`, `Transcript`, `ImageAnalysis`, `RAGResponse` |
| `src/models/` | **Model loading only** (Whisper, Qwen-VL, sentence-transformers, OpenAI-compatible HTTP client). No business logic. | `load_whisper`, `load_qwen_vl`, `OpenAICompatibleClient` |
| `src/audio/` | Audio decoding, resampling, transcription backends. | `AudioLoader`, `Transcriber` |
| `src/vision/` | Image loading and VLM-based analysis; converts images into searchable text. | `ImageAnalyzer` |
| `src/text/` | Text / PDF extraction and chunking. | `TextExtractor`, `TextChunker` |
| `src/embeddings/` | Text → vector. Dense (sentence-transformers) or hashing fallback. | `Embedder` |
| `src/vectorstore/` | FAISS index + metadata persistence. | `FaissVectorStore` |
| `src/retrieval/` | Query embedding, top-k search, threshold, metadata filters, dedup, rerank. | `Retriever` |
| `src/rag/` | Ingestion pipeline, context assembly, prompts, generation, evidence indicator, orchestration. | `IngestionPipeline`, `RAGPipeline` |
| `src/api/` | FastAPI server exposing ingest / query endpoints. | `app` |
| `app/` | Streamlit UI. | |

## Design principles

1. **Backends are swappable via environment variables.** Every model-backed
   component (speech, vision, embeddings, generation) is defined by a small
   `Protocol` in its package and selected by a factory that reads `Settings`.
   This lets the same code run with a local Qwen2-VL, a remote vLLM/Ollama
   endpoint, or an explicitly labelled offline fallback.
2. **Fallbacks are honest.** When a neural model is unavailable the system
   still runs (hashing embeddings, metadata-only image analysis, extractive
   evidence-only answers) but every result carries `backend` /
   `is_fallback` flags and the UI displays them. Nothing pretends to be a model
   output.
3. **Metadata never lives only in memory.** The FAISS index stores vectors;
   a JSONL sidecar stores chunk text + metadata keyed by FAISS row id. Both are
   saved and loaded together.
4. **Model loading is separated from business logic.** `src/models/` knows how
   to load weights; `src/audio`, `src/vision`, `src/rag` know what to do with
   them. Business logic can be tested with lightweight test doubles.
5. **Grounding is enforced at three points:** the prompt instructs the model
   to answer only from numbered evidence, the evidence indicator is computed
   from retrieval statistics (not from the model), and citations are shown
   verbatim from the retrieved chunks so users can verify claims.
