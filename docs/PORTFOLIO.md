# Portfolio notes

Copy-ready material for a portfolio site, GitHub profile README, CV or LinkedIn.

## One-paragraph project description

**Multimodal RAG Assistant** — an end-to-end retrieval-augmented generation system that
answers questions from text, images and voice. Whisper transcribes spoken questions with
timestamps and language detection; Qwen2-VL converts images (documents, diagrams, charts,
screenshots) into structured, searchable descriptions; sentence-transformer embeddings are
indexed in FAISS with persisted metadata; a retriever with thresholding, metadata filters,
de-duplication and optional cross-encoder reranking selects evidence; and a grounded
generation layer produces cited answers that explicitly report insufficient evidence.
Ships with a Streamlit UI, a FastAPI backend, 99 tests, CI, Docker, an evaluation harness
(Recall@K, Precision@K, MRR, groundedness, WER, latency) and honest offline fallbacks for
machines without GPUs or model access.

## Résumé bullets

- Designed and built a multimodal RAG assistant (Python, PyTorch, Hugging Face Transformers,
  FAISS, FastAPI, Streamlit) that answers questions over text, images and speech with cited
  evidence; swappable model backends (local Qwen2-VL / Whisper, OpenAI-compatible endpoints,
  documented offline fallbacks) selected through environment configuration.
- Implemented a persisted FAISS vector store with a JSONL metadata sidecar, idempotent
  ingestion and embedder-compatibility checks, plus retrieval quality controls (top-k,
  similarity threshold, metadata filters, near-duplicate removal, cross-encoder reranking).
- Built an evaluation harness measuring Recall@K, Precision@K, MRR, lexical groundedness,
  citation validity, no-evidence detection, WER and per-stage latency; used a threshold sweep
  to choose retrieval defaults and documented limitations.
- Delivered production hygiene: 99 unit and integration tests with protocol-based test
  doubles, ruff, GitHub Actions CI, Dockerfile, typed Pydantic schemas and structured logging.

## Talking points for interviews

- **Why structured image descriptions instead of captions?** Visible text carries the
  entities people search for; a one-line caption retrieves poorly. The analysis is chunked
  like any document, so long OCR output does not overflow the embedding model.
- **How is hallucination controlled?** Prompt rules (answer from numbered evidence, cite,
  admit gaps), a retrieval-derived evidence indicator that is deliberately *not* presented
  as a calibrated probability, and verbatim citations for verification.
- **What did the evaluation show?** With the lexical fallback, recall stays high until the
  threshold reaches ~0.15, after which out-of-corpus questions start being rejected;
  dense embeddings separate far better, which is why the default profile uses 0.25.
- **What would you do next?** Hybrid BM25 + dense retrieval, page-image ingestion for
  scanned PDFs, calibration of the evidence indicator against human labels, and an
  IVF/HNSW index for larger corpora.

## Links to include

- Case-study page (share it from its Share menu before linking publicly): https://claude.ai/artifact/TMmmtWeFNNv697ZQWc72mn
- Repository: https://github.com/khushbooshaurya5/Multimodal-RAG-Assistant
- Live demo: `https://huggingface.co/spaces/<your-hf-username>/multimodal-rag-assistant`
  (create it with `scripts/deploy_hf_space.sh`)
- Evaluation report: `reports/eval_report_offline.md`
- Architecture: `docs/ARCHITECTURE.md`

## GitHub profile README snippet

```markdown
### 🧠 Multimodal RAG Assistant
Qwen-VL + Whisper + FAISS retrieval-augmented generation over text, images and voice.
Grounded, cited answers · swappable model backends · evaluation harness · 99 tests · CI · Docker.
[Code](https://github.com/khushbooshaurya5/Multimodal-RAG-Assistant) · [Live demo](https://huggingface.co/spaces/<your-hf-username>/multimodal-rag-assistant)
```
