# Evaluation report

Generated 2026-09-22T00:08:52+00:00 · top-k=5 · threshold=0.15

## Configuration

- Embeddings: `hashing:384`
- Generator: `extractive`
- Vision: `metadata_only` · Speech: `disabled` · Reranker: `off`
- Device: `cpu`
- ⚠️ **Fallback backends active:** embedder, generator, vision, audio_disabled. Numbers below reflect the offline fallbacks (lexical hashing retrieval / extractive answers), not the neural models.

## Aggregate metrics

| Metric | Value |
| --- | --- |
| recall@5 | 0.857 |
| precision@5 | 0.571 |
| mrr | 0.821 |
| groundedness_lexical | 1.000 |
| citation_validity | 1.000 |
| expected_phrase_coverage | 0.643 |
| no_evidence_detection_rate | 0.500 |
| llm_judge_groundedness | n/a |
| audio_wer | n/a |
| audio_recall@5 | n/a |

## Latency (ms)

| Stage | mean | p50 | p95 |
| --- | --- | --- | --- |
| generation | 0.1 | 0.1 | 0.2 |
| retrieval | 0.5 | 0.5 | 0.8 |
| total | 0.7 | 0.7 | 0.9 |

## Per-question results

| id | R@k | P@k | MRR | grounded | phrases | evidence | retrieved sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q01 | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 | moderate | cnn_basics.md, qwen_vl.md |
| q02 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | moderate | cnn_basics.md |
| q03 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | strong | cnn_basics.md, embeddings.md, transformers.md |
| q04 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | moderate | transformers.md, transformers.md, transformers.md, transformers.md |
| q05 | 1.00 | 0.33 | 0.50 | 1.00 | 1.00 | moderate | qwen_vl.md, transformers.md, transformers.md, whisper.md |
| q06 | 1.00 | 0.50 | 0.50 | 1.00 | 1.00 | moderate | transformers.md, whisper.md |
| q07 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | moderate | whisper.md, whisper.md |
| q08 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 | moderate | faiss_vector_search.md, faiss_vector_search.md, transformers.md, embeddings.md |
| q09 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | moderate | faiss_vector_search.md |
| q10 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | strong | rag_overview.md, qwen_vl.md, rag_overview.md, rag_overview.md, qwen_vl.md |
| q11 | 0.50 | 0.25 | 1.00 | 1.00 | 0.00 | moderate | rag_overview.md, rag_overview.md, transformers.md, cnn_basics.md, qwen_vl.md |
| q12 | 0.50 | 0.25 | 0.50 | 1.00 | 0.00 | strong | transformers.md, transformers.md, rag_overview.md, whisper.md, cnn_basics.md |
| q13 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | moderate | qwen_vl.md |
| q14 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | weak | faiss_vector_search.md, whisper.md, transformers.md, rag_overview.md |
| q15 (unanswerable) | – | – | – | – | – | ❌ weak | faiss_vector_search.md |
| q16 (unanswerable) | – | – | – | – | – | ✅ none | – |

## Audio examples

No audio examples were evaluated (audio files missing or speech backend disabled).

Skipped audio examples: a01

## How to read these numbers

- Retrieval metrics are computed at the source-file level against hand-labelled relevant files.
- `groundedness_lexical` is a lexical proxy (share of answer sentences whose content words are mostly present in a retrieved passage). It flags unsupported text but cannot detect subtle misreadings.
- `expected_phrase_coverage` checks that key facts appear in the answer; it is a correctness proxy, not a full judgement.
- `no_evidence_detection_rate` is the share of out-of-corpus questions for which the system reported no evidence.
- Latencies are wall-clock milliseconds on the evaluation machine and depend on hardware and backends.
