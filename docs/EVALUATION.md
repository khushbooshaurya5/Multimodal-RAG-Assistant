# Evaluation methodology

`scripts/evaluate.py` indexes the corpus under `data/eval/corpus/` into a temporary
index, runs every question in `data/eval/eval_dataset.json` through the full
pipeline and writes `reports/eval_report<tag>.{json,md}`.

## Dataset

* 7 short, hand-written documents (CNNs, transformers, Whisper, FAISS, RAG,
  embeddings, Qwen-VL, MLOps notes).
* 14 answerable questions, each labelled with the **source files** that contain
  the answer and 1–2 **key phrases** a correct grounded answer should include.
* 2 out-of-corpus questions (`expected_no_evidence: true`) that test whether the
  system reports *no evidence* instead of guessing.
* 1 audio example with a reference transcript. No spoken audio is shipped; record
  the sentence yourself to `data/eval/audio/pooling_question.wav` and the runner
  will compute WER and retrieval recall for it (skipped otherwise).

## Metrics

| Metric | Definition | Why |
| --- | --- | --- |
| Recall@K | Share of labelled relevant *files* that appear among the top-K retrieved chunks' sources. | Did retrieval find the evidence at all? |
| Precision@K | Share of distinct top-K sources that are relevant. | How much noise reaches the prompt? |
| MRR | 1 / rank of the first relevant source. | Is the best evidence at the top? |
| Groundedness (lexical) | Share of answer sentences whose content words are ≥60 % covered by some retrieved passage. | Flags answer text that is not in the evidence. Lexical proxy, **not** a judgement of correctness. |
| Citation validity | Share of `[n]` markers that refer to an existing citation. | Catches invented citations. |
| Expected-phrase coverage | Share of labelled key phrases present in the answer. | Cheap correctness proxy. |
| No-evidence detection rate | Share of out-of-corpus questions for which the evidence indicator is *none*. | Does the threshold stop guessing? |
| LLM-judge groundedness (`--judge`) | The configured LLM rates 0–1 how much of the answer the evidence supports. | Only when a real LLM backend is configured; unavailable in the offline profile. |
| Audio WER | Word-level edit distance / reference length. | Transcription quality where references exist. |
| Latency | Mean / p50 / p95 per stage (transcription, image analysis, retrieval, generation). | Where the time goes. |

Retrieval is scored at the file level because the labels are file-level; chunk-level
relevance would need per-chunk annotation.

## Baseline result (offline profile, this repository's CI environment)

The committed `reports/eval_report_offline.md` was produced with the **offline
fallbacks** (hashing embeddings, extractive answers, no vision, no speech) on a
4-core CPU container without network access. It measures the plumbing and the
lexical baseline, not the neural models.

Threshold sweep for the hashing embedder (top-k = 5):

| threshold | Recall@5 | Precision@5 | MRR | phrase coverage | no-evidence detection |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.93 | 0.32 | 0.84 | 0.82 | 0.00 |
| 0.10 | 0.93 | 0.34 | 0.84 | 0.82 | 0.00 |
| **0.15** | **0.86** | **0.57** | **0.82** | **0.64** | **0.50** |
| 0.20 | 0.71 | 0.66 | 0.71 | 0.57 | 1.00 |
| 0.25 | 0.43 | 0.39 | 0.43 | 0.32 | 1.00 |

0.15 is the default in `configs/offline.env`: it keeps most recall while starting
to reject out-of-corpus questions. Dense embeddings (MiniLM) separate relevant
from irrelevant far better, which is why `configs/default.env` uses 0.25.

## Limitations

* **Tiny dataset.** 16 questions over 7 documents cannot distinguish small
  differences between configurations; it catches regressions, not nuance.
* **Lexical groundedness is an upper bound.** A sentence built from evidence
  words in a misleading order still counts as supported. Use `--judge` with a real
  LLM, or human review, for faithfulness claims.
* **Extractive answers are trivially grounded** (they are quotes), so the offline
  groundedness of 1.0 says nothing about an LLM's behaviour.
* **No calibration.** The evidence indicator bands are hand-picked; the report
  does not measure whether "strong" answers are actually more often correct.
* **Audio and image quality are not measured here** unless you add spoken audio
  and run with Whisper / Qwen-VL configured. The image use case is covered by
  integration tests with a stub VLM and by the `requires_models` tests.
* **Latency numbers are environment-specific** and, in the offline profile,
  exclude model inference entirely.
