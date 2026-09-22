"""Run the evaluation dataset through an :class:`AssistantService` and aggregate metrics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.metrics import (
    citation_validity,
    groundedness,
    mean_reciprocal_rank,
    percentile,
    phrase_coverage,
    precision_at_k,
    recall_at_k,
    word_error_rate,
)
from src.rag.service import AssistantService
from src.schemas import EvidenceLevel
from src.utils.logging import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT = """You are grading whether an answer is supported by evidence.

EVIDENCE:
{evidence}

ANSWER:
{answer}

Rate from 0 to 1 how much of the answer's factual content is directly supported by the evidence
(1 = every claim is supported, 0 = nothing is supported). Reply with only the number."""


@dataclass(slots=True)
class QuestionResult:
    id: str
    question: str
    relevant_sources: list[str]
    retrieved_sources: list[str]
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    groundedness: float | None
    citation_validity: float | None
    phrase_coverage: float | None
    evidence_level: str
    expected_no_evidence: bool
    no_evidence_correct: bool | None
    llm_judge_score: float | None
    latency_ms: dict[str, float]
    answer: str


@dataclass(slots=True)
class AudioResult:
    id: str
    audio: str
    reference_transcript: str
    hypothesis: str
    wer: float
    recall_at_k: float | None
    latency_ms: dict[str, float]


@dataclass(slots=True)
class EvaluationReport:
    created_at: str
    backends: dict[str, Any]
    top_k: int
    similarity_threshold: float
    num_questions: int
    num_audio_examples: int
    aggregate: dict[str, float | None]
    latency_ms: dict[str, dict[str, float | None]]
    questions: list[QuestionResult] = field(default_factory=list)
    audio: list[AudioResult] = field(default_factory=list)
    skipped_audio: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationRunner:
    """Ingests the evaluation corpus and scores every question."""

    def __init__(
        self,
        service: AssistantService,
        dataset_path: Path,
        *,
        top_k: int | None = None,
        use_llm_judge: bool = False,
    ) -> None:
        self.service = service
        self.dataset_path = Path(dataset_path)
        self.dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        self.top_k = top_k or service.settings.top_k
        self.use_llm_judge = use_llm_judge and not service.generator.is_fallback
        if use_llm_judge and service.generator.is_fallback:
            logger.warning(
                "LLM judge requested but the generator is the extractive fallback; skipping judge"
            )

    # --- Corpus -----------------------------------------------------------

    def ingest_corpus(self) -> None:
        corpus_dir = self.dataset_path.parent / self.dataset.get("corpus_dir", "corpus")
        results = self.service.ingest_directory(corpus_dir)
        failed = [r for r in results if r.error]
        if failed:
            raise RuntimeError(
                "Corpus ingestion failed: " + "; ".join(f"{r.source}: {r.error}" for r in failed)
            )
        logger.info("Evaluation corpus indexed: %d files", len(results))

    # --- Scoring ----------------------------------------------------------

    def _judge(self, answer: str, evidence: str) -> float | None:
        if not self.use_llm_judge:
            return None
        try:
            reply = self.service.generator.generate(
                JUDGE_PROMPT.format(evidence=evidence, answer=answer)
            )
            match = re.search(r"[01](?:\.\d+)?", reply)
            return float(match.group()) if match else None
        except Exception as exc:  # judge failures must not abort the run
            logger.warning("LLM judge failed: %s", exc)
            return None

    def evaluate_question(self, item: dict[str, Any]) -> QuestionResult:
        response = self.service.query(item["question"], top_k=self.top_k)
        retrieved_sources = [c.source for c in response.citations]
        relevant = list(item.get("relevant_sources", []))
        passages = [r.text for r in response.retrieved]
        expected_none = bool(item.get("expected_no_evidence", False))
        return QuestionResult(
            id=item["id"],
            question=item["question"],
            relevant_sources=relevant,
            retrieved_sources=retrieved_sources,
            recall_at_k=recall_at_k(retrieved_sources, relevant, self.top_k),
            precision_at_k=precision_at_k(retrieved_sources, relevant, self.top_k),
            mrr=mean_reciprocal_rank(retrieved_sources, relevant),
            groundedness=groundedness(response.answer, passages),
            citation_validity=citation_validity(response.answer, len(response.citations)),
            phrase_coverage=phrase_coverage(response.answer, item.get("expected_phrases", [])),
            evidence_level=response.evidence.level.value,
            expected_no_evidence=expected_none,
            no_evidence_correct=(response.evidence.level == EvidenceLevel.NONE)
            if expected_none
            else None,
            llm_judge_score=self._judge(response.answer, "\n\n".join(passages))
            if passages
            else None,
            latency_ms=dict(response.timings_ms),
            answer=response.answer,
        )

    def evaluate_audio(self, item: dict[str, Any]) -> AudioResult | None:
        path = self.dataset_path.parent / item["audio"]
        if not path.is_file() or self.service.transcriber is None:
            return None
        clip = self.service.load_audio(path.read_bytes(), path.name)
        response = self.service.query(None, audio=clip, top_k=self.top_k)
        hypothesis = response.transcript.text if response.transcript else ""
        return AudioResult(
            id=item["id"],
            audio=item["audio"],
            reference_transcript=item["reference_transcript"],
            hypothesis=hypothesis,
            wer=word_error_rate(item["reference_transcript"], hypothesis),
            recall_at_k=recall_at_k(
                [c.source for c in response.citations], item.get("relevant_sources", []), self.top_k
            ),
            latency_ms=dict(response.timings_ms),
        )

    def run(self) -> EvaluationReport:
        self.ingest_corpus()
        questions = [self.evaluate_question(q) for q in self.dataset["questions"]]
        audio_results: list[AudioResult] = []
        skipped: list[str] = []
        for item in self.dataset.get("audio_examples", []):
            result = self.evaluate_audio(item)
            if result is None:
                skipped.append(item["id"])
            else:
                audio_results.append(result)
        return self._aggregate(questions, audio_results, skipped)

    def _aggregate(
        self, questions: list[QuestionResult], audio: list[AudioResult], skipped: list[str]
    ) -> EvaluationReport:
        def avg(values: list[float | None]) -> float | None:
            present = [v for v in values if v is not None]
            return round(mean(present), 4) if present else None

        answerable = [q for q in questions if not q.expected_no_evidence]
        unanswerable = [q for q in questions if q.expected_no_evidence]
        aggregate: dict[str, float | None] = {
            f"recall@{self.top_k}": avg([q.recall_at_k for q in answerable]),
            f"precision@{self.top_k}": avg([q.precision_at_k for q in answerable]),
            "mrr": avg([q.mrr for q in answerable]),
            "groundedness_lexical": avg([q.groundedness for q in answerable]),
            "citation_validity": avg([q.citation_validity for q in answerable]),
            "expected_phrase_coverage": avg([q.phrase_coverage for q in answerable]),
            "no_evidence_detection_rate": avg(
                [
                    float(q.no_evidence_correct)
                    for q in unanswerable
                    if q.no_evidence_correct is not None
                ]
            ),
            "llm_judge_groundedness": avg([q.llm_judge_score for q in answerable]),
            "audio_wer": avg([a.wer for a in audio]),
            f"audio_recall@{self.top_k}": avg([a.recall_at_k for a in audio]),
        }
        stages = sorted(
            {k for q in questions for k in q.latency_ms} | {k for a in audio for k in a.latency_ms}
        )
        latency: dict[str, dict[str, float | None]] = {}
        for stage in stages:
            values = [q.latency_ms[stage] for q in questions if stage in q.latency_ms] + [
                a.latency_ms[stage] for a in audio if stage in a.latency_ms
            ]
            latency[stage] = {
                "mean": round(mean(values), 2) if values else None,
                "p50": round(percentile(values, 50) or 0.0, 2) if values else None,
                "p95": round(percentile(values, 95) or 0.0, 2) if values else None,
            }
        totals = [sum(q.latency_ms.values()) for q in questions]
        latency["total"] = {
            "mean": round(mean(totals), 2) if totals else None,
            "p50": round(percentile(totals, 50) or 0.0, 2) if totals else None,
            "p95": round(percentile(totals, 95) or 0.0, 2) if totals else None,
        }
        return EvaluationReport(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            backends=self.service.status().as_dict(),
            top_k=self.top_k,
            similarity_threshold=self.service.settings.similarity_threshold,
            num_questions=len(questions),
            num_audio_examples=len(audio),
            aggregate=aggregate,
            latency_ms=latency,
            questions=questions,
            audio=audio,
            skipped_audio=skipped,
        )


def render_markdown(report: EvaluationReport) -> str:
    """Render a human-readable Markdown report."""
    b = report.backends
    fallbacks = [k for k, v in b.get("fallbacks", {}).items() if v]
    lines = [
        "# Evaluation report",
        "",
        f"Generated {report.created_at} · top-k={report.top_k} · threshold={report.similarity_threshold}",
        "",
        "## Configuration",
        "",
        f"- Embeddings: `{b.get('embedder')}`",
        f"- Generator: `{b.get('generator')}`",
        f"- Vision: `{b.get('vision')}` · Speech: `{b.get('transcriber') or 'disabled'}` · Reranker: `{b.get('reranker') or 'off'}`",
        f"- Device: `{b.get('device')}`",
    ]
    if fallbacks:
        lines.append(
            f"- ⚠️ **Fallback backends active:** {', '.join(fallbacks)}. Numbers below reflect the offline "
            "fallbacks (lexical hashing retrieval / extractive answers), not the neural models."
        )
    lines += ["", "## Aggregate metrics", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in report.aggregate.items():
        lines.append(f"| {key} | {'n/a' if value is None else f'{value:.3f}'} |")
    lines += [
        "",
        "## Latency (ms)",
        "",
        "| Stage | mean | p50 | p95 |",
        "| --- | --- | --- | --- |",
    ]
    for stage, stats in report.latency_ms.items():
        fmt = lambda v: "n/a" if v is None else f"{v:.1f}"  # noqa: E731
        lines.append(
            f"| {stage} | {fmt(stats['mean'])} | {fmt(stats['p50'])} | {fmt(stats['p95'])} |"
        )
    lines += [
        "",
        "## Per-question results",
        "",
        "| id | R@k | P@k | MRR | grounded | phrases | evidence | retrieved sources |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for q in report.questions:
        f = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
        if q.expected_no_evidence:
            verdict = "✅ none" if q.no_evidence_correct else f"❌ {q.evidence_level}"
            lines.append(
                f"| {q.id} (unanswerable) | – | – | – | – | – | {verdict} | {', '.join(q.retrieved_sources) or '–'} |"
            )
        else:
            lines.append(
                f"| {q.id} | {f(q.recall_at_k)} | {f(q.precision_at_k)} | {f(q.mrr)} | {f(q.groundedness)} | "
                f"{f(q.phrase_coverage)} | {q.evidence_level} | {', '.join(q.retrieved_sources) or '–'} |"
            )
    lines += ["", "## Audio examples", ""]
    if report.audio:
        lines += ["| id | WER | R@k | transcript |", "| --- | --- | --- | --- |"]
        for a in report.audio:
            lines.append(
                f"| {a.id} | {a.wer:.3f} | {'–' if a.recall_at_k is None else f'{a.recall_at_k:.2f}'} | {a.hypothesis} |"
            )
    else:
        lines.append(
            "No audio examples were evaluated (audio files missing or speech backend disabled)."
        )
    if report.skipped_audio:
        lines.append(f"\nSkipped audio examples: {', '.join(report.skipped_audio)}")
    lines += [
        "",
        "## How to read these numbers",
        "",
        "- Retrieval metrics are computed at the source-file level against hand-labelled relevant files.",
        "- `groundedness_lexical` is a lexical proxy (share of answer sentences whose content words are mostly "
        "present in a retrieved passage). It flags unsupported text but cannot detect subtle misreadings.",
        "- `expected_phrase_coverage` checks that key facts appear in the answer; it is a correctness proxy, not a full judgement.",
        "- `no_evidence_detection_rate` is the share of out-of-corpus questions for which the system reported no evidence.",
        "- Latencies are wall-clock milliseconds on the evaluation machine and depend on hardware and backends.",
    ]
    return "\n".join(lines) + "\n"
