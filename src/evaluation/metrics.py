"""Metric implementations.

Retrieval metrics are computed at the **source (file) level** because the
evaluation set labels which documents contain the answer, not which chunk.

Groundedness is a **lexical proxy**: an answer sentence counts as supported
when most of its content words appear in at least one retrieved passage.
This catches text that was not in the evidence, but it cannot detect subtle
misreadings, so it is an upper bound on faithfulness rather than a judgement
of correctness. With an LLM backend configured, the runner can optionally use
the model as a judge (see :class:`EvaluationRunner`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been this that these those it its with as by "
    "from at what which who how why when where does do did can could would should about into than then "
    "there their they them we you your our not no also such more most any all each other into".split()
)


def _dedupe(seq: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def recall_at_k(
    retrieved_sources: Sequence[str], relevant_sources: Sequence[str], k: int
) -> float | None:
    """Fraction of relevant sources present in the top-k retrieved sources (``None`` if no relevants)."""
    if not relevant_sources:
        return None
    top = set(_dedupe(retrieved_sources)[:k])
    return len(top & set(relevant_sources)) / len(set(relevant_sources))


def precision_at_k(
    retrieved_sources: Sequence[str], relevant_sources: Sequence[str], k: int
) -> float | None:
    """Fraction of the top-k *distinct* retrieved sources that are relevant."""
    if not relevant_sources:
        return None
    top = _dedupe(retrieved_sources)[:k]
    if not top:
        return 0.0
    return sum(1 for s in top if s in set(relevant_sources)) / len(top)


def mean_reciprocal_rank(
    retrieved_sources: Sequence[str], relevant_sources: Sequence[str]
) -> float | None:
    """1 / rank of the first relevant source (0 if none retrieved)."""
    if not relevant_sources:
        return None
    relevant = set(relevant_sources)
    for rank, source in enumerate(_dedupe(retrieved_sources), start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2 and w.lower() not in _STOPWORDS}


def answer_sentences(answer: str) -> list[str]:
    """Split an answer into sentences, dropping notices, bullets and citation markers."""
    cleaned = re.sub(r"\[\d+\]", "", answer)
    sentences: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line or line.startswith("⚠️") or line.lower().startswith("most relevant evidence"):
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    return sentences


def groundedness(
    answer: str, passages: Sequence[str], support_threshold: float = 0.6
) -> float | None:
    """Share of answer sentences whose content words are mostly covered by some passage.

    Returns ``None`` when the answer has no evaluable sentences.
    """
    sentences = [s for s in answer_sentences(answer) if content_words(s)]
    if not sentences:
        return None
    passage_words = [content_words(p) for p in passages]
    supported = 0
    for sentence in sentences:
        words = content_words(sentence)
        best = max((len(words & pw) / len(words) for pw in passage_words), default=0.0)
        if best >= support_threshold:
            supported += 1
    return supported / len(sentences)


def citation_validity(answer: str, num_citations: int) -> float | None:
    """Fraction of ``[n]`` markers in the answer that refer to an existing citation."""
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    if not markers:
        return None
    return sum(1 for m in markers if 1 <= m <= num_citations) / len(markers)


def phrase_coverage(answer: str, phrases: Sequence[str]) -> float | None:
    """Fraction of expected key phrases present in the answer (case-insensitive)."""
    if not phrases:
        return None
    lowered = answer.lower()
    return sum(1 for p in phrases if p.lower() in lowered) / len(phrases)


def _normalise_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by reference length."""
    ref, hyp = _normalise_words(reference), _normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if r == h else 1))
        prev = curr
    return prev[-1] / len(ref)


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]
