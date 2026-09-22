"""Evidence indicator derived from retrieval statistics.

This is a *heuristic* signal for the UI ("how much retrieved support is
behind this answer?"). It is computed from similarity scores and hit counts,
not from the language model, and it is **not** a calibrated probability of
correctness. The thresholds below were chosen by inspection, not fitted.
"""

from __future__ import annotations

from statistics import mean

from src.schemas import EvidenceIndicator, EvidenceLevel, RetrievedChunk


def compute_evidence(
    retrieved: list[RetrievedChunk],
    *,
    similarity_threshold: float,
    lexical_backend: bool = False,
) -> EvidenceIndicator:
    """Summarise retrieval support as an :class:`EvidenceIndicator`.

    Args:
        retrieved: Chunks that passed the threshold, best first.
        similarity_threshold: The threshold in force (shown in the note).
        lexical_backend: ``True`` when the hashing embedder is used; its cosine
            scores are systematically lower, so bands are shifted down.
    """
    if not retrieved:
        return EvidenceIndicator(
            level=EvidenceLevel.NONE,
            num_sources=0,
            best_score=None,
            mean_score=None,
            note=(
                f"No indexed passage scored above the similarity threshold ({similarity_threshold:.2f}). "
                "The answer is not grounded in your documents."
            ),
        )

    scores = [r.score for r in retrieved]
    best, avg = max(scores), mean(scores)
    distinct_sources = len({r.metadata.source for r in retrieved})

    strong, moderate = (0.35, 0.18) if lexical_backend else (0.60, 0.40)
    if best >= strong and len(retrieved) >= 2:
        level = EvidenceLevel.STRONG
    elif best >= moderate:
        level = EvidenceLevel.MODERATE
    else:
        level = EvidenceLevel.WEAK

    note = (
        f"{len(retrieved)} passage(s) from {distinct_sources} source(s); best cosine similarity "
        f"{best:.2f}, mean {avg:.2f}. Heuristic indicator based on retrieval scores only - "
        "not a calibrated probability that the answer is correct."
    )
    return EvidenceIndicator(
        level=level, num_sources=len(retrieved), best_score=best, mean_score=avg, note=note
    )
