from __future__ import annotations

import pytest
from src.evaluation import (
    citation_validity,
    groundedness,
    mean_reciprocal_rank,
    phrase_coverage,
    precision_at_k,
    recall_at_k,
    word_error_rate,
)
from src.evaluation.metrics import answer_sentences, percentile


def test_recall_precision_mrr():
    retrieved = ["a.md", "a.md", "b.md", "c.md"]  # duplicates collapse to distinct sources
    assert recall_at_k(retrieved, ["a.md", "z.md"], k=3) == 0.5
    assert precision_at_k(retrieved, ["a.md", "z.md"], k=3) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(retrieved, ["b.md"]) == 0.5
    assert mean_reciprocal_rank(retrieved, ["z.md"]) == 0.0
    assert recall_at_k(retrieved, [], k=3) is None
    assert precision_at_k([], ["a.md"], k=3) == 0.0


def test_groundedness_lexical_proxy():
    passages = [
        "Pooling layers reduce the spatial size of feature maps and add translation invariance."
    ]
    supported = "Pooling layers reduce the spatial size of feature maps. [1]"
    unsupported = "Pooling layers were invented by Napoleon in 1805."
    assert groundedness(supported, passages) == 1.0
    assert groundedness(unsupported, passages) == 0.0
    assert groundedness(supported + " " + unsupported, passages) == 0.5
    assert groundedness("⚠️ notice only", passages) is None


def test_answer_sentences_strips_notices_and_markers():
    text = "⚠️ No language model.\n\nMost relevant evidence sentences:\n- First fact. [1]\n- Second fact! [2]"
    assert answer_sentences(text) == ["First fact.", "Second fact!"]


def test_citation_validity_and_phrase_coverage():
    assert citation_validity("A [1] and B [3].", num_citations=2) == 0.5
    assert citation_validity("no markers", num_citations=2) is None
    assert phrase_coverage("Uses max pooling and ReLU.", ["max pooling", "softmax"]) == 0.5
    assert phrase_coverage("x", []) is None


def test_word_error_rate():
    assert word_error_rate("the cat sat", "the cat sat") == 0.0
    assert word_error_rate("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert word_error_rate("the cat sat", "a dog sat down") == pytest.approx(3 / 3)
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "x") == 1.0


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5
