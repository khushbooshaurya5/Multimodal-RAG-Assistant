"""Offline lexical embeddings via feature hashing.

This is the **fallback** used when no neural embedding model can be loaded
(air-gapped machines, CI). It is a real, deterministic sparse-to-dense lexical
representation - word unigrams + bigrams and character n-grams hashed into a
fixed-size vector with sub-linear term-frequency weighting - equivalent in
spirit to scikit-learn's ``HashingVectorizer``.

What it can and cannot do:

* It matches on shared vocabulary and morphology (character n-grams give some
  robustness to inflection and typos).
* It has **no semantic understanding**: "car" and "automobile" are unrelated.
* Cosine scores are lower and less separable than dense embeddings, so the
  similarity threshold must be tuned separately (see ``configs/offline.env``).

It is flagged ``is_fallback=True`` and the UI reports it.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from src.embeddings.base import l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def _stable_hash(feature: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little"
    )


class HashingEmbedder:
    """Deterministic hashed n-gram embedder (no model weights required)."""

    is_fallback = True

    def __init__(self, dimension: int = 384, char_ngrams: tuple[int, ...] = (3, 4, 5)) -> None:
        if dimension < 16:
            raise ValueError("dimension must be >= 16")
        self.dimension = dimension
        self.char_ngrams = char_ngrams
        self.name = f"hashing:{dimension}"

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.lower() for t in _TOKEN.findall(text)]

    def _features(self, text: str) -> dict[str, float]:
        tokens = self.tokenize(text)
        feats: dict[str, float] = {}
        for tok in tokens:
            feats[f"w:{tok}"] = feats.get(f"w:{tok}", 0.0) + 1.0
            padded = f"#{tok}#"
            for n in self.char_ngrams:
                for i in range(max(0, len(padded) - n + 1)):
                    key = f"c{n}:{padded[i : i + n]}"
                    feats[key] = feats.get(key, 0.0) + 0.5
        for a, b in zip(tokens, tokens[1:], strict=False):
            key = f"b:{a}_{b}"
            feats[key] = feats.get(key, 0.0) + 1.5
        return feats

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for feature, count in self._features(text).items():
            h = _stable_hash(feature)
            index = h % self.dimension
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return l2_normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
