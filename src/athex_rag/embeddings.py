"""Embedding backends.

``SentenceTransformerEmbedder`` runs a multilingual model locally (no API cost).
``HashingEmbedder`` is a dependency-free stand-in used by tests and CI.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    """Local multilingual embeddings.

    Defaults to ``intfloat/multilingual-e5-small`` (fast on CPU, good Greek).
    Try ``intfloat/multilingual-e5-base`` or ``BAAI/bge-m3`` in the evaluation.
    E5 models expect "query: " / "passage: " prefixes.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefix = "passage: " if self._e5 else ""
        return self._encode([prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        prefix = "query: " if self._e5 else ""
        return self._encode([prefix + text])[0]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed vector. Deterministic, no downloads."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vector(text))


def make_embedder(name: str) -> Embedder:
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 512
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(name)
