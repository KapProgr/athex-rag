"""Optional second-stage reranking.

First-stage retrieval (BM25 / embeddings) scores query and passage independently.
A cross-encoder reads them *together*, which is slower but much better at telling
"net profit 2024" apart from "net profit 2023" in the same table. It runs on the
top-N candidates only.
"""

from __future__ import annotations

from typing import Protocol

from .retrieval import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: -float(pair[1]))
        return [Hit(h.chunk, float(s)) for h, s in ranked[:k]]


class KeywordOverlapReranker:
    """Cheap, dependency-free reranker used in tests: share of query tokens in chunk."""

    name = "keyword-overlap"

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        from .text import tokenize

        q = set(tokenize(query))
        scored = [(h, len(q & set(tokenize(h.chunk.text))) / max(len(q), 1)) for h in hits]
        scored.sort(key=lambda pair: -pair[1])
        return [Hit(h.chunk, s) for h, s in scored[:k]]


def make_reranker(name: str | None) -> Reranker | None:
    if not name or name == "none":
        return None
    if name == "keyword-overlap":
        return KeywordOverlapReranker()
    return CrossEncoderReranker(name)
