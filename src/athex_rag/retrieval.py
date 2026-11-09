"""Dense, keyword (BM25) and hybrid retrieval with metadata filters.

Hybrid uses Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across retrievers.
RRF needs no score calibration between cosine similarity and BM25, which is why
it is the default here; the evaluation harness compares all three modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .index import Index
from .text import normalize

MODES = ("dense", "bm25", "hybrid")


@dataclass
class Hit:
    chunk: Chunk
    score: float


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _words(text: str) -> str:
    """Normalized words padded with spaces, for whole-phrase matching."""
    return " " + " ".join(re.findall(r"\w+", normalize(text))) + " "


def infer_filters(
    query: str,
    companies: list[str],
    years: list[int],
    aliases: dict[str, list[str]] | None = None,
) -> tuple[str | None, int | None]:
    """Pick up a company or a year mentioned in the question, if unambiguous.

    ``aliases`` maps a company code to the names people actually type
    ({"PPC": ["ΔΕΗ", "Δημόσια Επιχείρηση Ηλεκτρισμού"]}).
    """
    norm = _words(query)
    found_companies = []
    for code in companies:
        names = [code, *(aliases or {}).get(code, [])]
        if any(_words(name) in norm for name in names):
            found_companies.append(code)
    found_years = [y for y in years if f" {y} " in norm]
    company = found_companies[0] if len(found_companies) == 1 else None
    year = found_years[0] if len(found_years) == 1 else None
    return company, year


class Retriever:
    def __init__(
        self,
        index: Index,
        embedder,
        mode: str = "hybrid",
        candidates: int = 50,
        reranker=None,
        rerank_candidates: int = 20,
        aliases: dict[str, list[str]] | None = None,
        glossary: dict[str, list[str]] | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if embedder.name != index.embedder_name:
            raise ValueError(
                f"index was built with {index.embedder_name!r}, query embedder is {embedder.name!r}"
            )
        self.index, self.embedder, self.mode, self.candidates = index, embedder, mode, candidates
        self.reranker, self.rerank_candidates = reranker, rerank_candidates
        self.aliases = aliases or {}
        # phrase -> extra search terms (see data/glossary.json)
        self.glossary = {_words(k): v for k, v in (glossary or {}).items()}

    def keyword_tokens(self, query: str) -> list[str]:
        """Query tokens for BM25, expanded with glossary terms for phrases in the query
        (e.g. "ελεγκτική εταιρεία" also searches "ορκωτός ελεγκτής λογιστής", "ΣΟΕΛ")."""
        tokens = self.index.query_tokens(query)
        norm = _words(query)
        for phrase, extra in self.glossary.items():
            if phrase in norm:
                for term in extra:
                    tokens += self.index.query_tokens(term)
        return tokens

    def _mask(self, company: str | None, year: int | None) -> np.ndarray:
        mask = np.ones(len(self.index.chunks), dtype=bool)
        if company:
            mask &= np.array([c.company == company.upper() for c in self.index.chunks])
        if year:
            mask &= np.array([c.year == int(year) for c in self.index.chunks])
        return mask

    def _ranking(self, scores: np.ndarray, mask: np.ndarray) -> list[int]:
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[: self.candidates]
        return [int(i) for i in order if np.isfinite(scores[i])]

    def search(
        self,
        query: str,
        k: int = 5,
        company: str | None = None,
        year: int | None = None,
        auto_filter: bool = True,
    ) -> list[Hit]:
        if not self.index.chunks:
            return []
        if auto_filter and (company is None or year is None):
            c, y = infer_filters(query, self.index.companies, self.index.years, self.aliases)
            company, year = company or c, year or y
        mask = self._mask(company, year)

        dense = bm25 = None
        if self.mode in ("dense", "hybrid"):
            dense_scores = self.index.vectors @ self.embedder.embed_query(query)
            dense = self._ranking(dense_scores, mask)
        if self.mode in ("bm25", "hybrid"):
            bm25_scores = self.index.bm25.scores(self.keyword_tokens(query))
            # chunks with zero keyword overlap are not candidates
            bm25 = [i for i in self._ranking(bm25_scores, mask) if bm25_scores[i] > 0]

        if self.mode == "dense":
            ranked = [(i, float(dense_scores[i])) for i in dense]
        elif self.mode == "bm25":
            ranked = [(i, float(bm25_scores[i])) for i in bm25]
        else:
            fused = rrf([dense, bm25])
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        if self.reranker is not None:
            pool = [Hit(self.index.chunks[i], s) for i, s in ranked[: self.rerank_candidates]]
            return self.reranker.rerank(query, pool, k)
        return [Hit(self.index.chunks[i], s) for i, s in ranked[:k]]
