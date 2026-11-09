"""On-disk index: chunks + dense vectors + a BM25 keyword index.

Kept deliberately simple (NumPy, no vector DB): a few thousand pages of annual
reports fit comfortably in memory, and every scoring step stays inspectable.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .text import search_tokens


class BM25:
    """Okapi BM25 over Greek-normalized tokens."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(d) for d in docs]
        self.doc_lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.doc_freqs), dtype=np.float32)
        if not self.doc_freqs:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens / max(self.avgdl, 1e-9))
        for tok in set(query_tokens):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            tf = np.array([d.get(tok, 0) for d in self.doc_freqs], dtype=np.float32)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class Index:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder_name: str,
        stem_prefix: int | None = 6,
        k1: float = 0.9,  # tuned on the dev questions, see reports/ablation.md
        b: float = 0.4,
    ):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedder_name = embedder_name
        self.stem_prefix = stem_prefix
        self.bm25 = BM25([search_tokens(c.text, stem_prefix) for c in chunks], k1=k1, b=b)

    def query_tokens(self, query: str) -> list[str]:
        return search_tokens(query, self.stem_prefix)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> Index:
        vectors = embedder.embed_documents([c.text for c in chunks])
        return cls(chunks, vectors, embedder.name)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder_name, "n_chunks": len(self.chunks)}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, folder: Path, stem_prefix: int | None = 6, k1: float = 0.9, b: float = 0.4
    ) -> Index:
        plain, packed = folder / "chunks.jsonl", folder / "chunks.jsonl.gz"
        opener = (
            (lambda: open(plain, encoding="utf-8"))
            if plain.exists()
            else (lambda: gzip.open(packed, "rt", encoding="utf-8"))
        )
        with opener() as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = np.load(folder / "vectors.npy")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, vectors, meta["embedder"], stem_prefix, k1, b)

    @property
    def companies(self) -> list[str]:
        return sorted({c.company for c in self.chunks})

    @property
    def years(self) -> list[int]:
        return sorted({c.year for c in self.chunks})
