"""Split page text into retrieval chunks.

Chunks never cross a page boundary: every answer can then cite an exact page,
which is what the evaluation checks.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_SENTENCE_RE = re.compile(r"(?<=[.;!?·])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    company: str
    year: int
    page: int  # 1-based, as printed in a PDF viewer
    text: str
    kind: str = "text"  # "text" or "table"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_RE.split(paragraph):
        while len(sentence) > max_chars:  # a "sentence" with no punctuation at all
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_page(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Pack paragraphs into chunks of at most ``max_chars``, with a tail overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units = [piece for p in paragraphs for piece in _split_long(" ".join(p.split()), max_chars)]
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            # start the overlap on a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail}\n\n{unit}" if tail else unit
            if len(current) > max_chars:
                current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks
