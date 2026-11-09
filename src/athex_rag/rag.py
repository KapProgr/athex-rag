"""Retrieval-augmented answering with numbered, page-level citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .retrieval import Hit, Retriever
from .text import normalize

SYSTEM_PROMPT = """Είσαι αναλυτής που απαντά ερωτήσεις για ετήσιες οικονομικές εκθέσεις \
εισηγμένων εταιρειών του Χρηματιστηρίου Αθηνών.

Κανόνες:
1. Απάντησε ΜΟΝΟ με βάση τις πηγές που σου δίνονται. Μη χρησιμοποιείς δική σου γνώση.
2. Μετά από κάθε ισχυρισμό βάλε την πηγή σε αγκύλες, π.χ. [1] ή [2][3].
3. Για αριθμούς, γράψε τον αριθμό ακριβώς όπως στην πηγή μαζί με τη μονάδα \
(π.χ. "σε χιλ. €") και τη χρήση στην οποία αναφέρεται.
4. Αν οι πηγές δεν περιέχουν την απάντηση, γράψε ακριβώς: \
"Δεν βρέθηκε η πληροφορία στα έγγραφα."
5. Απάντησε στη γλώσσα της ερώτησης, σύντομα."""

NOT_FOUND = "Δεν βρέθηκε η πληροφορία στα έγγραφα."
_REFUSAL_MARKERS = [
    normalize(NOT_FOUND),
    "δεν βρεθηκε",
    "not found in the documents",
    "δεν περιεχουν",
    "δεν αναφερεται",
]


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)  # 1-based source numbers the model cited

    @property
    def refused(self) -> bool:
        return is_refusal(self.text)

    @property
    def cited_hits(self) -> list[Hit]:
        return [self.sources[n - 1] for n in self.cited if 1 <= n <= len(self.sources)]


def is_refusal(text: str) -> bool:
    norm = normalize(text)
    return any(marker in norm for marker in _REFUSAL_MARKERS)


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    # some models (gpt-oss) write full-width brackets, with line refs: 【1】 【1†L1-L3】
    for n in re.findall(r"[\[【](\d{1,2})(?:†[^\]】]*)?[\]】]", text):
        if int(n) not in seen:
            seen.append(int(n))
    return seen


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        label = "πίνακας" if c.kind == "table" else "κείμενο"
        blocks.append(f"[{n}] {c.company} {c.year}, σελ. {c.page} ({label})\n{c.text}")
    return "Πηγές:\n\n" + "\n\n".join(blocks) + f"\n\nΕρώτηση: {question}\nΑπάντηση:"


class RAG:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 5):
        self.retriever, self.llm, self.k = retriever, llm, k

    def ask(
        self,
        question: str,
        company: str | None = None,
        year: int | None = None,
        k: int | None = None,
    ) -> Answer:
        hits = self.retriever.search(question, k=k or self.k, company=company, year=year)
        if not hits:
            return Answer(question, NOT_FOUND, [])
        text = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, hits))
        return Answer(question, text, hits, parse_citations(text))
