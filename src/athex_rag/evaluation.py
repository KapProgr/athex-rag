"""Evaluation harness.

Each line of the questions file is a JSON object:

    {"id": "opap24-01", "question": "...", "company": "OPAP", "year": 2024,
     "gold_pages": [45], "type": "numeric", "answer": "1.234,5 χιλ. €", "answer_value": 1234500}

``type`` is one of ``numeric`` | ``text`` | ``unanswerable``. Unanswerable questions
(the answer is not in the report) measure whether the system refuses instead of
inventing an answer - the failure mode that matters most for financial documents.

Metrics
* hit@k        - a retrieved chunk is on a gold page of the right document
* mrr          - reciprocal rank of the first such chunk
* correct      - numeric: value match at any unit scale; text: >= 80% of the gold
                 answer's tokens appear in the answer (LLM answers are wordy, so
                 recall is fairer than F1; F1 is still reported); unanswerable:
                 the model refused
* citation_prec- share of cited sources that are on a gold page
* refusal_rate - on answerable questions (lower is better)
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import Chunk
from .rag import RAG, is_refusal, parse_citations
from .retrieval import Hit, Retriever
from .text import numbers_match, token_f1, token_recall


@dataclass
class EvalItem:
    id: str
    question: str
    company: str
    year: int
    gold_pages: list[int]
    type: str = "text"
    answer: str = ""
    answer_value: float | None = None
    source: str = "text"  # where the answer lives: "text" or "table"
    evidence: str = ""  # verbatim quote from the gold page, for human verification
    alternatives: list[str] = field(default_factory=list)  # other accepted text answers

    @classmethod
    def from_dict(cls, data: dict) -> EvalItem:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class ItemResult:
    id: str
    type: str
    hit: bool
    rr: float
    correct: bool | None = None
    f1: float | None = None
    refused: bool | None = None
    citation_precision: float | None = None
    answer_in_context: bool | None = None  # a retrieved chunk contains the answer
    citation_support: float | None = None  # share of cited chunks that contain the answer
    answer: str = ""
    latency_s: float = 0.0
    retrieved: list[str] = field(default_factory=list)
    source: str = "text"
    category: str = ""


def load_items(path: Path) -> list[EvalItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                items.append(EvalItem.from_dict(json.loads(line)))
    return items


def _on_gold(hit: Hit, item: EvalItem) -> bool:
    c = hit.chunk
    return c.company == item.company.upper() and c.year == item.year and c.page in item.gold_pages


def retrieval_scores(hits: list[Hit], item: EvalItem) -> tuple[bool, float]:
    for rank, hit in enumerate(hits, start=1):
        if _on_gold(hit, item):
            return True, 1.0 / rank
    return False, 0.0


def judge_answer(text: str, item: EvalItem) -> tuple[bool, float | None]:
    if item.type == "unanswerable":
        return is_refusal(text), None
    if item.type == "numeric" and item.answer_value is not None:
        return numbers_match(text, float(item.answer_value)), None
    golds = [item.answer, *item.alternatives]
    f1 = max(token_f1(text, g) for g in golds)
    return any(token_recall(text, g) >= 0.8 for g in golds), f1


def _distinctive(value: float) -> bool:
    """A number specific enough that finding it in a chunk means the chunk states the
    answer (2,3839 or 1.232.904.830 yes; 50% or 0,40 no: those appear everywhere)."""
    digits = f"{abs(value):.6f}".replace(".", "").strip("0")
    return len(digits) >= 4


def supports(chunk: Chunk, item: EvalItem) -> bool:
    """Does this chunk state the answer? Gold-page chunks always count; other chunks
    count if they contain a distinctive answer value or the gold answer text. The same
    figure is often repeated on several pages, in different formats (1.232.904.830 on
    the statement page, 1.232,90 εκατ. in the management report)."""
    if chunk.company != item.company.upper() or chunk.year != item.year:
        return False
    if chunk.page in item.gold_pages:
        return True
    if item.type == "numeric" and item.answer_value is not None:
        value = float(item.answer_value)
        return _distinctive(value) and numbers_match(chunk.text, value, rel_tol=0.001)
    if item.type == "text":
        return any(token_recall(chunk.text, g) >= 0.8 for g in [item.answer, *item.alternatives])
    return False


def score(
    item: EvalItem,
    chunks: list[Chunk],
    answer_text: str | None,
    latency: float = 0.0,
) -> ItemResult:
    """Score one question from its retrieved chunks (in rank order) and the answer."""
    answerable = item.type != "unanswerable"
    hit, rr = (False, 0.0)
    for rank, chunk in enumerate(chunks, start=1):
        if (
            chunk.company == item.company.upper()
            and chunk.year == item.year
            and chunk.page in item.gold_pages
        ):
            hit, rr = True, 1.0 / rank
            break
    result = ItemResult(
        item.id,
        item.type,
        hit,
        rr,
        latency_s=latency,
        retrieved=[c.id for c in chunks],
        source=item.source,
    )
    if answerable:
        result.answer_in_context = any(supports(c, item) for c in chunks)
    if answer_text is not None:
        result.answer = answer_text
        result.correct, result.f1 = judge_answer(answer_text, item)
        result.refused = is_refusal(answer_text)
        cited = [chunks[n - 1] for n in parse_citations(answer_text) if 1 <= n <= len(chunks)]
        if cited and answerable:
            on_gold = [
                c.company == item.company.upper()
                and c.year == item.year
                and c.page in item.gold_pages
                for c in cited
            ]
            result.citation_precision = sum(on_gold) / len(cited)
            result.citation_support = sum(supports(c, item) for c in cited) / len(cited)
    result.category = categorize(result)
    return result


def evaluate_item(
    item: EvalItem, retriever: Retriever, rag: RAG | None, k: int, use_filters: bool
) -> ItemResult:
    company = item.company if use_filters else None
    year = item.year if use_filters else None
    start = time.perf_counter()
    if rag is None:
        hits = retriever.search(item.question, k=k, company=company, year=year)
        answer_text = None
    else:
        answer = rag.ask(item.question, company=company, year=year, k=k)
        hits, answer_text = answer.sources, answer.text
    return score(item, [h.chunk for h in hits], answer_text, time.perf_counter() - start)


def rescore_run(path: Path, items: list[EvalItem], chunks_by_id: dict) -> list[ItemResult]:
    """Re-score a saved run (``reports/runs/*.jsonl``) with the current metrics,
    without calling the retriever or the LLM again."""
    by_id = {it.id: it for it in items}
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            item = by_id.get(row["id"])
            if item is None:
                continue
            chunks = [chunks_by_id[c] for c in row["retrieved"] if c in chunks_by_id]
            text = row.get("answer") if row.get("correct") is not None else None
            results.append(score(item, chunks, text, row.get("latency_s", 0.0)))
    return results


ERROR_CATEGORIES = {
    "ok": "Correct, and cites a source that states the answer",
    "ok_bad_citation": "Correct value, but no cited source states it",
    "retrieval_miss": "No retrieved source states the answer",
    "false_refusal": "The answer was in the sources, but the model said it could not find it",
    "wrong_answer": "The answer was in the sources, but the model got it wrong",
    "hallucination": "Unanswerable question, but the model invented an answer",
    "refused_correctly": "Unanswerable question, correctly refused",
    "retrieved": "Retrieval-only run: gold page retrieved",
}


def categorize(r: ItemResult) -> str:
    found = r.hit or bool(r.answer_in_context)
    if r.correct is None:  # retrieval-only run
        if r.type == "unanswerable":
            return ""
        return "retrieved" if found else "retrieval_miss"
    if r.type == "unanswerable":
        return "refused_correctly" if r.correct else "hallucination"
    if r.correct:
        supported = (r.citation_support or 0) > 0 or (r.citation_precision or 0) > 0
        return "ok" if supported else "ok_bad_citation"
    if not found:
        return "retrieval_miss"
    return "false_refusal" if r.refused else "wrong_answer"


def error_report(results: list[ItemResult], items: list[EvalItem], examples: int = 3) -> str:
    """Markdown breakdown of outcomes with a few examples per failure category."""
    by_id = {it.id: it for it in items}
    counts: dict[str, list[ItemResult]] = {}
    for r in results:
        if r.category:
            counts.setdefault(r.category, []).append(r)
    total = sum(len(v) for v in counts.values()) or 1
    lines = ["| outcome | count | share | meaning |", "|---|---|---|---|"]
    for cat in ERROR_CATEGORIES:
        if cat in counts:
            n = len(counts[cat])
            lines.append(f"| {cat} | {n} | {n / total:.0%} | {ERROR_CATEGORIES[cat]} |")
    for cat in (
        "retrieval_miss",
        "wrong_answer",
        "false_refusal",
        "hallucination",
        "ok_bad_citation",
    ):
        if cat not in counts:
            continue
        lines += ["", f"**{cat}** examples:", ""]
        for r in counts[cat][:examples]:
            it = by_id[r.id]
            gold = it.answer or (f"{it.answer_value:g}" if it.answer_value is not None else "–")
            got = " ".join(r.answer.split())[:220] if r.answer else "(retrieval only)"
            lines.append(
                f"- `{r.id}` {it.question}  \n  gold: {gold} (p. {it.gold_pages}) · "
                f"got: {got} · retrieved: {', '.join(r.retrieved[:3])}"
            )
    return "\n".join(lines)


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summarize(results: list[ItemResult]) -> dict:
    answerable = [r for r in results if r.type != "unanswerable"]
    unanswerable = [r for r in results if r.type == "unanswerable"]
    summary = {
        "n": len(results),
        "hit@k": _mean(r.hit for r in answerable),
        "answer@k": _mean(r.answer_in_context for r in answerable),
        "mrr": _mean(r.rr for r in answerable),
        "correct": _mean(r.correct for r in results),
        "correct_numeric": _mean(r.correct for r in results if r.type == "numeric"),
        "correct_text": _mean(r.correct for r in results if r.type == "text"),
        "hit@k_table_q": _mean(r.hit for r in answerable if r.source == "table"),
        "correct_table_q": _mean(r.correct for r in answerable if r.source == "table"),
        "refused_unanswerable": _mean(r.refused for r in unanswerable),
        "refusal_rate_answerable": _mean(r.refused for r in answerable),
        "citation_precision": _mean(r.citation_precision for r in answerable),
        "citation_support": _mean(r.citation_support for r in answerable),
        "latency_s": _mean(r.latency_s for r in results),
    }
    return summary


def to_markdown(rows: dict[str, dict]) -> str:
    """Render ``{config_name: summary}`` as a markdown comparison table."""
    columns = [
        "hit@k",
        "answer@k",
        "mrr",
        "correct",
        "correct_numeric",
        "correct_text",
        "hit@k_table_q",
        "correct_table_q",
        "refused_unanswerable",
        "refusal_rate_answerable",
        "citation_precision",
        "citation_support",
        "latency_s",
    ]
    present = [c for c in columns if any(s.get(c) is not None for s in rows.values())]
    lines = ["| config | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
    for name, summary in rows.items():
        cells = []
        for col in present:
            value = summary.get(col)
            if value is None:
                cells.append("–")
            elif col == "latency_s":
                cells.append(f"{value:.2f}")
            else:
                cells.append(f"{value:.1%}" if col != "mrr" else f"{value:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_run(folder: Path, name: str, results: list[ItemResult], summary: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    (folder / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _squash(text: str) -> str:
    from .text import normalize

    return " ".join(normalize(text).split())


def check_items(
    items: list[EvalItem], pages_by_doc: dict[str, list[str]], fix: bool = False
) -> list[str]:
    """Verify each item's evidence quote against the report text.

    The gold pages of an answerable question are *every* page whose text contains the
    evidence quote (the same figure often appears in the management report and in the
    financial statements). With ``fix=True`` gold_pages are rewritten accordingly.
    For unanswerable questions, the evidence field lists forbidden key phrases separated
    by "|" that must appear on no page. Returns a list of problems.
    """
    problems = []
    ids = set()
    for it in items:
        if it.id in ids:
            problems.append(f"{it.id}: duplicate id")
        ids.add(it.id)
        doc = f"{it.company.upper()}_{it.year}"
        pages = pages_by_doc.get(doc)
        if pages is None:
            problems.append(f"{it.id}: no document {doc}")
            continue
        squashed = [_squash(t) for t in pages]
        if it.type == "unanswerable":
            for phrase in filter(None, (x.strip() for x in it.evidence.split("|"))):
                hits = [i + 1 for i, t in enumerate(squashed) if _squash(phrase) in t]
                if hits:
                    problems.append(f"{it.id}: unanswerable but '{phrase}' found on {hits[:5]}")
            if fix:
                it.gold_pages = []
            continue
        if not it.evidence:
            problems.append(f"{it.id}: no evidence quote")
            continue
        found = [i + 1 for i, t in enumerate(squashed) if _squash(it.evidence) in t]
        if not found:
            problems.append(f"{it.id}: evidence not found in {doc}: {it.evidence!r}")
        elif fix:
            it.gold_pages = found
        elif set(it.gold_pages) - set(found):
            problems.append(f"{it.id}: gold pages {it.gold_pages} but evidence on {found}")
        if it.type == "numeric" and it.answer_value is not None:
            if not numbers_match(it.evidence, float(it.answer_value)):
                problems.append(f"{it.id}: answer_value {it.answer_value} not in evidence")
    return problems


def save_items(path: Path, items: list[EvalItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            row = {k: v for k, v in asdict(it).items() if v not in (None, "", [])}
            row.setdefault("gold_pages", [])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
