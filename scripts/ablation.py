"""Retrieval improvements, one at a time, with a dev/test split.

    python scripts/ablation.py                 # all steps, writes reports/ablation.md
    python scripts/ablation.py --show-misses   # dev misses of the last step (for analysis)

Split: questions about the 2024 reports are the *dev* set (look at failures, tune);
questions about the 2025 reports are the *test* set (only measured). The two years ask
parallel questions about the same companies, so a fix found on 2024 should carry over
to 2025 if it is real and not fitted to specific questions.

Keyword retrieval only (BM25), so this runs anywhere in seconds; no models needed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from athex_rag.companies import load_aliases, load_glossary  # noqa: E402
from athex_rag.embeddings import NullEmbedder  # noqa: E402
from athex_rag.evaluation import load_items, score  # noqa: E402
from athex_rag.index import Index  # noqa: E402
from athex_rag.ingest import chunks_for_document, iter_pdfs  # noqa: E402
from athex_rag.retrieval import Retriever  # noqa: E402


@dataclass(frozen=True)
class Step:
    name: str
    chunk: int = 600
    layout: str = "sorted"
    headings: bool = False
    glossary: bool = False
    k1: float = 1.5
    b: float = 0.75


_chunk_cache: dict[tuple, list] = {}


def chunks_for(step: Step) -> list:
    key = (step.chunk, step.layout, step.headings)
    if key not in _chunk_cache:
        chunks = []
        for pdf in iter_pdfs(ROOT / "data" / "raw"):
            chunks += chunks_for_document(
                pdf,
                step.chunk,
                step.chunk // 6,
                True,
                ROOT / "data" / "cache",
                layout=step.layout,
                headings=step.headings,
            )
        _chunk_cache[key] = chunks
    return _chunk_cache[key]


def evaluate(step: Step, items, k: int = 5):
    chunks = chunks_for(step)
    index = Index(
        chunks,
        NullEmbedder().embed_documents([c.text for c in chunks]),
        "none",
        k1=step.k1,
        b=step.b,
    )
    retriever = Retriever(
        index,
        NullEmbedder(),
        mode="bm25",
        aliases=load_aliases(),
        glossary=load_glossary() if step.glossary else None,
    )
    results = {}
    for it in items:
        if it.type == "unanswerable":
            continue
        hits = retriever.search(it.question, k=k)
        results[it.id] = score(it, [h.chunk for h in hits], None)
    return results


def summary(results: dict, ids: list[str]) -> dict:
    rs = [results[i] for i in ids]
    return {
        "n": len(rs),
        "hit@5": statistics.fmean(r.hit for r in rs),
        "answer@5": statistics.fmean(bool(r.answer_in_context) for r in rs),
        "mrr": statistics.fmean(r.rr for r in rs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    items = load_items(ROOT / "eval" / "questions.jsonl")
    dev = [i.id for i in items if i.year == 2024 and i.type != "unanswerable"]
    test = [i.id for i in items if i.year == 2025 and i.type != "unanswerable"]

    base = Step("baseline: BM25 + stemming, chunk 600")
    steps = [
        base,
        replace(base, name="+ column-aware text", layout="columns"),
        # measured, but not kept: lower MRR on dev
        replace(base, name="(+ page headings: not kept)", layout="columns", headings=True),
        replace(base, name="+ glossary expansion", layout="columns", glossary=True),
    ]
    # BM25 parameters: chosen on dev only
    grid = [(k1, b) for k1 in (0.9, 1.2, 1.5, 2.0) for b in (0.4, 0.6, 0.75, 0.9)]
    tuned = None
    rows = []
    for step in steps:
        res = evaluate(step, items)
        rows.append((step, summary(res, dev), summary(res, test), res))
        print(
            f"{step.name:40} dev {rows[-1][1]['answer@5']:.3f}  test {rows[-1][2]['answer@5']:.3f}",
            flush=True,
        )
    if not args.show_misses:
        best_dev, best = -1.0, None
        last = steps[-1]
        for k1, b in grid:
            s = replace(last, name=f"+ BM25 k1={k1}, b={b}", k1=k1, b=b)
            d = summary(evaluate(s, items), dev)
            key = d["answer@5"] + d["mrr"] / 10  # MRR breaks ties
            if key > best_dev:
                best_dev, best = key, s
        tuned = best
        res = evaluate(tuned, items)
        rows.append((tuned, summary(res, dev), summary(res, test), res))
        print(
            f"{tuned.name:40} dev {rows[-1][1]['answer@5']:.3f}  test {rows[-1][2]['answer@5']:.3f}"
        )

    if args.show_misses:
        step, _, _, res = rows[-1]
        by_id = {i.id: i for i in items}
        print(f"\nDev misses ({step.name}):")
        for i in dev:
            r = res[i]
            if not r.answer_in_context:
                print(
                    f"  {i}: {by_id[i].question}  gold p{by_id[i].gold_pages[:4]}  "
                    f"got {', '.join(r.retrieved[:3])}"
                )
        return

    def fmt(s: dict) -> str:
        return f"{s['answer@5']:.1%} | {s['hit@5']:.1%} | {s['mrr']:.3f}"

    lines = [
        "# Retrieval improvements (dev/test split)",
        "",
        f"Dev = questions on the 2024 reports ({len(dev)} answerable), used to find failures and "
        f"tune. Test = questions on the 2025 reports ({len(test)} answerable), only measured. "
        "BM25 with Greek stemming, k = 5. *answer@5*: a retrieved chunk states the answer; "
        "*hit@5*: a retrieved chunk is on a gold page.",
        "",
        "| step | dev answer@5 | dev hit@5 | dev MRR | test answer@5 | test hit@5 | test MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for step, d, t, _ in rows:
        lines.append(f"| {step.name} | {fmt(d)} | {fmt(t)} |")
    lines += [
        "",
        f"Each dev question is worth {100 / len(dev):.1f} points, each test question "
        f"{100 / len(test):.1f}; differences of one or two questions are noise.",
    ]
    out = ROOT / "reports" / "ablation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
