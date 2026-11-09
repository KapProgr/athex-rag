"""Command line: ingest PDFs, ask questions, run evaluations.

athex-rag ingest data/raw --out index
athex-rag ask "Ποια ήταν τα καθαρά κέρδη του ΟΠΑΠ το 2024;"
athex-rag eval eval/questions.jsonl --modes bm25 dense hybrid --no-llm
athex-rag eval eval/questions.jsonl --modes hybrid --llm ollama:qwen2.5:7b
athex-rag experiments eval/questions.jsonl          # the whole experiment plan
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .companies import load_aliases
from .embeddings import make_embedder
from .evaluation import evaluate_item, load_items, save_run, summarize, to_markdown
from .index import Index
from .ingest import chunks_for_document, iter_pdfs
from .llm import make_llm
from .rag import RAG
from .rerank import make_reranker
from .retrieval import MODES, Retriever

DEFAULT_EMBEDDER = "intfloat/multilingual-e5-small"


def cmd_ingest(args: argparse.Namespace) -> int:
    pdfs = list(iter_pdfs(Path(args.folder)))
    if not pdfs:
        print(f"No PDFs in {args.folder}. Expected files like OPAP_2024.pdf", file=sys.stderr)
        return 1
    chunks = []
    for pdf in pdfs:
        doc_chunks = chunks_for_document(
            pdf,
            args.max_chars,
            args.overlap,
            with_tables=not args.no_tables,
            cache_dir=Path(args.cache),
        )
        n_tables = sum(c.kind == "table" for c in doc_chunks)
        print(f"{pdf.name}: {len(doc_chunks)} chunks ({n_tables} tables)")
        chunks.extend(doc_chunks)
    embedder = make_embedder(args.embedder)
    print(f"Embedding {len(chunks)} chunks with {embedder.name} ...")
    index = Index.build(chunks, embedder)
    index.save(Path(args.out))
    print(f"Saved index to {args.out}")
    return 0


def _load(index_dir: str, mode: str, reranker: str | None = None) -> Retriever:
    index = Index.load(Path(index_dir))
    return Retriever(
        index,
        make_embedder(index.embedder_name),
        mode=mode,
        reranker=make_reranker(reranker),
        aliases=load_aliases(),
    )


def cmd_ask(args: argparse.Namespace) -> int:
    rag = RAG(_load(args.index, args.mode, args.rerank), make_llm(args.llm), k=args.k)
    answer = rag.ask(args.question, company=args.company, year=args.year)
    print(answer.text, "\n")
    for n, hit in enumerate(answer.sources, start=1):
        mark = "*" if n in answer.cited else " "
        c = hit.chunk
        print(f"{mark}[{n}] {c.company} {c.year} p.{c.page} ({c.kind}) score={hit.score:.3f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    items = load_items(Path(args.questions))
    llm = None if args.no_llm else make_llm(args.llm)
    rows = {}
    for mode in args.modes:
        retriever = _load(args.index, mode, args.rerank)
        rag = RAG(retriever, llm, k=args.k) if llm else None
        results = [evaluate_item(it, retriever, rag, args.k, args.filters) for it in items]
        name = f"{mode}" + (f"+{llm.name}" if llm else "") + ("+filters" if args.filters else "")
        summary = summarize(results)
        save_run(Path(args.out), name.replace(":", "_").replace("/", "_"), results, summary)
        rows[name] = summary
        print(f"done: {name}")
    table = to_markdown(rows)
    print("\n" + table)
    (Path(args.out) / "results.md").write_text(table + "\n", encoding="utf-8")
    return 0


def cmd_check_eval(args: argparse.Namespace) -> int:
    from .evaluation import check_items, save_items
    from .ingest import load_pages

    items = load_items(Path(args.questions))
    pages_by_doc = {}
    for pdf in iter_pdfs(Path(args.pdfs)):
        pages_by_doc[pdf.stem.upper()] = [t for _, t, _ in load_pages(pdf, Path(args.cache))]
    problems = check_items(items, pages_by_doc, fix=args.fix)
    for problem in problems:
        print("PROBLEM", problem)
    kinds = {}
    for it in items:
        kinds[(it.type, it.source)] = kinds.get((it.type, it.source), 0) + 1
    print(
        f"{len(items)} questions over {len({(i.company, i.year) for i in items})} reports:",
        ", ".join(f"{t}/{s}={n}" for (t, s), n in sorted(kinds.items())),
    )
    if args.fix:
        save_items(Path(args.questions), items)
        print("gold_pages updated from evidence")
    return 1 if problems else 0


def cmd_rescore(args: argparse.Namespace) -> int:
    import json

    from .chunking import Chunk
    from .evaluation import error_report, rescore_run, save_run

    items = load_items(Path(args.questions))
    chunks_by_id = {}
    for folder in args.indexes:
        with open(Path(folder) / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                chunk = Chunk.from_dict(json.loads(line))
                chunks_by_id[chunk.id] = chunk
    rows, sections = {}, []
    for run in args.runs:
        path = Path(run)
        results = rescore_run(path, items, chunks_by_id)
        name = path.stem
        rows[name] = summarize(results)
        save_run(Path(args.out), name, results, rows[name])
        if any(r.correct is not None for r in results):
            sections.append(f"## Error analysis: {name}\n\n" + error_report(results, items, 5))
    text = "# Rescored runs\n\n" + to_markdown(rows) + "\n\n" + "\n\n".join(sections) + "\n"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "rescored.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    from .experiments import Plan, Runner, update_readme

    items = load_items(Path(args.questions))
    if not list(iter_pdfs(Path(args.pdfs))):
        print(f"No PDFs in {args.pdfs}", file=sys.stderr)
        return 1
    plan = Plan(
        pdf_dir=Path(args.pdfs),
        items=items,
        embedders=args.embedders,
        llms=args.llms,
        chunk_sizes=args.chunk_sizes,
        reranker=None if args.reranker == "none" else args.reranker,
        k=args.k,
        out=Path(args.out),
    )
    report = Runner(plan).execute(skip_llm=args.skip_llm)
    print("\n" + report)
    if args.update_readme and update_readme(Path("README.md"), report):
        print("README.md results section updated.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="athex-rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="parse PDFs and build the index")
    p.add_argument("folder")
    p.add_argument("--out", default="index")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--no-tables", action="store_true")
    p.add_argument("--cache", default="data/cache", help="parsed-page cache folder")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--index", default="index")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--llm", default=None)
    p.add_argument("--company")
    p.add_argument("--year", type=int)
    p.add_argument("-k", type=int, default=5)
    p.add_argument(
        "--rerank", default=None, help="cross-encoder model, e.g. BAAI/bge-reranker-v2-m3"
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation set")
    p.add_argument("questions")
    p.add_argument("--index", default="index")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--llm", default=None)
    p.add_argument("--no-llm", action="store_true", help="retrieval metrics only (fast)")
    p.add_argument(
        "--filters",
        action="store_true",
        help="pass gold company/year as filters (upper bound for retrieval)",
    )
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out", default="reports")
    p.add_argument("--rerank", default=None)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("check-eval", help="verify evidence quotes and gold pages")
    p.add_argument("questions")
    p.add_argument("--pdfs", default="data/raw")
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--fix", action="store_true", help="set gold_pages from evidence")
    p.set_defaults(func=cmd_check_eval)

    p = sub.add_parser("rescore", help="re-score saved runs with the current metrics")
    p.add_argument("questions")
    p.add_argument("runs", nargs="+", help="reports/runs/*.jsonl files")
    p.add_argument("--indexes", nargs="+", required=True, help="index folders with chunks.jsonl")
    p.add_argument("--out", default="reports/rescored")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("experiments", help="run the full experiment plan and write a report")
    p.add_argument("questions")
    p.add_argument("--pdfs", default="data/raw")
    p.add_argument(
        "--embedders",
        nargs="+",
        default=[DEFAULT_EMBEDDER, "intfloat/multilingual-e5-base", "BAAI/bge-m3"],
    )
    p.add_argument("--llms", nargs="+", default=["ollama:qwen2.5:7b", "ollama:llama3.1:8b"])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[600, 1200, 2000])
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3", help="'none' to skip")
    p.add_argument("--skip-llm", action="store_true", help="retrieval stages only")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out", default="reports")
    p.add_argument("--update-readme", action="store_true")
    p.set_defaults(func=cmd_experiments)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
