"""Run the full experiment plan with one command and write a report.

Stages (each picks the best configuration from the previous one):

1. Chunk size x retrieval mode       (600 / 1200 / 2000 chars x bm25 / dense / hybrid)
2. Table extraction ablation          (with vs without table chunks)
3. Embedding model                    (dense + hybrid for each embedder)
4. Reranker                           (best config + cross-encoder)
5. LLM                                (full pipeline for each LLM, with error analysis)

"Best" = highest hit@k, then MRR. Indexes are cached under ``indexes/`` and parsed
PDF pages under ``data/cache/``, so re-running after adding questions is fast.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .companies import load_aliases
from .embeddings import make_embedder
from .evaluation import (
    EvalItem,
    ItemResult,
    error_report,
    evaluate_item,
    save_run,
    summarize,
    to_markdown,
)
from .index import Index
from .ingest import chunks_for_document, iter_pdfs
from .llm import make_llm
from .rag import RAG
from .rerank import make_reranker
from .retrieval import Retriever

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


@dataclass
class Plan:
    pdf_dir: Path
    items: list[EvalItem]
    embedders: list[str]
    llms: list[str]
    chunk_sizes: list[int] = field(default_factory=lambda: [600, 1200, 2000])
    reranker: str | None = "BAAI/bge-reranker-v2-m3"
    k: int = 5
    out: Path = Path("reports")
    index_root: Path = Path("indexes")
    cache_dir: Path = Path("data/cache")
    overlap_ratio: float = 1 / 6


@dataclass(frozen=True)
class Config:
    chunk: int
    tables: bool
    embedder: str
    mode: str
    reranker: str | None = None
    llm: str | None = None

    @property
    def name(self) -> str:
        parts = [
            self.mode,
            _short(self.embedder) if self.mode != "bm25" else None,
            f"chunk{self.chunk}",
            None if self.tables else "no-tables",
            f"rerank:{_short(self.reranker)}" if self.reranker else None,
            self.llm,
        ]
        return " · ".join(p for p in parts if p)


def _short(model: str | None) -> str:
    return (model or "").split("/")[-1]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


class Runner:
    def __init__(self, plan: Plan, log=print):
        self.plan, self.log = plan, log
        self._embedders: dict[str, object] = {}
        self._indexes: dict[tuple, Index] = {}
        self._rerankers: dict[str, object] = {}
        self.aliases = load_aliases()
        self.rows: dict[str, dict[str, dict]] = {}  # stage -> config name -> summary
        self.error_sections: list[tuple[str, str]] = []

    def embedder(self, name: str):
        if name not in self._embedders:
            self._embedders[name] = make_embedder(name)
        return self._embedders[name]

    def index_dir(self, chunk: int, tables: bool, embedder: str) -> Path:
        name = f"c{chunk}-{'tab' if tables else 'notab'}-{_short(embedder)}"
        return self.plan.index_root / _slug(name)

    def index(self, chunk: int, tables: bool, embedder: str) -> Index:
        key = (chunk, tables, embedder)
        if key in self._indexes:
            return self._indexes[key]
        folder = self.index_dir(chunk, tables, embedder)
        if (folder / "meta.json").exists():
            index = Index.load(folder)
        else:
            chunks = []
            for pdf in iter_pdfs(self.plan.pdf_dir):
                chunks += chunks_for_document(
                    pdf,
                    chunk,
                    int(chunk * self.plan.overlap_ratio),
                    tables,
                    cache_dir=self.plan.cache_dir,
                )
            self.log(f"  embedding {len(chunks)} chunks ({folder.name}) ...")
            index = Index.build(chunks, self.embedder(embedder))
            index.save(folder)
        self._indexes[key] = index
        return index

    def run(self, stage: str, cfg: Config) -> tuple[dict, list[ItemResult]]:
        index = self.index(cfg.chunk, cfg.tables, cfg.embedder)
        reranker = None
        if cfg.reranker:
            if cfg.reranker not in self._rerankers:
                self._rerankers[cfg.reranker] = make_reranker(cfg.reranker)
            reranker = self._rerankers[cfg.reranker]
        retriever = Retriever(
            index,
            self.embedder(cfg.embedder),
            mode=cfg.mode,
            reranker=reranker,
            aliases=self.aliases,
        )
        rag = RAG(retriever, make_llm(cfg.llm), k=self.plan.k) if cfg.llm else None
        start = time.perf_counter()
        results = [evaluate_item(it, retriever, rag, self.plan.k, False) for it in self.plan.items]
        summary = summarize(results)
        save_run(self.plan.out / "runs", _slug(f"{stage}-{cfg.name}"), results, summary)
        self.rows.setdefault(stage, {})[cfg.name] = summary
        self.log(
            f"  {cfg.name}: hit@{self.plan.k}={summary['hit@k']:.1%} "
            f"({time.perf_counter() - start:.0f}s)"
        )
        return summary, results

    @staticmethod
    def _score(summary: dict) -> tuple:
        return (summary.get("hit@k") or 0, summary.get("mrr") or 0)

    def best(self, candidates: list[tuple[Config, dict]]) -> Config:
        return max(candidates, key=lambda pair: self._score(pair[1]))[0]

    def execute(self, skip_llm: bool = False) -> str:
        p = self.plan
        emb0 = p.embedders[0]

        self.log("Stage 1: chunk size x retrieval mode")
        stage1 = []
        for chunk in p.chunk_sizes:
            for mode in ("bm25", "dense", "hybrid"):
                cfg = Config(chunk, True, emb0, mode)
                stage1.append((cfg, self.run("1 chunk x mode", cfg)[0]))
        best = self.best(stage1)

        self.log("Stage 2: table extraction")
        no_tab = Config(best.chunk, False, best.embedder, best.mode)
        with_tab_summary = dict(stage1)[best]
        self.rows.setdefault("2 tables", {})[best.name] = with_tab_summary
        self.run("2 tables", no_tab)

        self.log("Stage 3: embedding model")
        stage3 = [(best, with_tab_summary)]
        self.rows.setdefault("3 embedder", {})[best.name] = with_tab_summary
        for emb in p.embedders[1:]:
            for mode in ("dense", "hybrid"):
                cfg = Config(best.chunk, True, emb, mode)
                stage3.append((cfg, self.run("3 embedder", cfg)[0]))
        best = self.best(stage3)

        if p.reranker:
            self.log("Stage 4: reranker")
            rr_cfg = Config(best.chunk, True, best.embedder, best.mode, reranker=p.reranker)
            self.rows.setdefault("4 reranker", {})[best.name] = dict(stage3)[best]
            rr_summary = self.run("4 reranker", rr_cfg)[0]
            if self._score(rr_summary) > self._score(dict(stage3)[best]):
                best = rr_cfg

        if not skip_llm:
            self.log("Stage 5: LLM (full pipeline)")
            for llm in p.llms:
                cfg = Config(best.chunk, best.tables, best.embedder, best.mode, best.reranker, llm)
                _, results = self.run("5 llm", cfg)
                self.error_sections.append((cfg.name, error_report(results, p.items)))

        return self.report(best)

    def report(self, best: Config) -> str:
        p = self.plan
        n = len(p.items)
        n_unans = sum(i.type == "unanswerable" for i in p.items)
        docs = sorted({f"{i.company} {i.year}" for i in p.items})
        lines = [
            "# Experiment results",
            "",
            f"{n} questions ({n - n_unans} answerable, {n_unans} unanswerable) over "
            f"{len(docs)} reports. k = {p.k}. Best retrieval configuration: **{best.name}**.",
        ]
        titles = {
            "1 chunk x mode": "1. Chunk size × retrieval mode",
            "2 tables": "2. Table extraction",
            "3 embedder": "3. Embedding model",
            "4 reranker": "4. Reranking",
            "5 llm": "5. Answer generation",
        }
        for stage, rows in self.rows.items():
            lines += ["", f"## {titles.get(stage, stage)}", "", to_markdown(rows)]
        for name, section in self.error_sections:
            lines += ["", f"## Error analysis: {name}", "", section]
        text = "\n".join(lines) + "\n"
        p.out.mkdir(parents=True, exist_ok=True)
        (p.out / "experiments.md").write_text(text, encoding="utf-8")
        (p.out / "experiments.json").write_text(
            json.dumps(
                {
                    "best": best.name,
                    "best_config": {
                        "index_dir": str(self.index_dir(best.chunk, best.tables, best.embedder)),
                        "mode": best.mode,
                        "reranker": best.reranker,
                        "chunk": best.chunk,
                        "tables": best.tables,
                        "embedder": best.embedder,
                    },
                    "stages": self.rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return text


def update_readme(readme: Path, report: str) -> bool:
    """Replace the block between the RESULTS markers with the report body."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    body = report.split("\n", 2)[2].strip()  # drop the "# Experiment results" heading
    body = body.replace("\n## ", "\n### ")
    if body.startswith("## "):
        body = "#" + body
    new = (
        text[: text.index(README_START) + len(README_START)]
        + "\n\n"
        + body
        + "\n\n"
        + text[text.index(README_END) :]
    )
    readme.write_text(new, encoding="utf-8")
    return True
