"""HTTP API and web UI.

Run:  uvicorn athex_rag.api:app --reload      -> open http://localhost:8000
Env:  ATHEX_INDEX_DIR   index folder (default ./index)
      ATHEX_PDF_DIR     local PDFs; when a report is not there, "Open PDF" redirects to the
                        company's official copy listed in data/sources.json (default ./data/raw)
      ATHEX_LLM         LLM spec (default ollama:qwen2.5:7b)
      ATHEX_RETRIEVAL   dense | bm25 | hybrid (default hybrid)
      ATHEX_RERANKER    cross-encoder model name, or unset for none
      ATHEX_WARMUP=1    load the index at startup
      ATHEX_RATE_PER_MINUTE / ATHEX_RATE_PER_DAY
                        per-visitor and global limits on /ask (default: unlimited). Set them
                        on a public deployment so nobody can drain the LLM quota.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .companies import display_name, load_aliases, load_companies
from .embeddings import make_embedder
from .index import Index
from .llm import make_llm
from .rag import RAG
from .rerank import make_reranker
from .retrieval import Retriever

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("athex_rag.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    company: str | None = None
    year: int | None = None
    k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    n: int
    doc_id: str
    company: str
    company_name: str
    year: int
    page: int
    kind: str
    cited: bool
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    latency_s: float


class RateLimiter:
    """In-memory sliding-window limits: per client per minute, and globally per day."""

    def __init__(self, per_minute: int = 0, per_day: int = 0):
        self.per_minute, self.per_day = per_minute, per_day
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._day: deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(
            int(os.environ.get("ATHEX_RATE_PER_MINUTE", "0")),
            int(os.environ.get("ATHEX_RATE_PER_DAY", "0")),
        )

    def check(self, client: str, now: float | None = None) -> str | None:
        """Record a request; return an error message if it is over a limit."""
        now = time.time() if now is None else now
        with self._lock:
            if self.per_day:
                while self._day and now - self._day[0] > 86_400:
                    self._day.popleft()
                if len(self._day) >= self.per_day:
                    return "Το demo έφτασε το ημερήσιο όριο ερωτήσεων. Δοκιμάστε ξανά αύριο."
            if self.per_minute:
                window = self._clients[client]
                while window and now - window[0] > 60:
                    window.popleft()
                if len(window) >= self.per_minute:
                    return "Πολλές ερωτήσεις σε λίγο χρόνο. Περιμένετε ένα λεπτό."
                window.append(now)
            if self.per_day:
                self._day.append(now)
        return None


@lru_cache(maxsize=1)
def get_rag() -> RAG:
    index_dir = Path(os.environ.get("ATHEX_INDEX_DIR", "index"))
    if not (index_dir / "meta.json").exists():
        raise RuntimeError(f"No index at {index_dir}. Run `athex-rag ingest data/raw` first.")
    index = Index.load(index_dir)
    default_mode = "bm25" if index.embedder_name == "none" else "hybrid"
    retriever = Retriever(
        index,
        make_embedder(index.embedder_name),
        mode=os.environ.get("ATHEX_RETRIEVAL", default_mode),
        reranker=make_reranker(os.environ.get("ATHEX_RERANKER")),
        aliases=load_aliases(),
    )
    return RAG(retriever, make_llm())


def get_pdf_dir() -> Path:
    return Path(os.environ.get("ATHEX_PDF_DIR", "data/raw"))


def load_sources(path: Path | None = None) -> dict[str, str]:
    path = path or Path(os.environ.get("ATHEX_SOURCES", "data/sources.json"))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")  # behind the Spaces proxy
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def create_app() -> FastAPI:
    app = FastAPI(
        title="athex-rag", version="0.3.0", description="Cited Q&A over ATHEX annual reports"
    )
    companies = load_companies()
    sources = load_sources()
    limiter = RateLimiter.from_env()

    if os.environ.get("ATHEX_WARMUP") == "1":
        # load the index in the background at startup, not on the first visitor's question
        threading.Thread(target=get_rag, daemon=True).start()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/documents")
    def documents(rag: RAG = Depends(get_rag)) -> dict:
        docs = sorted({(c.company, c.year) for c in rag.retriever.index.chunks})
        return {
            "documents": [
                {"company": c, "company_name": display_name(c, companies), "year": y}
                for c, y in docs
            ]
        }

    @app.get("/pdf/{doc_id}", include_in_schema=False)
    def pdf(doc_id: str):
        if not re.fullmatch(r"[A-Za-z0-9\-]+_\d{4}", doc_id):
            raise HTTPException(status_code=404)
        path = get_pdf_dir() / f"{doc_id}.pdf"
        if path.is_file():
            return FileResponse(path, media_type="application/pdf")
        if doc_id in sources:
            # browsers keep the #page=N fragment across the redirect
            return RedirectResponse(sources[doc_id], status_code=302)
        raise HTTPException(status_code=404, detail="PDF not available")

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request, rag: RAG = Depends(get_rag)) -> AskResponse:
        limited = limiter.check(client_id(request))
        if limited:
            raise HTTPException(status_code=429, detail=limited)
        start = time.perf_counter()
        try:
            answer = rag.ask(req.question, company=req.company, year=req.year, k=req.k)
        except Exception as exc:  # LLM backend down, rate-limited, etc.
            log.exception("answer generation failed")
            raise HTTPException(
                status_code=502,
                detail="Η δημιουργία απάντησης απέτυχε προσωρινά. Δοκιμάστε ξανά σε λίγο.",
            ) from exc
        cited = set(answer.cited)
        sources_out = [
            Source(
                n=n,
                doc_id=h.chunk.doc_id,
                company=h.chunk.company,
                company_name=display_name(h.chunk.company, companies),
                year=h.chunk.year,
                page=h.chunk.page,
                kind=h.chunk.kind,
                cited=n in cited,
                snippet=h.chunk.text[:600],
            )
            for n, h in enumerate(answer.sources, start=1)
        ]
        return AskResponse(
            answer=answer.text,
            refused=answer.refused,
            sources=sources_out,
            latency_s=round(time.perf_counter() - start, 2),
        )

    return app


app = create_app()
