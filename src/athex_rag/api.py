"""HTTP API and web UI.

Run:  uvicorn athex_rag.api:app --reload      -> open http://localhost:8000
Env:  ATHEX_INDEX_DIR   index folder (default ./index)
      ATHEX_PDF_DIR     PDFs, served so citations open on the right page (default ./data/raw)
      ATHEX_LLM         LLM spec (default ollama:qwen2.5:7b)
      ATHEX_RETRIEVAL   dense | bm25 | hybrid (default hybrid)
      ATHEX_RERANKER    cross-encoder model name, or unset for none
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .companies import display_name, load_aliases, load_companies
from .embeddings import make_embedder
from .index import Index
from .llm import make_llm
from .rag import RAG
from .rerank import make_reranker
from .retrieval import Retriever

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    company: str | None = None
    year: int | None = None
    k: int = Field(default=5, ge=1, le=20)


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


@lru_cache(maxsize=1)
def get_rag() -> RAG:
    index_dir = Path(os.environ.get("ATHEX_INDEX_DIR", "index"))
    if not (index_dir / "meta.json").exists():
        raise RuntimeError(f"No index at {index_dir}. Run `athex-rag ingest data/raw` first.")
    index = Index.load(index_dir)
    retriever = Retriever(
        index,
        make_embedder(index.embedder_name),
        mode=os.environ.get("ATHEX_RETRIEVAL", "hybrid"),
        reranker=make_reranker(os.environ.get("ATHEX_RERANKER")),
        aliases=load_aliases(),
    )
    return RAG(retriever, make_llm())


def get_pdf_dir() -> Path:
    return Path(os.environ.get("ATHEX_PDF_DIR", "data/raw"))


def create_app() -> FastAPI:
    app = FastAPI(
        title="athex-rag", version="0.2.0", description="Cited Q&A over ATHEX annual reports"
    )
    companies = load_companies()

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
    def pdf(doc_id: str) -> FileResponse:
        if not re.fullmatch(r"[A-Za-z0-9\-]+_\d{4}", doc_id):
            raise HTTPException(status_code=404)
        path = get_pdf_dir() / f"{doc_id}.pdf"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="PDF not available")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, rag: RAG = Depends(get_rag)) -> AskResponse:
        start = time.perf_counter()
        try:
            answer = rag.ask(req.question, company=req.company, year=req.year, k=req.k)
        except Exception as exc:  # LLM backend down, etc.
            raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}") from exc
        cited = set(answer.cited)
        sources = [
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
            sources=sources,
            latency_s=round(time.perf_counter() - start, 2),
        )

    return app


app = create_app()
