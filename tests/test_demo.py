"""The keyword-only deployment used for the hosted demo."""

import gzip
import json

from fastapi.testclient import TestClient

from athex_rag import api
from athex_rag.api import RateLimiter
from athex_rag.embeddings import NullEmbedder, make_embedder
from athex_rag.index import Index
from athex_rag.llm import FakeLLM
from athex_rag.rag import RAG
from athex_rag.retrieval import Retriever


def test_rate_limiter_per_client_and_per_day():
    limiter = RateLimiter(per_minute=2, per_day=3)
    assert limiter.check("a", now=0) is None
    assert limiter.check("a", now=1) is None
    assert "λεπτό" in limiter.check("a", now=2)  # third in the same minute
    assert limiter.check("b", now=3) is None  # other client is fine
    assert "ημερήσιο" in limiter.check("c", now=4)  # global daily cap reached (3)
    assert limiter.check("a", now=100_000) is None  # a day later
    assert RateLimiter().check("x") is None  # no limits configured


def test_bm25_only_index_with_gzipped_chunks(chunks, tmp_path):
    index = Index.build(chunks, NullEmbedder())
    index.save(tmp_path)
    raw = (tmp_path / "chunks.jsonl").read_bytes()
    (tmp_path / "chunks.jsonl.gz").write_bytes(gzip.compress(raw))
    (tmp_path / "chunks.jsonl").unlink()
    loaded = Index.load(tmp_path)
    assert loaded.embedder_name == "none" and loaded.vectors.shape == (len(chunks), 0)
    hits = Retriever(loaded, make_embedder("none"), mode="bm25").search("καταστήματα Ρουμανία")
    assert hits[0].chunk.page == 40


def test_get_rag_defaults_to_bm25_for_keyword_index(chunks, tmp_path, monkeypatch):
    Index.build(chunks, NullEmbedder()).save(tmp_path)
    monkeypatch.setenv("ATHEX_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("ATHEX_LLM", "fake")
    monkeypatch.delenv("ATHEX_RETRIEVAL", raising=False)
    api.get_rag.cache_clear()
    try:
        assert api.get_rag().retriever.mode == "bm25"
    finally:
        api.get_rag.cache_clear()


def test_pdf_redirects_to_official_source(tmp_path, monkeypatch):
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"OTE_2025": "https://example.com/ΟΤΕ 2025.pdf"}), "utf-8")
    monkeypatch.setenv("ATHEX_SOURCES", str(sources))
    monkeypatch.setenv("ATHEX_PDF_DIR", str(tmp_path / "none"))
    client = TestClient(api.create_app(), follow_redirects=False)
    r = client.get("/pdf/OTE_2025")
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://example.com/")
    assert " " not in r.headers["location"]  # URL-encoded
    assert client.get("/pdf/JUMBO_2025").status_code == 404


def test_ask_is_rate_limited_and_hides_llm_errors(retriever, monkeypatch):
    monkeypatch.setenv("ATHEX_RATE_PER_MINUTE", "1")

    class Down:
        name = "down"

        def generate(self, system, prompt):
            raise RuntimeError("secret upstream detail")

    app = api.create_app()
    app.dependency_overrides[api.get_rag] = lambda: RAG(retriever, Down())
    client = TestClient(app)
    first = client.post("/ask", json={"question": "καθαρά κέρδη OPAP 2024"})
    assert first.status_code == 502 and "secret" not in first.text
    assert client.post("/ask", json={"question": "καθαρά κέρδη OPAP 2024"}).status_code == 429

    app.dependency_overrides[api.get_rag] = lambda: RAG(retriever, FakeLLM())
    monkeypatch.setenv("ATHEX_RATE_PER_MINUTE", "0")
