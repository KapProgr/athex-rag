import json

import pytest
from fastapi.testclient import TestClient

from athex_rag import api
from athex_rag.evaluation import EvalItem, ItemResult, categorize, error_report
from athex_rag.experiments import Plan, Runner, update_readme
from athex_rag.ingest import load_pages
from athex_rag.llm import FakeLLM
from athex_rag.rag import RAG
from athex_rag.rerank import KeywordOverlapReranker
from athex_rag.retrieval import Retriever, infer_filters


def test_aliases_handle_greek_names():
    aliases = {"ETE": ["Εθνικής Τράπεζας"], "PPC": ["ΔΕΗ"]}
    assert infer_filters("Κέρδη της Εθνικής Τράπεζας 2025", ["ETE", "PPC"], [2025], aliases) == (
        "ETE",
        2025,
    )
    assert infer_filters("Εργαζόμενοι στη ΔΕΗ", ["ETE", "PPC"], [2025], aliases) == ("PPC", None)
    # "ΔΕΗ" must match as a whole word, not inside another word
    assert infer_filters("ΔΕΗΣ", ["PPC"], [], aliases) == (None, None)


def test_reranker_is_applied(index, embedder):
    retriever = Retriever(index, embedder, mode="bm25", reranker=KeywordOverlapReranker())
    hits = retriever.search("καταστήματα Ρουμανία Βουλγαρία", k=2)
    assert hits[0].chunk.page == 40 and len(hits) <= 2


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"type": "numeric", "hit": False, "correct": False, "refused": False}, "retrieval_miss"),
        ({"type": "numeric", "hit": True, "correct": False, "refused": True}, "false_refusal"),
        ({"type": "numeric", "hit": True, "correct": False, "refused": False}, "wrong_answer"),
        ({"type": "text", "hit": True, "correct": True, "citation_precision": 1.0}, "ok"),
        ({"type": "text", "hit": True, "correct": True}, "ok_bad_citation"),
        ({"type": "unanswerable", "hit": False, "correct": False}, "hallucination"),
        ({"type": "unanswerable", "hit": False, "correct": True}, "refused_correctly"),
        ({"type": "numeric", "hit": True}, "retrieved"),
    ],
)
def test_categorize(kwargs, expected):
    kwargs = {"rr": 0.0, **kwargs}
    assert categorize(ItemResult(id="x", **kwargs)) == expected


def test_error_report_lists_examples():
    item = EvalItem("q1", "Πόσα μέλη;", "OPAP", 2024, [30], "text", answer="εννέα")
    res = ItemResult("q1", "text", False, 0.0, correct=False, refused=False, answer="δέκα")
    res.category = categorize(res)
    report = error_report([res], [item])
    assert "retrieval_miss" in report and "Πόσα μέλη;" in report


def _make_pdfs(folder):
    pymupdf = pytest.importorskip("pymupdf")
    texts = {
        "ACME_2024": ["Net profit for 2024 was 120.5 million euro.", "The board has 9 members."],
        "ACME_2023": ["Net profit for 2023 was 98.1 million euro.", "The board has 7 members."],
    }
    for doc_id, pages in texts.items():
        doc = pymupdf.open()
        for text in pages:
            doc.new_page().insert_text((72, 72), text)
        doc.save(folder / f"{doc_id}.pdf")


def test_page_cache_roundtrip(tmp_path):
    _make_pdfs(tmp_path)
    pdf = tmp_path / "ACME_2024.pdf"
    first = load_pages(pdf, tmp_path / "cache")
    assert (tmp_path / "cache" / "ACME_2024.json").exists()
    assert load_pages(pdf, tmp_path / "cache") == first


def test_full_experiment_plan_runs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_pdfs(raw)
    items = [
        EvalItem("a", "net profit ACME 2024", "ACME", 2024, [1], "numeric", answer_value=120.5e6),
        EvalItem("b", "board members ACME 2023", "ACME", 2023, [2], "text", answer="7 members"),
        EvalItem("c", "CEO salary of ACME", "ACME", 2024, [], "unanswerable"),
    ]
    plan = Plan(
        pdf_dir=raw,
        items=items,
        embedders=["hashing-128", "hashing-256"],
        llms=["fake"],
        chunk_sizes=[300, 800],
        reranker="keyword-overlap",
        k=3,
        out=tmp_path / "reports",
        index_root=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
    )
    report = Runner(plan, log=lambda *_: None).execute()
    assert "## 1. Chunk size × retrieval mode" in report
    assert "## 5. Answer generation" in report and "Error analysis" in report
    data = json.loads((tmp_path / "reports" / "experiments.json").read_text(encoding="utf-8"))
    assert (tmp_path / data["best_config"]["index_dir"]).exists() or __import__("pathlib").Path(
        data["best_config"]["index_dir"]
    ).exists()
    assert set(data["stages"]) == {
        "1 chunk x mode",
        "2 tables",
        "3 embedder",
        "4 reranker",
        "5 llm",
    }

    readme = tmp_path / "README.md"
    readme.write_text("# x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nend\n")
    assert update_readme(readme, report)
    text = readme.read_text()
    assert "old" not in text and "### 1. Chunk size" in text and text.endswith("end\n")


def test_web_ui_and_pdf_route(retriever, tmp_path, monkeypatch):
    (tmp_path / "OPAP_2024.pdf").write_bytes(b"%PDF-1.4 test")
    monkeypatch.setenv("ATHEX_PDF_DIR", str(tmp_path))
    api.app.dependency_overrides[api.get_rag] = lambda: RAG(retriever, FakeLLM())
    try:
        client = TestClient(api.app)
        assert "athex-rag" in client.get("/").text
        assert client.get("/pdf/OPAP_2024").content.startswith(b"%PDF")
        assert client.get("/pdf/..%2Fsecret_2024").status_code == 404
        assert client.get("/pdf/JUMBO_2024").status_code == 404
        body = client.post("/ask", json={"question": "καθαρά κέρδη OPAP 2024"}).json()
        assert body["sources"][0]["doc_id"] == "OPAP_2024" and "latency_s" in body
    finally:
        api.app.dependency_overrides.clear()


def test_llm_failure_keeps_retrieval_results(tmp_path, monkeypatch):
    from athex_rag import experiments

    raw = tmp_path / "raw"
    raw.mkdir()
    _make_pdfs(raw)
    items = [EvalItem("a", "net profit ACME 2024", "ACME", 2024, [1], "numeric", answer_value=1)]
    plan = Plan(
        pdf_dir=raw,
        items=items,
        embedders=["hashing-64"],
        llms=["fake"],
        chunk_sizes=[800],
        reranker=None,
        k=3,
        out=tmp_path / "reports",
        index_root=tmp_path / "idx",
        cache_dir=tmp_path / "cache",
    )
    calls = {"n": 0}

    class Boom:
        name = "boom"

        def generate(self, system, prompt):
            calls["n"] += 1
            if calls["n"] > 1:  # the preflight check passes, the real run fails
                raise RuntimeError("HTTP 500")
            return "OK"

    monkeypatch.setattr(experiments, "make_llm", lambda spec: Boom())
    report = Runner(plan, log=lambda *_: None).execute()
    assert "## 1. Chunk size" in report and "Run failed" in report
    assert (tmp_path / "reports" / "experiments.json").exists()
