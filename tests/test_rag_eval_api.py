import json

from fastapi.testclient import TestClient

from athex_rag import api
from athex_rag.evaluation import EvalItem, evaluate_item, load_items, summarize, to_markdown
from athex_rag.llm import FakeLLM, make_llm
from athex_rag.rag import RAG, build_prompt, is_refusal, parse_citations


def test_parse_citations_dedupes_in_order():
    assert parse_citations("Χ [2] και Ψ [1][2]. [12]") == [2, 1, 12]


def test_refusal_detection():
    assert is_refusal("Δεν βρέθηκε η πληροφορία στα έγγραφα.")
    assert not is_refusal("Τα κέρδη ήταν 10 εκατ. [1]")


def test_prompt_labels_sources(retriever):
    hits = retriever.search("καθαρά κέρδη", k=2)
    prompt = build_prompt("Ερώτηση;", hits)
    assert "[1] " in prompt and "σελ. " in prompt and prompt.rstrip().endswith("Απάντηση:")


def test_rag_with_fake_llm_cites_source(retriever):
    answer = RAG(retriever, FakeLLM()).ask("καθαρά κέρδη OPAP 2024")
    assert answer.cited == [1]
    assert answer.cited_hits[0].chunk.page == 12


def test_make_llm_specs():
    assert make_llm("fake").name == "fake"
    assert make_llm("ollama:llama3.1:8b").name == "ollama:llama3.1:8b"
    assert make_llm("openai:m@http://x/v1").base_url == "http://x/v1"


def test_evaluate_numeric_item(retriever):
    item = EvalItem(
        "q1", "Καθαρά κέρδη OPAP 2024;", "OPAP", 2024, [12], "numeric", answer_value=1_234_500_000
    )
    result = evaluate_item(item, retriever, RAG(retriever, FakeLLM()), k=3, use_filters=False)
    assert result.hit and result.rr == 1.0
    assert result.correct and result.citation_precision == 1.0


def test_summary_and_markdown(retriever, tmp_path):
    path = tmp_path / "q.jsonl"
    rows = [
        {
            "id": "a",
            "question": "μέλη διοικητικού συμβουλίου",
            "company": "OPAP",
            "year": 2024,
            "gold_pages": [30],
            "type": "text",
            "answer": "εννέα μέλη",
        },
        {
            "id": "b",
            "question": "κύκλος εργασιών JUMBO",
            "company": "JUMBO",
            "year": 2024,
            "gold_pages": [8],
            "type": "numeric",
            "answer_value": 1_050_000_000,
        },
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    items = load_items(path)
    results = [evaluate_item(i, retriever, None, 3, False) for i in items]
    summary = summarize(results)
    assert summary["hit@k"] == 1.0
    table = to_markdown({"hybrid": summary})
    assert "| hybrid | 100.0% |" in table


def test_api_ask(retriever):
    api.app.dependency_overrides[api.get_rag] = lambda: RAG(retriever, FakeLLM())
    try:
        client = TestClient(api.app)
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/ask", json={"question": "καταστήματα σε ποιες χώρες;"})
        body = r.json()
        assert r.status_code == 200
        assert body["sources"][0]["cited"] is True
        assert body["refused"] is False
        docs = client.get("/documents").json()["documents"]
        assert {"company": "JUMBO", "company_name": "Jumbo Α.Ε.Ε.", "year": 2024} in docs
    finally:
        api.app.dependency_overrides.clear()


def test_alternative_answers_are_accepted():
    from athex_rag.evaluation import judge_answer

    item = EvalItem(
        "x",
        "Ελεγκτής;",
        "PPC",
        2024,
        [1],
        "text",
        answer="Ernst & Young",
        alternatives=["ΕΡΝΣΤ ΓΙΑΝΓΚ"],
    )
    assert judge_answer("Ο ελεγκτής είναι η ΕΡΝΣΤ ΓΙΑΝΓΚ (ΕΛΛΑΣ) [1]", item)[0]
    assert judge_answer("Ernst & Young [2]", item)[0]
    assert not judge_answer("Deloitte [1]", item)[0]


class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_groq_spec_and_rate_limit_retry(monkeypatch):
    from athex_rag import llm as llm_mod

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = make_llm("groq:llama-3.3-70b-versatile")
    assert client.name == "groq:llama-3.3-70b-versatile"
    assert client.base_url == "https://api.groq.com/openai/v1"

    calls = []
    ok = {"choices": [{"message": {"content": " answer [1] "}}]}
    responses = [_Resp(429, headers={"retry-after": "0"}), _Resp(503), _Resp(200, ok)]
    monkeypatch.setattr(client, "_post", lambda payload: calls.append(payload) or responses.pop(0))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    assert client.generate("sys", "prompt") == "answer [1]"
    assert len(calls) == 3 and calls[0]["model"] == "llama-3.3-70b-versatile"


def test_missing_api_key_is_a_clear_error(monkeypatch):
    import pytest

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        make_llm("groq:x").generate("s", "p")


def test_http_error_includes_server_message(monkeypatch):
    import pytest

    monkeypatch.setenv("LLM_API_KEY", "k")
    client = make_llm("groq:old-model")
    body = {"error": {"message": "The model `old-model` does not exist", "code": "model_not_found"}}
    monkeypatch.setattr(client, "_post", lambda payload: _Resp(404, body))
    with pytest.raises(RuntimeError, match="model_not_found"):
        client.generate("s", "p")


def test_reasoning_block_is_removed():
    from athex_rag.llm import strip_reasoning

    raw = "<think>Ίσως 999 εκατ. [3]... όχι, [1].</think>\n\nΤα κέρδη ήταν 501,1 εκατ. € [1]"
    assert strip_reasoning(raw) == "Τα κέρδη ήταν 501,1 εκατ. € [1]"
    assert strip_reasoning("σκέψη [2]</think> Απάντηση [1]") == "Απάντηση [1]"
    assert strip_reasoning("Απλή απάντηση [1]") == "Απλή απάντηση [1]"


def test_full_width_citations_are_parsed():
    assert parse_citations("Κέρδη 501,1 εκατ.【1】【3】 και [2]") == [1, 3, 2]


def test_answer_found_on_non_gold_page_counts(chunks):
    from athex_rag.evaluation import score

    # the answer value appears on page 12 of OPAP 2024 (conftest), but gold says page 99
    item = EvalItem("q", "Κέρδη;", "OPAP", 2024, [99], "numeric", answer_value=1_234_500_000)
    on_12 = [c for c in chunks if c.company == "OPAP" and c.year == 2024 and c.page == 12]
    r = score(item, on_12, "Τα κέρδη ήταν 1.234,5 εκατ. ευρώ 【1】")
    assert not r.hit and r.answer_in_context and r.citation_support == 1.0
    assert r.category == "ok"


def test_common_small_numbers_do_not_count_as_support(chunks):
    from athex_rag.evaluation import score

    # 9% appears in the JUMBO chunk, but "9" is not distinctive enough to prove support
    item = EvalItem("q", "Μέλη ΔΣ;", "JUMBO", 2024, [99], "numeric", answer_value=9)
    jumbo = [c for c in chunks if c.company == "JUMBO" and c.page == 8]
    r = score(item, jumbo, "9 μέλη [1]")
    assert r.answer_in_context is False and r.category == "ok_bad_citation"


def test_rescore_saved_run(tmp_path, chunks):
    from athex_rag.evaluation import rescore_run

    item = EvalItem("q", "Κέρδη;", "OPAP", 2024, [12], "numeric", answer_value=1_234_500_000)
    row = {
        "id": "q",
        "retrieved": ["OPAP_2024:p12:t0"],
        "answer": "1.234,5 εκατ. 【1】",
        "correct": True,
        "latency_s": 1.0,
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    (r,) = rescore_run(path, [item], {c.id: c for c in chunks})
    assert r.correct and r.citation_precision == 1.0 and r.category == "ok"
