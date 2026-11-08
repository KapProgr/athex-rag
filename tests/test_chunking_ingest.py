from pathlib import Path

import pytest

from athex_rag.chunking import chunk_page
from athex_rag.ingest import _table_to_markdown, chunks_for_document, parse_doc_name


def test_chunks_respect_max_chars():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(10))
    chunks = chunk_page(text, max_chars=500, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Sentence number {i} " + "x" * 300 for i in range(4))
    chunks = chunk_page(text, max_chars=700, overlap_chars=150)
    assert any(chunks[i][-50:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_very_long_sentence_is_split():
    chunks = chunk_page("a" * 3000, max_chars=1000, overlap_chars=0)
    assert [len(c) for c in chunks] == [1000, 1000, 1000]


def test_parse_doc_name():
    assert parse_doc_name(Path("opap_2024.pdf")) == ("OPAP", 2024)
    with pytest.raises(ValueError):
        parse_doc_name(Path("annual report.pdf"))


def test_table_to_markdown():
    md = _table_to_markdown([["Item", "2024"], ["Revenue", "1.050"], [None, None]])
    assert md.splitlines() == ["| Item | 2024 |", "| --- | --- |", "| Revenue | 1.050 |"]


def test_ingest_real_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}: net profit was {100 + i} million euro.")
    path = tmp_path / "TEST_2024.pdf"
    doc.save(path)
    chunks = chunks_for_document(path, with_tables=False)
    assert [c.page for c in chunks] == [1, 2, 3]
    assert chunks[1].company == "TEST" and chunks[1].year == 2024
    assert "101 million" in chunks[1].text


def test_table_numbers_survive_extraction(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    rows = [
        ["Item", "2024", "2023"],
        ["Revenue", "2.150,3", "1.980,2"],
        ["EBITDA", "890,1", "801,0"],
    ]
    y = 100
    for row in rows:
        for i, cell in enumerate(row):
            page.draw_rect(pymupdf.Rect(72 + i * 120, y, 192 + i * 120, y + 20))
            page.insert_text((76 + i * 120, y + 14), cell, fontsize=10)
        y += 20
    path = tmp_path / "TAB_2024.pdf"
    doc.save(path)
    tables = [c for c in chunks_for_document(path) if c.kind == "table"]
    assert tables, "table not detected"
    assert "| Revenue | 2.150,3 | 1.980,2 |" in tables[0].text


def test_crashing_page_keeps_text_and_other_tables(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    from athex_rag.ingest import extract_pages

    doc = pymupdf.open()
    for i in range(5):
        doc.new_page().insert_text((72, 72), f"page {i + 1} text")
    path = tmp_path / "CRASH_2024.pdf"
    doc.save(path)
    monkeypatch.setenv("ATHEX_TEST_CRASH_PAGE", "3")
    with pytest.warns(UserWarning, match="p.3"):
        pages = extract_pages(path, workers=2, batch=2)
    assert [p[0] for p in pages] == [1, 2, 3, 4, 5]
    assert "page 3 text" in pages[2][1]
