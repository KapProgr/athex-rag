"""Column-aware text, page headings and glossary expansion."""

import pytest

from athex_rag.chunking import Chunk
from athex_rag.embeddings import NullEmbedder
from athex_rag.index import Index
from athex_rag.ingest import column_texts, page_headings, page_text
from athex_rag.retrieval import Retriever


def _two_column_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=1190, height=842)  # landscape, like the Motor Oil report
    group = "Group row {}: consolidated revenue of the group for the financial year 11.482.{}"
    company = "Company row {}: revenue of the parent company for the financial year 7.698.{}"
    for i in range(6):
        # two columns of running text whose lines sit at slightly different heights,
        # wide enough to count as columns (not table cells)
        page.insert_text((60, 100 + i * 30), group.format(i, i))
        page.insert_text((640, 112 + i * 30), company.format(i, i))
    doc.save(path)
    return pymupdf.open(path)[0]


def _order(text: str) -> list[str]:
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def test_two_column_page_is_read_column_by_column(tmp_path):
    page = _two_column_pdf(tmp_path / "TWO_2025.pdf")
    # sorted by height, group and company lines alternate
    assert _order(page_text(page, "sorted"))[:4] == ["Group", "Company", "Group", "Company"]
    # column by column: all group lines, then all company lines
    assert _order(page_text(page, "columns")) == ["Group"] * 6 + ["Company"] * 6
    assert len(column_texts(tmp_path / "TWO_2025.pdf")) == 1


def test_one_column_page_is_unchanged(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(8):
        page.insert_text((72, 80 + i * 20), f"A full-width line of running text number {i}")
    assert page_text(page, "columns") == page_text(page, "sorted")


def test_page_headings_skip_running_headers():
    pages = [f"ΟΜΙΛΟΣ ΧΥΖ Α.Ε.\nΕτήσια Έκθεση 2025\nΕνότητα {i}\nκείμενο" for i in range(10)]
    pages[3] = "ΟΜΙΛΟΣ ΧΥΖ Α.Ε.\nΕτήσια Έκθεση 2025\nΚΑΤΑΣΤΑΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ\n12.345 67.890"
    heads = page_headings(pages)
    assert heads[3].startswith("ΚΑΤΑΣΤΑΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    assert "ΟΜΙΛΟΣ" not in heads[3] and "12.345" not in heads[3]


def test_glossary_expands_keyword_query():
    chunks = [
        Chunk("A_2025:p1:t0", "A_2025", "A", 2025, 1, "Η εταιρεία ανέθεσε τον έλεγχο της χρήσης"),
        Chunk(
            "A_2025:p9:t0",
            "A_2025",
            "A",
            2025,
            9,
            "Ο Ορκωτός Ελεγκτής Λογιστής Γιάννης Π. ΑΜ ΣΟΕΛ 12345, Deloitte",
        ),
    ]
    index = Index(chunks, NullEmbedder().embed_documents([c.text for c in chunks]), "none")
    question = "Ποια ελεγκτική εταιρεία έλεγξε την Α;"
    plain = Retriever(index, NullEmbedder(), mode="bm25")
    expanded = Retriever(
        index,
        NullEmbedder(),
        mode="bm25",
        glossary={"ελεγκτική εταιρεία": ["ορκωτός ελεγκτής λογιστής", "ΣΟΕΛ"]},
    )
    assert "σοελ" in expanded.keyword_tokens(question)
    assert expanded.search(question, k=1)[0].chunk.page == 9
    assert plain.search(question, k=1)[0].chunk.page == 1
