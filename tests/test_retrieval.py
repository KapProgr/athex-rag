import pytest

from athex_rag.index import Index
from athex_rag.retrieval import Retriever, infer_filters, rrf


def test_rrf_prefers_items_ranked_high_in_both():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in (1, 2)
    assert fused[4] < fused[2]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_finds_board_size(index, embedder, mode):
    hits = Retriever(index, embedder, mode=mode).search("Πόσα μέλη έχει το διοικητικό συμβούλιο;")
    assert hits[0].chunk.page == 30


def test_accent_insensitive_keyword_search(index, embedder):
    hits = Retriever(index, embedder, mode="bm25").search("ΔΙΟΙΚΗΤΙΚΟ ΣΥΜΒΟΥΛΙΟ")
    assert hits[0].chunk.page == 30


def test_year_filter_inferred_from_question(retriever):
    hits = retriever.search("καθαρά κέρδη OPAP 2023", k=1)
    assert (hits[0].chunk.company, hits[0].chunk.year) == ("OPAP", 2023)


def test_explicit_filters(retriever):
    hits = retriever.search("κύκλος εργασιών", company="jumbo", year=2024)
    assert hits and all(h.chunk.company == "JUMBO" for h in hits)


def test_infer_filters_ambiguous_returns_none():
    assert infer_filters("OPAP vs JUMBO", ["JUMBO", "OPAP"], [2024]) == (None, None)


def test_save_and_load_roundtrip(index, embedder, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path)
    assert loaded.chunks == index.chunks
    hits = Retriever(loaded, embedder).search("καταστήματα Ρουμανία")
    assert hits[0].chunk.page == 40


def test_embedder_mismatch_is_rejected(index):
    from athex_rag.embeddings import HashingEmbedder

    with pytest.raises(ValueError):
        Retriever(index, HashingEmbedder(64))
