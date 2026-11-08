import pytest

from athex_rag.chunking import Chunk
from athex_rag.embeddings import HashingEmbedder
from athex_rag.index import Index
from athex_rag.retrieval import Retriever

TOY = [
    ("OPAP", 2024, 12, "Τα καθαρά κέρδη του Ομίλου ανήλθαν σε 1.234,5 εκατ. ευρώ."),
    ("OPAP", 2024, 30, "Το διοικητικό συμβούλιο αποτελείται από εννέα μέλη."),
    ("OPAP", 2023, 12, "Τα καθαρά κέρδη του Ομίλου ανήλθαν σε 980,1 εκατ. ευρώ."),
    ("JUMBO", 2024, 8, "Ο κύκλος εργασιών αυξήθηκε κατά 9% σε 1.050 εκατ. ευρώ."),
    (
        "JUMBO",
        2024,
        40,
        "Η εταιρεία διαθέτει καταστήματα σε Ελλάδα, Κύπρο, Βουλγαρία και Ρουμανία.",
    ),
]


@pytest.fixture
def chunks():
    return [Chunk(f"{c}_{y}:p{p}:t0", f"{c}_{y}", c, y, p, text) for c, y, p, text in TOY]


@pytest.fixture
def embedder():
    return HashingEmbedder(256)


@pytest.fixture
def index(chunks, embedder):
    return Index.build(chunks, embedder)


@pytest.fixture
def retriever(index, embedder):
    return Retriever(index, embedder, mode="hybrid")
