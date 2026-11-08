"""PDF ingestion: annual report PDFs -> page text + tables -> chunks.

Files are expected as ``data/raw/<COMPANY>_<YEAR>.pdf`` (e.g. ``OPAP_2024.pdf``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

from .chunking import Chunk, chunk_page

_NAME_RE = re.compile(r"^(?P<company>[A-Za-z0-9\-]+)_(?P<year>\d{4})$")


def parse_doc_name(path: Path) -> tuple[str, int]:
    match = _NAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"{path.name}: expected <COMPANY>_<YEAR>.pdf, e.g. OPAP_2024.pdf")
    return match["company"].upper(), int(match["year"])


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    clean = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    clean = [row for row in clean if any(row)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def _page_tables(page) -> list[str]:
    import pymupdf

    tables: list[str] = []
    try:
        for table in page.find_tables().tables:
            # Read each cell by clipping the page to the cell box. Table.extract()
            # can scramble digit order and separators in numeric cells
            # ("2.150,3" -> "2 150 3 . ,"), which ruins financial figures.
            rows = [
                [
                    page.get_text("text", clip=pymupdf.Rect(cell), sort=True) if cell else None
                    for cell in row.cells
                ]
                for row in table.rows
            ]
            md = _table_to_markdown(rows)
            if md:
                tables.append(md)
    except Exception:  # table detection is best-effort
        pass
    return tables


def _extract_range(
    path: str, start: int, end: int, with_tables: bool
) -> list[tuple[int, str, list[str]]]:
    """Pages [start, end) of one PDF. Runs in a worker process (see extract_pages)."""
    import pymupdf

    out = []
    with pymupdf.open(path) as doc:
        for index in range(start, min(end, len(doc))):
            page = doc[index]
            text = page.get_text("text", sort=True)
            if with_tables and os.environ.get("ATHEX_TEST_CRASH_PAGE") == str(index + 1):
                os._exit(139)  # test hook: simulate a MuPDF segfault on this page
            out.append((index + 1, text, _page_tables(page) if with_tables else []))
    return out


def extract_pages(
    path: Path, with_tables: bool = True, workers: int | None = None, batch: int = 20
) -> list[tuple[int, str, list[str]]]:
    """Return ``(page_number, text, tables_as_markdown)`` for each page.

    Table detection runs in worker processes, ``batch`` pages at a time. MuPDF can
    *segfault* on unusual pages (it happened on two of the ten reports), which would
    kill a single-process run; here a crashed batch is retried page by page, and a page
    that crashes on its own keeps its text but loses its tables, with a warning.
    """
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    import pymupdf

    with pymupdf.open(path) as doc:
        n_pages = len(doc)
    if not with_tables:
        return _extract_range(str(path), 0, n_pages, False)

    workers = workers or max(1, (os.cpu_count() or 2))
    ranges = [(s, min(s + batch, n_pages)) for s in range(0, n_pages, batch)]
    done: dict[int, tuple[int, str, list[str]]] = {}
    failed: list[tuple[int, int]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_range, str(path), s, e, True): (s, e) for s, e in ranges}
        for future, rng in futures.items():
            try:
                for page in future.result():
                    done[page[0]] = page
            except BrokenProcessPool:
                failed.append(rng)

    for start, end in failed:  # isolate the crashing page(s)
        for index in range(start, end):
            if index + 1 in done:
                continue
            try:
                with ProcessPoolExecutor(max_workers=1) as single:
                    (page,) = single.submit(
                        _extract_range, str(path), index, index + 1, True
                    ).result()
            except BrokenProcessPool:
                warnings.warn(
                    f"{path.name} p.{index + 1}: table detection crashed; keeping text only",
                    stacklevel=2,
                )
                (page,) = _extract_range(str(path), index, index + 1, False)
            done[page[0]] = page
    return [done[i] for i in sorted(done)]


def file_key(path: Path) -> str:
    """Size + SHA-1 of the first and last MiB: cheap, and stable across copies."""
    size = path.stat().st_size
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        digest.update(f.read(1 << 20))
        if size > 1 << 20:
            f.seek(max(size - (1 << 20), 0))
            digest.update(f.read())
    return f"{size}-{digest.hexdigest()[:16]}"


def load_pages(path: Path, cache_dir: Path | None = None) -> list[tuple[int, str, list[str]]]:
    """``extract_pages`` with an on-disk JSON cache.

    Table detection on a 300-page report takes minutes; experiments re-chunk the same
    PDFs many times, so parsed pages are cached, keyed on file size + content hash
    (not mtime, so a cache built on one machine is valid on another).
    """
    if cache_dir is None:
        return extract_pages(path)
    key = file_key(path)
    cache_file = cache_dir / f"{path.stem}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            return [(p["page"], p["text"], p["tables"]) for p in cached["pages"]]
    pages = extract_pages(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "pages": [{"page": n, "text": t, "tables": tabs} for n, t, tabs in pages],
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return pages


def chunks_for_document(
    path: Path,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    with_tables: bool = True,
    cache_dir: Path | None = None,
) -> list[Chunk]:
    company, year = parse_doc_name(path)
    doc_id = f"{company}_{year}"
    chunks: list[Chunk] = []
    pages = (
        load_pages(path, cache_dir)
        if cache_dir is not None
        else extract_pages(path, with_tables=with_tables)
    )
    for page_no, text, tables in pages:
        if not with_tables:
            tables = []
        for i, piece in enumerate(chunk_page(text, max_chars, overlap_chars)):
            chunks.append(Chunk(f"{doc_id}:p{page_no}:t{i}", doc_id, company, year, page_no, piece))
        for i, table in enumerate(tables):
            # Long tables are kept whole up to 3x the text budget; retrieval works on
            # the header + rows, and the LLM needs complete rows to read figures.
            chunks.append(
                Chunk(
                    f"{doc_id}:p{page_no}:tab{i}",
                    doc_id,
                    company,
                    year,
                    page_no,
                    table[: max_chars * 3],
                    kind="table",
                )
            )
    return chunks


def iter_pdfs(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())
