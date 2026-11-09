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


LAYOUTS = ("sorted", "columns")


def _column_split(page) -> float | None:
    """x-coordinate between the two columns if the page is laid out in two columns of
    running text, else None. Each PyMuPDF text block belongs to one column, but sorting
    lines by height interleaves the columns (Motor Oil prints group figures on the left
    and parent-company figures on the right, so rows of the two get mixed).

    Wide tables also have cells on both sides of the middle, so columns must consist of
    *wide* blocks (median >= 30% of the page width); narrow cells do not qualify."""
    width = page.rect.width
    mid, gap = width / 2, width * 0.04
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    left = [b for b in blocks if b[2] <= mid + gap]
    right = [b for b in blocks if b[0] >= mid - gap and b[2] > mid + gap]
    spanning = len(blocks) - len(left) - len(right)
    if len(left) < 4 or len(right) < 4 or spanning > 2:
        return None

    def median_width(bs: list) -> float:
        widths = sorted(b[2] - b[0] for b in bs)
        return widths[len(widths) // 2]

    if min(median_width(left), median_width(right)) < 0.3 * width:
        return None
    # split in the widest empty vertical band around the middle
    left_edge = max(b[2] for b in left)
    right_edge = min(b[0] for b in right)
    return (left_edge + right_edge) / 2 if left_edge < right_edge else mid


def page_text(page, layout: str = "sorted") -> str:
    """Page text in reading order. ``columns`` reads a two-column page column by column
    (rows inside each column stay intact) and falls back to ``sorted`` otherwise."""
    if layout == "columns":
        split = _column_split(page)
        if split is not None:
            import pymupdf

            r = page.rect
            left = page.get_text("text", clip=pymupdf.Rect(r.x0, r.y0, split, r.y1), sort=True)
            right = page.get_text("text", clip=pymupdf.Rect(split, r.y0, r.x1, r.y1), sort=True)
            return f"{left.rstrip()}\n\n{right.lstrip()}"
    return page.get_text("text", sort=True)


def column_texts(path: Path) -> list[str]:
    """Text of every page with ``layout="columns"`` (text only: fast, no tables)."""
    import pymupdf

    with pymupdf.open(path) as doc:
        return [page_text(page, "columns") for page in doc]


def _extract_range(
    path: str, start: int, end: int, with_tables: bool
) -> list[tuple[int, str, list[str]]]:
    """Pages [start, end) of one PDF. Runs in a worker process (see extract_pages)."""
    import pymupdf

    out = []
    with pymupdf.open(path) as doc:
        for index in range(start, min(end, len(doc))):
            page = doc[index]
            text = page_text(page, "sorted")
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


def load_pages(
    path: Path, cache_dir: Path | None = None, layout: str = "sorted"
) -> list[tuple[int, str, list[str]]]:
    """``extract_pages`` with an on-disk JSON cache.

    Table detection on a 300-page report takes minutes; experiments re-chunk the same
    PDFs many times, so parsed pages are cached, keyed on file size + content hash
    (not mtime, so a cache built on one machine is valid on another). Other text layouts
    reuse the cached tables and only re-read the text (seconds).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if cache_dir is None:
        pages = extract_pages(path)
        if layout != "sorted":
            texts = column_texts(path)
            pages = [(n, texts[n - 1], tabs) for n, _, tabs in pages]
        return pages
    key = file_key(path)
    cache_file = cache_dir / f"{path.stem}.json"
    cached = None
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("key") != key:
            cached = None
    if cached is None:
        cached = {
            "key": key,
            "pages": [{"page": n, "text": t, "tables": tabs} for n, t, tabs in extract_pages(path)],
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    pages = [(p["page"], p["text"], p["tables"]) for p in cached["pages"]]
    if layout != "sorted":
        field = f"text_{layout}"
        if field not in cached:
            cached[field] = column_texts(path)
            cache_file.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        texts = cached[field]
        pages = [(n, texts[n - 1], tabs) for n, _, tabs in pages]
    return pages


def page_headings(texts: list[str], max_len: int = 120) -> list[str]:
    """A short heading per page: its first meaningful line(s).

    Lines repeated on many pages (running headers such as the company name and report
    title) carry no information and are skipped. A table row like
    "Κύκλος εργασιών 5.1 1.232.904.830" says nothing about *which* statement it belongs
    to; the page heading ("ΚΑΤΑΣΤΑΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ") does.
    """

    def lines(text: str) -> list[str]:
        return [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    counts: dict[str, int] = {}
    for text in texts:
        for line in set(lines(text)[:6]):
            counts[line] = counts.get(line, 0) + 1
    common = {line for line, n in counts.items() if n >= max(3, len(texts) * 0.2)}
    headings = []
    for text in texts:
        picked: list[str] = []
        for line in lines(text)[:8]:
            if line in common or sum(ch.isalpha() for ch in line) < 4:
                continue  # running header, page number, or a row of figures
            picked.append(line[:max_len])
            if len(picked) == 2:
                break
        headings.append(" · ".join(picked))
    return headings


def chunks_for_document(
    path: Path,
    max_chars: int = 1200,
    overlap_chars: int = 200,
    with_tables: bool = True,
    cache_dir: Path | None = None,
    layout: str = "columns",
    headings: bool = False,
) -> list[Chunk]:
    """Chunks of one report. ``layout``: see ``page_text``. ``headings``: prefix every
    chunk with its page heading (as ``[heading]``) so retrieval and the LLM know which
    section a fragment of text or a table row comes from."""
    company, year = parse_doc_name(path)
    doc_id = f"{company}_{year}"
    chunks: list[Chunk] = []
    if cache_dir is not None:
        pages = load_pages(path, cache_dir, layout)
    else:
        pages = extract_pages(path, with_tables=with_tables)
        if layout != "sorted":
            texts = column_texts(path)
            pages = [(n, texts[n - 1], tabs) for n, _, tabs in pages]
    heads = page_headings([t for _, t, _ in pages]) if headings else [""] * len(pages)
    for (page_no, text, tables), head in zip(pages, heads, strict=True):
        prefix = f"[{head}]\n" if head else ""
        if not with_tables:
            tables = []
        for i, piece in enumerate(chunk_page(text, max_chars, overlap_chars)):
            chunks.append(
                Chunk(f"{doc_id}:p{page_no}:t{i}", doc_id, company, year, page_no, prefix + piece)
            )
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
                    prefix + table[: max_chars * 3],
                    kind="table",
                )
            )
    return chunks


def iter_pdfs(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())
