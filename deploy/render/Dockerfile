# Render (free web service). Build context: repository root.
# Keyword-only demo (BM25): no PyTorch, no PDFs; "Open PDF" links to the official reports.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
COPY --chown=user data/companies.json data/sources.json ./data/
COPY --chown=user deploy/demo-index ./index
# runtime deps only: the demo never parses PDFs, so PyMuPDF (and its heavy deps) is skipped
RUN pip install --no-cache-dir numpy fastapi uvicorn httpx && pip install --no-cache-dir --no-deps .

USER user
ENV ATHEX_INDEX_DIR=/app/index \
    ATHEX_RETRIEVAL=bm25 \
    ATHEX_COMPANIES=/app/data/companies.json \
    ATHEX_SOURCES=/app/data/sources.json \
    ATHEX_PDF_DIR=/app/no-pdfs \
    ATHEX_WARMUP=1
# Render sets $PORT
CMD ["sh", "-c", "uvicorn athex_rag.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
