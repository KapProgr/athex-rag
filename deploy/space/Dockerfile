# Hugging Face Space (Docker SDK). Built by scripts/deploy_space.py; keyword-only (BM25),
# so no PyTorch or model download: small image, fast cold start.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user . /app
# runtime deps only: the demo never parses PDFs, so PyMuPDF (and its heavy deps) is skipped
RUN pip install --no-cache-dir numpy fastapi uvicorn httpx && pip install --no-cache-dir --no-deps .

USER user
ENV ATHEX_INDEX_DIR=/app/index \
    ATHEX_RETRIEVAL=bm25 \
    ATHEX_COMPANIES=/app/data/companies.json \
    ATHEX_SOURCES=/app/data/sources.json \
    ATHEX_GLOSSARY=/app/data/glossary.json \
    ATHEX_PDF_DIR=/app/no-pdfs \
    ATHEX_WARMUP=1
EXPOSE 7860
CMD ["uvicorn", "athex_rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
