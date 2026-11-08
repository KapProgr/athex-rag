FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[embeddings]"

# Mount your built index at /app/index. Ollama runs on the host:
#   docker run -p 8000:8000 -v ./index:/app/index \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 athex-rag
ENV ATHEX_INDEX_DIR=/app/index
EXPOSE 8000
CMD ["uvicorn", "athex_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
