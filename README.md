# athex-rag

**Ask questions about Greek listed companies' annual reports and get answers with page-level citations, measured against a hand-checked evaluation set.**

Annual financial reports of Athens Stock Exchange (ATHEX) companies are 150–400 page Greek PDFs full of tables. `athex-rag` parses them (including tables), retrieves the relevant passages with hybrid search, and has a local LLM answer *only* from those passages, citing `[company, year, page]` for every claim, or refusing when the answer is not there.

The point of the project is the **evaluation**: every design choice (chunk size, keyword vs dense vs hybrid retrieval, table extraction, embedding model, reranking, LLM) is compared on the same questions with known answers and known source pages, and failures are broken down by cause.

<!-- After running the UI on the real reports, save a screenshot as docs/screenshot.png and uncomment:
![Web UI](docs/screenshot.png)
-->


## Results

### Keyword baseline (BM25)

Retrieval only, measured on the 93-question set: is a gold page among the top 5 chunks?

| BM25 configuration | hit@5 | MRR |
|---|---|---|
| chunk 1200, no stemming | 63.6% | 0.498 |
| chunk 1200, Greek prefix stemming (6 chars) | 68.8% | 0.523 |
| chunk 600, stemming | 71.4% | 0.528 |

Light stemming fixes inflection mismatches (`ελεγκτική` vs `Ελεγκτές`). Most remaining misses are generic terms such as `κύκλος εργασιών`, which appear on dozens of pages of a 650-page report. Full breakdown: [reports/bm25_baseline.md](reports/bm25_baseline.md).

### Full pipeline (dense, hybrid, reranker, LLM)

<!-- RESULTS:START -->

_Run `.\scripts\run_all.ps1` (or `athex-rag experiments eval/questions.jsonl --update-readme`) to fill this section._

<!-- RESULTS:END -->

## How it works

```
PDF ──► PyMuPDF: page text + tables (cell-by-cell → markdown)       [cached per PDF]
     ──► page-bounded chunks (never cross a page → exact citations)
     ──► multilingual embeddings  +  BM25 on accent-folded, stemmed Greek tokens
     ──► Reciprocal Rank Fusion · company/year filter auto-detected from the question
     ──► optional cross-encoder reranking of the top 20
     ──► local LLM (Ollama) with numbered sources → answer with [n] citations, or refusal
```

Greek-specific details that matter:

- **Normalization**: accents and final sigma are folded (`ΧΡΉΣΗΣ` = `χρήσης`) for keyword search, filters and answer checking.
- **Stemming**: keyword search cuts words to their first 6 characters, so different inflections of a word match (+5 points hit@5 in the BM25 baseline).
- **Company names**: `data/companies.json` maps tickers to the inflected names people type (`της Εθνικής Τράπεζας` → `ETE`).
- **Numbers**: `1.234.567,89`, `(1.234)` for negatives and unit words (`χιλ. €`, `εκατ.`) are parsed, so `1.234,5 εκατ.` matches a gold value of `1234500000`.
- **Tables**: cells are read by clipping each cell box. PyMuPDF's default table extraction scrambled Greek-format numbers (`2.150,3` → `2 150 3 . ,`) in testing.
- **Robust parsing**: MuPDF segfaults on a few pages of these reports (3 of 4,100). Table detection runs in worker processes, and a crashing page keeps its text but loses its tables instead of killing the run.

## Data

10 annual financial reports of 5 ATHEX companies (ΟΤΕ, Jumbo, ΔΕΗ, Motor Oil, Εθνική Τράπεζα) for 2024 and 2025: about 4,100 Greek pages and 2,000+ tables. Sources and quirks are listed in [data/README.md](data/README.md).

## Evaluation

`eval/questions.jsonl` has **93 questions**: 63 numeric (28 answered from financial tables), 14 factual text questions and 16 unanswerable ones. Each line holds the question, the company and year of the report, the answer, and a verbatim **evidence** quote. `athex-rag check-eval eval/questions.jsonl --fix` checks every quote against the PDF text and sets the **gold pages** to every page where it appears. For unanswerable questions, it confirms that the key phrases appear nowhere in the report.

> The question set was drafted with the help of an AI assistant by reading the actual report pages. Every evidence quote is verified automatically against the PDF text with `check-eval`.

| type | what it tests |
|---|---|
| `numeric` (`source: text` / `table`) | figures from narrative text or from financial tables |
| `text` | names, counts, facts (auditor, board size, ...) |
| `unanswerable` | plausible questions the report does **not** answer: does the system refuse instead of inventing? |

Metrics: **hit@k** and **MRR** (is a gold page retrieved), **answer correctness** (numeric match at any unit scale / gold-token recall / refusal), **citation precision** (do cited sources sit on a gold page), and an **error breakdown** (retrieval miss, wrong answer, false refusal, hallucination). See `src/athex_rag/evaluation.py`.

`athex-rag experiments` runs the whole plan in stages, each building on the best configuration of the previous one: chunk size × retrieval mode → table extraction → embedding model → reranker → LLM. It writes `reports/experiments.md` and updates the Results section above.

## Quick start (Windows)

Requirements: Python 3.10+, [Ollama](https://ollama.com), and the annual report PDFs in `data/raw/` (see [data/README.md](data/README.md)).

```powershell
.\scripts\run_all.ps1            # install, pull models, run all experiments, open the web UI
.\scripts\run_all.ps1 -Quick     # 1 embedder, 1 LLM, no reranker
.\scripts\run_all.ps1 -ServeOnly # just the web UI, using the best configuration found
```

Then open <http://localhost:8000>. Clicking a citation highlights its source, and "Open PDF" jumps to the cited page.

### Using a hosted LLM (Groq) instead of Ollama

Without a GPU, a local 7B model takes about a minute per question. A free [Groq](https://console.groq.com) key is much faster:

```powershell
setx LLM_API_KEY "<your key>"        # once; then open a new PowerShell
.\scripts\run_all.ps1 -Quick -Llm groq:llama-3.3-70b-versatile
athex-rag ask "Ποιο μέρισμα προτείνει η ΔΕΗ για τη χρήση 2025;" --llm groq:llama-3.3-70b-versatile
```

The key is read from the environment and never stored in the repo. Rate-limit responses (HTTP 429) are retried automatically. Any OpenAI-compatible API works with `--llm openai:<model>@<base_url>`.

### Manual

```bash
pip install -e ".[embeddings,dev]"
ollama pull qwen2.5:7b

athex-rag ingest data/raw                                       # build ./index
athex-rag ask "Ποιος είναι ο ορκωτός ελεγκτής του ΟΤΕ για το 2025;"
athex-rag eval eval/questions.jsonl --no-llm                    # retrieval metrics, all modes
athex-rag experiments eval/questions.jsonl --update-readme      # the full plan
uvicorn athex_rag.api:app                                       # web UI + API (/docs)
```

API: `POST /ask {"question", "company"?, "year"?, "k"?}` → answer, refusal flag, sources with page and snippet. Also `GET /documents`, `GET /pdf/{COMPANY_YEAR}`, `GET /health`.

Docker: `docker build -t athex-rag .` (see the Dockerfile for the run command; Ollama stays on the host).

## Project layout

```
src/athex_rag/
  ingest.py       PDF → pages + tables → chunks (with page cache)
  text.py         Greek normalization, tokenization, number parsing
  index.py        chunks + vectors + BM25, saved as JSONL/NumPy
  retrieval.py    dense / bm25 / hybrid (RRF), filters, aliases
  rerank.py       optional cross-encoder
  llm.py          Ollama / OpenAI-compatible / fake
  rag.py          prompt, citations, refusal detection
  evaluation.py   metrics, error categories
  experiments.py  staged experiment runner + README update
  api.py, static/ FastAPI + web UI
eval/             questions
scripts/          run_all.ps1
tests/            pytest suite, no model downloads needed (hashing embedder, fake LLM)
```

## Development

```bash
pytest -q
ruff check . && ruff format --check .
```

CI runs both on Python 3.10 and 3.12.

## Limitations

- Scanned (image-only) PDFs need OCR, which is not included.
- Questions that compare years or companies need answers from several documents; the retriever filters to one company/year when the question names exactly one.
- The evaluation set is small (around 100 questions), so differences of a few points between configurations are within noise.

## License

MIT
