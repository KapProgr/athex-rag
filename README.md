# athex-rag

**Ask questions about Greek listed companies' annual reports and get answers with page-level citations, measured on a 93-question evaluation set with verified evidence.**

Annual financial reports of Athens Stock Exchange (ATHEX) companies are 200–670 page Greek PDFs full of tables. `athex-rag` parses them (including tables), retrieves the relevant passages with keyword, dense or hybrid search, and has an LLM (local via Ollama, or hosted via Groq) answer *only* from those passages, citing `[company, year, page]` for every claim, or refusing when the answer is not there.

The point of the project is the **evaluation**: every design choice (chunk size, keyword vs dense vs hybrid retrieval, table extraction, embedding model, reranking, LLM) is compared on the same questions with known answers and known source pages, and failures are broken down by cause.

<!-- After running the UI on the real reports, save a screenshot as docs/screenshot.png and uncomment:
![Web UI](docs/screenshot.png)
-->


## Results

93 questions over 10 reports (77 answerable, 16 unanswerable), top-5 retrieved chunks, answers generated with two LLMs on Groq. Full tables: [reports/answer_quality.md](reports/answer_quality.md).

### Answers

| | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| **Correct answers** (all 93) | **82.8%** | **83.9%** |
| numeric questions | 84.1% | 85.7% |
| numeric, answer in a table | 67.9% | 71.4% |
| text questions | 57.1% | 57.1% |
| **Unanswerable questions correctly refused** | **100%** | **100%** |
| answerable questions refused | 16.9% | 15.6% |
| cited sources that state the answer | 78.8% | 77.2% |
| latency per question | 11.1 s | 12.4 s |

**The system never invented an answer to an unanswerable question.** When it fails, it fails by saying "Δεν βρέθηκε η πληροφορία στα έγγραφα", which in a financial setting is the right way to fail.

### Retrieval (k = 5)

| | gold page in top 5 | a chunk stating the answer in top 5 | MRR |
|---|---|---|---|
| **BM25, Greek stemming, chunk 600** | **71.4%** | **80.5%** | 0.528 |
| hybrid (BM25 + e5-small, RRF), chunk 600 | 67.5% | 77.9% | 0.512 |
| dense (multilingual-e5-small), chunk 600 | 61.0% | 68.8% | 0.482 |
| BM25, chunk 1200 | 68.8% | 74.0% | 0.523 |
| BM25, chunk 600, without table chunks | 68.8% | 77.9% | 0.524 |
| BM25 without stemming, chunk 1200 | 63.6% | – | 0.498 |

The second column exists because the same figure often appears on several pages in different formats (`1.232.904.830` in the income statement, `1.232,90 εκατ.` in the management report). A chunk counts if it contains the exact answer text or a distinctive number (at least 4 significant digits, within 0.1%).

### What the evaluation shows

1. **Keywords beat the small multilingual embedding model on Greek financial text.** BM25 with light stemming outperforms both dense retrieval (+10.4 points hit@5) and the hybrid (+3.9). Questions ask for specific terms (`κύκλος εργασιών`, `CET1`, `ορκωτός ελεγκτής`), and exact matching on them is hard to beat. Larger embedding models (e5-base, bge-m3) and a cross-encoder reranker are wired into the experiment runner but were not run yet.
2. **Retrieval is the bottleneck, not the LLM.** About 75% of the answerable questions end with a correct answer and a citation that states it. Of the 15–16 wrong or refused answers, 12 are retrieval misses; only 1–2 are wrong readings of a retrieved source. The two LLMs score almost the same.
3. **Failures cluster in three places:**
   - *Auditor questions* (5 of 10 fail). The auditor's name sits on the signature page of the audit report, which shares almost no words with "ποια ελεγκτική εταιρεία έλεγξε…".
   - *Motor Oil's statements* (3 of 5 table questions fail). Its pages are two-column, and plain-text extraction interleaves the columns, so table rows are scrambled.
   - *Group vs company columns.* Statements put group and parent-company figures side by side; once, the model returned the company's 492,8 instead of the group's 478,8.
4. **Table chunks help a little** (+2.6 points hit@5 for BM25 at chunk 600).
5. **One question is ambiguous in the source itself.** PPC's 2024 report gives total investments as €3 δισ. on one page and €1.875 εκατ. on another. Both models answered the second. Real documents are not always consistent, and an evaluation set has to be read with that in mind.

### Latest automatic run

<!-- RESULTS:START -->

See [reports/experiments.md](reports/experiments.md). `athex-rag experiments ... --update-readme` writes its report here.

<!-- RESULTS:END -->

### Next improvements, in order of expected gain

Column-aware text extraction for two-column layouts; a cross-encoder reranker on the top 20; a "find the signature page" query rewrite for auditor questions; a larger multilingual embedding model.

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
# list the model ids your key can use (Groq retires models regularly):
(Invoke-RestMethod https://api.groq.com/openai/v1/models -Headers @{Authorization="Bearer $env:LLM_API_KEY"}).data.id
.\scripts\run_all.ps1 -Quick -Llm groq:<model-id>
```

The key is read from the environment and never stored in the repo. The LLM is checked with one request before the experiments start. Rate-limit responses (HTTP 429) are retried automatically. Any OpenAI-compatible API works with `--llm openai:<model>@<base_url>`.

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
