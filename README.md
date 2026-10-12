# RAG Fundamentals: Kenyan data-protection and cybercrime law

A from-scratch Retrieval-Augmented Generation (RAG) pipeline that answers questions about three
Kenyan legal texts and **shows its evidence**: the ranked passages (with similarity score, source
and page) and an answer that cites them, or an explicit "insufficient evidence" when the
documents do not contain the answer.

| Document | File in `data/raw/` | Version |
|---|---|---|
| Computer Misuse and Cybercrimes Act (Cap. 79C) | `Computer Misuse and Cybercrimes.pdf` | as at 4 Nov 2025 |
| Data Protection Act, 2019 (Cap. 411C) | `Data Protection Act.pdf` | as at 31 Dec 2022 |
| Data Protection (General) Regulations, 2021 | `THE-DATA-PROTECTION-GENERAL-REGULATIONS-2021-1.pdf` | Kenya Gazette Supp. No. 236, Legal Notice 263 |

```
PDFs -> ingest -> chunk (1000 chars / 200 overlap) -> embed (MiniLM, local) -> ChromaDB (cosine)
question -> embed -> top-5 chunks -> grounded prompt -> Gemini -> answer + citations
                         \-> similarity guard: too far from every chunk => no LLM call, "insufficient evidence"
```

**Measured results** (details in [docs/RESULTS.md](docs/RESULTS.md)):

| | RAG | Base Gemini (no documents) |
|---|---|---|
| Correct on the 22 answerable questions | **20** | 6 |
| Unanswerable questions (tax, employment, GDPR, off-topic) refused | **6 of 6** | n/a (answers confidently) |
| Verbatim answer passage in the top 5 chunks | 20 of 22 (0.91) | n/a |

## 1. Setup (Windows PowerShell; macOS/Linux commands are the same apart from activation)

You need Python **3.13** (3.14 is too new for the `torch`/`onnxruntime` wheels), ~2 GB of disk
for PyTorch, and internet access for the first run.

```powershell
git clone <this-repo-url>
cd rag-fundamentals

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

Copy-Item .env.example .env           # macOS/Linux: cp .env.example .env
# then open .env and set GEMINI_API_KEY=<your key>
```

* **API key:** free from Google AI Studio, <https://aistudio.google.com/apikey>. It is read from
  `.env` (git-ignored) through `python-dotenv`; the code never prints or logs it.
* **PDFs:** already committed in `data/raw/`. To obtain fresh copies, the Acts are published by
  Kenya Law (<https://kenyalaw.org>; each PDF prints its own identifier, e.g.
  `/akn/ke/act/2019/24/eng@2022-12-31`) and the Regulations are Kenya Gazette Supplement No. 236
  of 31 December 2021, also available from the Office of the Data Protection Commissioner. Keep
  the three file names exactly as above.
* **Check the key works** (should print `key works`):

  ```powershell
  python -c "from google import genai; from src import config as c; cl = genai.Client(api_key=c.get_gemini_api_key()); print(cl.models.generate_content(model=c.GEMINI_MODEL, contents='Reply with exactly: key works').text)"
  ```

## 2. Run

```powershell
python -m src.ingest                  # extraction report: pages, noise removed, OCR repairs
python -m src.embed_store --build     # chunk + embed + store in ./chroma_db  (~30 s)
python demo.py "Within how many days must a data controller respond to a request to erase personal data?" --base
```

The demo prints, in order: the query; the top-5 chunks with **rank, cosine score, document, page
(and section headings found in the chunk)**; the RAG answer with the sources it cited; and with
`--base` the answer Gemini gives with no retrieval, for comparison.

```text
python demo.py "<question>" [-k 5] [--base] [--full] [--rebuild] [--no-cache]
```

Useful stages on their own:

| Command | Shows |
|---|---|
| `python -m src.chunking --section 26` | structure-aware vs fixed-size chunks for the same text |
| `python -m src.embed_store --explain` | what an embedding is; cosine worked by hand; Chroma `distance = 1 - cosine` |
| `python -m src.retrieve "<query>" -k 5` | retrieval only |
| `python -m eval.eval_set` | re-verifies every evaluation question against the PDFs |
| `python -m eval.chunking_experiment` | the chunk size/overlap experiment (28 configurations, ~6 min) |
| `python -m eval.strategy_end_to_end` | fixed vs structure-aware chunks, judged on final answers (needs the API key) |
| `python -m eval.calibrate_threshold` | score distribution behind the similarity guard |
| `python -m eval.make_results_md` | regenerates `docs/RESULTS.md` from the last live run |

First-run costs: the embedding model (~90 MB) downloads once from Hugging Face; ingestion OCRs
18 text rows that the Regulations PDF draws as vector outlines (about 2-3 minutes once, then
cached in `.llm_cache/ocr`).

## 3. Test

```powershell
pytest                # 62 offline tests, ~2 s, no network, no model download
pytest --live         # + 74 live tests: real embedding model and real Gemini
```

* **Offline** tests cover chunking invariants, Chroma retrieval (with a deterministic fake
  embedder), ingestion helpers, and the generation guard, retry/backoff, model fallback and cache
  (with a scripted fake client).
* **Live** tests (`--live`, skipped by default) check: (a) verifiable facts against the PDFs,
  (b) RAG vs base Gemini over *all* questions, (c) "insufficient evidence" for tax, employment,
  other-jurisdiction and off-topic questions, (d) the expected passage is retrieved,
  (e) every figure in an answer appears in the retrieved text. Two known retrieval misses are
  `xfail(strict=True)` so they are visible, not hidden.
* Gemini responses are cached in `.llm_cache/` so reruns are fast and repeatable. Delete the
  folder to force fresh calls. Free-tier models return `503`/`504`/`429` at times; the generator
  retries with backoff, honours retry hints, and falls back to the next model.

## 4. Configuration (`src/config.py`)

| Constant | Value | Why |
|---|---|---|
| `CHUNK_STRATEGY` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | `fixed` / 1000 / 200 | best MRR (0.724) of 28 configs; largest size fully inside the model's 256-token input; 20% overlap restores 100% answer coverage ([eval/chunking_results.md](eval/chunking_results.md)) |
| `STRUCTURE_MAX_SIZE` | 1200 | best structure-aware variant (set `CHUNK_STRATEGY = "structure"` to use it) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | local, 384 dimensions, 256-token input |
| `TOP_K` | 5 | hit@5 = 0.91 vs hit@3 = 0.82 |
| `MIN_SIMILARITY` | 0.50 | refuses clearly off-topic queries without an LLM call; 0 of 22 valid questions refused |
| `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS` | `gemini-3.8-flash`, then `3.6-flash`, `3.1-flash-lite` | newest free-tier model per Google's pricing page, with fallbacks for overload |

Changing a chunking constant is safe: the index stores the config it was built with and is
rebuilt automatically when it no longer matches.

## 5. Project layout

```
data/raw/          the three source PDFs
src/
  config.py        every tunable value, with the evidence for it
  ingest.py        PDF -> cleaned pages (headers, TOC, margin notes, OCR repair of outlined lines)
  chunking.py      fixed-size and structure-aware chunkers (with page + section metadata)
  embed_store.py   sentence-transformers embeddings -> persistent ChromaDB (cosine)
  retrieve.py      top-k retrieval: text, score, rank, source, page
  generate.py      grounded Gemini prompt, similarity guard, retry/backoff/fallback, cache
  pipeline.py      the whole flow behind one object
demo.py            command-line demo
tests/             offline tests + opt-in live tests
eval/              questions.json (verified), chunking experiment, calibration, results generator
docs/              RESULTS.md (test evidence), TECHNICAL_NOTES.md (design and limitations)
```

## 6. Know the limits

Dense-only retrieval with a fixed top-k, no re-ranker, residual hallucination risk, source PDF
defects and legal cross-references are discussed honestly in
[docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md). This is a learning project, **not legal
advice**: always read the cited passage.
