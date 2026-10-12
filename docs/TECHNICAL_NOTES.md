# Technical notes

Design decisions, the evidence behind them, and the limits. Numbers are from this repository's
own runs: `eval/chunking_results.md`, `eval/live_results.json`, `docs/RESULTS.md`.

## 1. Architecture

```
                    BUILD (once, ~30 s after first-run OCR)
 data/raw/*.pdf --> ingest.py --> chunking.py --> embed_store.py --> chroma_db/
   3 PDFs           cleaned        284 chunks       384-d unit        persistent,
   114 pages        pages          + metadata       vectors           cosine HNSW

                    ASK (per question)
 question --> retrieve.py --> generate.py --> answer
              embed query     guard: best score < 0.50 ? --> "insufficient evidence" (no LLM call)
              top-5 chunks    prompt: numbered passages + rules --> Gemini (retry / fallback / cache)
              score+rank      parse + validate citations
```

| Module | Responsibility | Key decision |
|---|---|---|
| `ingest.py` | PDF to clean, page-attributed lines | geometry-based cleaning; OCR only the unreadable rows |
| `chunking.py` | pages to chunks with `source`, `page_start/end`, `section` | chunk across page breaks; two strategies |
| `embed_store.py` | text to vectors to Chroma | unit-length vectors; cosine; vectors always supplied by us |
| `retrieve.py` | query to ranked chunks | score = `1 - cosine distance`; ranks 1..k |
| `generate.py` | grounded answer | three independent anti-fabrication layers |
| `pipeline.py` | one object for build + ask | index stamped with its config; auto-rebuild on mismatch |
| `config.py` | all tunables | every value has its evidence beside it |

## 2. Ingestion: the PDFs are not what they look like

All three PDFs are text-selectable (no scans), 114 pages in total, from two toolchains
(Apache FOP for the two Acts, Adobe PDF Library for the Regulations). "Text-selectable" is not
"clean": a plain `page.get_text()` produces text that would quietly damage retrieval. What was
found by inspecting the files, and what was done:

| Finding | Evidence | Treatment | Residual |
|---|---|---|---|
| **19 vector paths (18 text rows, 10 pages) in the Regulations are drawn as outlines, not text** | three independent extractors (pypdf, pdfminer, PyMuPDF) dropped the same words; the page's content stream holds `m/l/c/h` path operators; rendering shows them. Mostly defined terms, e.g. "transferring entity" | detect filled Bezier paths; OCR just those rows (RapidOCR, 300/200/150 dpi variants **voted** on, scored against the document's own vocabulary); keep all native text | one cosmetic quote artefact class; ~2.5 min one-off, cached |
| Page-level OCR first tried, then **rejected** | it garbled ordinary lines that native extraction had right (`sus ys ugs rss g u ynsg`) | replaced by row-level repair above | none |
| Running headers and page numbers inside the text | `Kenya Subsidiary Legislation, 2021 1999`; header at y=42 pt in the Acts | lines in the top 15% / bottom 8% that repeat on >=25% of pages, or are bare numbers | a unique line in those bands would be kept |
| Contents pages with dot leaders; Regulations contents without them | 79 + 90 dot-leader lines; 76 contents rows | dropped (they would out-rank real text for title-like queries) | the Part headings remain |
| Cover and licence boilerplate | "Creative Commons ...", "Share widely and freely" | page dropped | |
| Section number and heading are separate text runs | `8.` at x=80, `Functions of the Office` at x=98 | merge runs sharing a row (4.5 pt tolerance) | |
| Section titles live in a right-hand **margin column** (Regulations) | notes at x~418, body ends at x~406 | body edge computed per page; each note becomes a heading line placed before the regulation it labels; cross-references like "No. 24 of 2019." dropped | 8 notes dropped |
| Ligatures (`Ofﬁce`) | U+FB01 in extracted text | NFKC normalisation | |
| Hyphen at line end (`sub-` / `regulation`) | seen in Regulations | re-joined when the continuation is lowercase | |
| File ends with the *next* instrument's contents list | `LEGAL NOTICE NO. 264` on the last page | truncated at that marker | |

`python -m src.ingest` prints this report per document. Counts removed: header/footer lines
98 / 83 / 88 (CMCA / DPA / Regulations); dot-leader lines 79 / 90 / 20; contents rows 1 / 1 / 76.

**Page numbers.** `page` is the 1-based **PDF page index** (what a viewer's page box shows).
The printed numbers differ (the Regulations print 1979 on PDF page 3; the Acts print "27" on
PDF page 32). Using the PDF index makes every citation checkable by opening the file and typing
the number.

## 3. Chunking: decisions with evidence

### Why chunk at all
An embedding turns one input into one vector. A whole Act becomes a blur of everything; a
sentence loses its context. Chunk size is the knob between those. The embedding model
(`all-MiniLM-L6-v2`) also has a **hard 256 word-piece input limit and truncates silently**, so
text beyond it cannot influence retrieval at all.

### Experiment
28 configurations: fixed windows of 300 / 600 / 800 / 1000 / 1200 / 1500 characters x overlap
0 / 10 / 20 / 30 %, plus the structure-aware chunker with 4 maximum sizes. For each, the whole
corpus is re-chunked, embedded, indexed and queried with the 22 answerable questions.

Metrics: **hit@k** (verbatim anchor phrase from the right PDF in the top k), **MRR**, and two
diagnostics that explain the numbers: **coverage** (is the anchor intact inside *any* chunk?)
and **% of chunks over 256 tokens**.

| config | chunks | median tokens | >256 tok | coverage | hit@1 | hit@3 | hit@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 300 / 0 | 764 | 61 | 0% | 0.73 | 0.18 | 0.32 | 0.45 | 0.282 |
| fixed 300 / 60 | 947 | 62 | 0% | 1.00 | 0.27 | 0.41 | 0.45 | 0.384 |
| fixed 600 / 0 | 382 | 121 | 0% | 0.73 | 0.36 | 0.41 | 0.50 | 0.418 |
| fixed 600 / 60 | 422 | 121 | 0% | 0.95 | 0.41 | 0.64 | 0.68 | 0.514 |
| fixed 800 / 240 | 405 | 161 | 0% | 1.00 | 0.59 | 0.68 | 0.73 | 0.646 |
| fixed 1000 / 0 | 229 | 201 | 0% | 0.95 | 0.36 | 0.59 | 0.73 | 0.502 |
| fixed 1000 / 100 | 254 | 201 | 0% | 1.00 | 0.41 | 0.73 | 0.73 | 0.588 |
| **fixed 1000 / 200** | **284** | **202** | **0%** | **1.00** | **0.59** | **0.82** | **0.91** | **0.724** |
| fixed 1000 / 300 | 325 | 201 | 0% | 1.00 | 0.45 | 0.73 | 0.86 | 0.605 |
| fixed 1200 / 120 | 211 | 242 | 18% | 1.00 | 0.50 | 0.73 | 0.95 | 0.658 |
| fixed 1500 / 300 | 190 | 301 | **99%** | 1.00 | 0.55 | 0.77 | 0.86 | 0.672 |
| structure, max 800 | 416 | 127 | 0% | 1.00 | 0.50 | 0.77 | 0.86 | 0.646 |
| structure, max 1200 | 302 | 162 | 3% | 1.00 | 0.50 | 0.77 | 0.86 | 0.653 |
| structure, max 2000 | 237 | 180 | 30% | 1.00 | 0.50 | 0.73 | 0.86 | 0.635 |

(All 28 rows: `eval/chunking_results.md`.)

### What the numbers say
1. **Overlap 0 loses answers.** At 1000 characters, 5% of anchors straddle a boundary and exist
   in no chunk (coverage 0.95); at 300/600 characters it is 27%. With 10-20% overlap coverage
   is 100%. Overlap is also statistically real: 1000/200 vs 1000/0 is **+0.222 MRR, 95% paired
   bootstrap interval [+0.073, +0.376]**.
2. **Size has an upper bound set by the model, not by taste.** 1000 characters is about 200
   tokens (median) and 0% of chunks are truncated. At 1200 characters 15-19% are truncated;
   at 1500 characters 98-99% are, so the "1500" rows are really measuring about the first 1000
   characters and then paying for 50% more chunk text in the prompt.
3. **Small is worse.** 300-character chunks (about 60 tokens) reach MRR 0.28-0.40: a clause
   without its neighbours rarely matches a question.
4. **The top results are not distinguishable from each other.** 1000/200 beats the best
   structure-aware config by +0.071 MRR with a 95% interval of [-0.054, +0.198], and the
   non-monotone wiggles (1000/300 below 1000/200) show the grid is noisy. One question is worth
   4.5 percentage points. Do not read the third decimal.

### Decision: fixed, 1000 characters, 200 overlap (20%)
Highest point estimate on every metric, **and** principled on its own: the largest size that
fits the model's input, with enough overlap for full coverage, and the cheapest configuration
among those. The structure-aware chunker (`CHUNK_STRATEGY = "structure"`) is kept as a
first-class alternative: it keeps every clause whole and labels chunks with their section,
at statistically indistinguishable retrieval quality.

### What the retrieval metric does not capture
The hit test asks whether the answer phrase is *present* in a retrieved chunk. It cannot see
**scope words cut off by a boundary**. Example (`cmca-05`): the cited fixed-size chunk started
mid-sentence, so the model read "...sections 16 and 17 on a protected computer system" instead
of "sections 14, 15, 16 and 17", and answered for two of the four sections (adding that it had
no evidence for sections 14 and 15). Right figure, incomplete scope. Overlap reduces but cannot
remove this. With whole-section chunks the model saw the full list and still wrote
"sections 14, 16, or 17": better, but it dropped 15, which is plain LLM sloppiness that no
chunking choice fixes (section 9, residual hallucination).

### End-to-end check of the choice
Retrieval metrics are a proxy, so `python -m eval.strategy_end_to_end` answers all 22 questions
with the full pipeline under both strategies and grades the answers with the same fact patterns
as the live tests (`eval/strategy_end_to_end.json`):

| strategy | correct | declined |
|---|---:|---:|
| fixed 1000 / 200 | **20 / 22** | 2 (`dpa-07`, `reg-05`) |
| structure, max 1200 | 19 / 22 | 3 (the same two, plus `dpa-04`) |

The only difference is `dpa-04`, the embedding effect described in section 4: the exact
section-73 chunk is correct but ranks below other "penalty" passages, so the model is never
shown it. The retrieval-level choice is confirmed, within the noise of 22 questions.

## 4. Embeddings

`all-MiniLM-L6-v2` (sentence-transformers), run locally: 384 dimensions, 256 word-piece input,
output L2-normalised, so cosine similarity is a plain dot product.

An embedding is a fixed-length list of numbers produced by a neural network such that texts
that mean similar things land close together. `python -m src.embed_store --explain` prints a
worked example:

| pair | cosine |
|---|---:|
| "The data controller must notify the Data Commissioner when personal data is breached." vs "Organisations have to tell the regulator if people's personal information is leaked." | **0.443** (same meaning, almost no shared words) |
| the first sentence vs "Elephants migrate across the Serengeti plains every year." | **-0.028** (unrelated) |

Honest properties of this model: it is small and fast but **lexically noisy**. The question
"general penalty for an offence with no specific penalty" scored the exact section 73 chunk at
0.585, below four other penalty/liability passages at 0.63-0.68, so with section-sized chunks
the right passage fell out of the top 10 (fixed windows happened to bundle it with
"offence" language from section 72 and ranked it first). That is an embedding effect, not a
chunking bug.

## 5. Vector storage (ChromaDB)

* Persistent local client in `chroma_db/` (git-ignored; rebuilt from the PDFs).
* Collection created with HNSW `space = cosine`. Chroma returns a **distance**
  `1 - cosine_similarity` (0 identical, 1 unrelated, 2 opposite). The pipeline reports
  `score = 1 - distance`. This mapping is verified numerically in `--explain` and in
  `tests/test_retrieval.py` (against numpy cosine).
* `embedding_function=None`: vectors are always supplied by this code, so Chroma can never
  silently embed a query with a different model than the chunks.
* Each chunk stores `source`, `page_start`, `page_end`, `chunk_index`, `char_start/end`,
  `strategy`, `section`. The collection itself stores the chunking/embedding config it was
  built with, and `pipeline.ensure_index()` rebuilds on mismatch, so an index cannot silently
  go stale after a config change.
* HNSW is approximate in principle; with 284 vectors it is effectively exact.

## 6. Similarity search and ranking

The query is embedded with the same model; Chroma returns the nearest chunks by cosine
distance; they are ranked by similarity, rank 1 first. `k = 5` is fixed.

**What k trades off**, with this corpus's numbers (fixed 1000/200): hit@1 0.59, hit@3 0.82,
**hit@5 0.91**, hit@10 0.95. Small k is cheap and precise but loses answers that sit at rank
k+1 (nothing downstream can recover them). Large k raises recall but fills the prompt with
distractors (the model has to pick the right passage out of five; for `reg-02` the right one
was rank 2 behind a different regulation with the same vocabulary) and costs more tokens.
Overlap makes this worse in one specific way: neighbouring chunks share 200 characters, so the
top-5 can contain near-duplicates and fewer *distinct* passages.

## 7. Generation and fabrication control

Three independent layers (each can fail without the others failing):

1. **Prompt rule.** Answer only from numbered passages; cite them as `[n]`; quote numbers
   exactly; otherwise reply `INSUFFICIENT EVIDENCE: ...`. Temperature 0.
2. **Similarity guard.** If the best chunk scores below `MIN_SIMILARITY = 0.50` the LLM is not
   called. Calibrated (`python -m eval.calibrate_threshold`): answerable questions scored
   0.570-0.791 (22 of 22 pass at 0.50, margin 0.07); clearly off-topic queries scored below
   0.50 (7 of 12 calibration negatives blocked). **Trade-off:** raising the threshold blocks more
   junk but starts refusing valid, unusually-phrased questions; lowering it hands more
   off-topic queries to the model. Kept conservative because a false refusal is a visible
   failure and a false pass is still caught by layer 1.
3. **Citation check.** Cited passage numbers are parsed and validated against what was
   supplied; invalid citations are reported (`Answer.invalid_citations`).

**The guard cannot separate near-domain negatives.** "Penalty for a data breach under India's
DPDP Act?" scored 0.609 and the CCPA question 0.583, *higher than a valid Kenyan question*
(0.570). For those, only layer 1 stands, and it held in the live tests (the GDPR question was
refused by the model). That is the honest ceiling of a score threshold.

**Reliability engineering** (the free tier is not reliable, and this is visible in the data):
`gemini-3.8-flash`, the newest free-tier model, was returning `504 DEADLINE_EXCEEDED` and
`503 high demand` for most of the run, and `429 RESOURCE_EXHAUSTED` appeared under load.
The generator therefore has: bounded exponential backoff with jitter; 429 waits as long as the
server's retry hint; a per-model circuit breaker (a model that exhausted its retries is skipped
for 5 minutes instead of costing ~90 s per question); fallback through
`gemini-3.6-flash` and `gemini-3.1-flash-lite`; an on-disk response cache keyed on
(models, prompt); and a client held on `self`, because a garbage-collected SDK client fails
mid-request with "client has been closed". All of it is unit-tested with a scripted fake
client. The answering model is recorded on every answer.

## 8. Evaluation

* **Question set** (`eval/questions.json`): 22 answerable questions across the three documents
  (timelines, penalties, definitions, lists, a cross-page clause, an OCR-repaired definition)
  and 4 unanswerable. Each answerable one carries source, section, pages, a verbatim anchor
  and a verbatim evidence passage. `python -m eval.eval_set` re-checks every one against the
  ingested text (anchor unique, on the stated page, evidence verbatim): an expected answer that
  was not read from the PDF cannot pass.
* **Retrieval:** hit rate and MRR as above. **End to end** (`pytest --live`, evidence in
  `docs/RESULTS.md`): RAG correct on 20/22 vs base Gemini 6/22; 15 questions RAG-only, 1
  base-only (`dpa-07`, which base answered by citing the GDPR), 1 both wrong (`reg-05`, where
  RAG declined and base fabricated); 6/6 unanswerable questions refused; 20/22 passages
  retrieved in the top 5; every figure in every answer found in the retrieved text.
* **What base Gemini gets wrong** is mostly jurisdiction: it answers for the GDPR, the UK, the
  US. Asked how long a controller has to honour an access request it says "one month"; the
  Kenyan Regulations say seven days.

## 9. Limitations (honest)

**Source documents**
* *Parsing is a pipeline, not a solved problem.* 19 outlined paths needed OCR; the repaired
  text is checked only against the document's own vocabulary, so a plausible mis-OCR of a rare
  word would pass. Straight vs curly quote marks in repaired rows are inconsistent. Unusual
  layouts in other PDFs would need new rules (the cleaning is tuned to these three files).
* *Versions.* The Computer Misuse Act is consolidated to 4 Nov 2025 (it carries "Act No. 17 of
  2025" amendment notes); the Data Protection Act is as at 31 Dec 2022 and does not reflect
  anything later. The corpus is a snapshot, and the system has no notion of a provision being
  amended, repealed or suspended by a court.

**Retrieval**
* *Dense-only, fixed top-k, no re-ranker.* Two of 22 questions are never in the top 5:
  `dpa-07` (a paraphrase gap: "who has to prove ... agreed" vs "burden of proof ... consent")
  and `reg-05` (a definition inside an interpretation regulation, ranked 7th). A lexical or
  cross-encoder re-ranker is the standard fix and was deliberately not added here.
* *Fixed k = 5* serves a question that needs six scattered provisions badly and an easy one
  wastefully.
* *Legal cross-references are not followed.* "Pursuant to section 32(4) of the Act" and
  "regulation 41" point at text that may not be in the top 5; the system will not fetch it.
  Answers that depend on reading two instruments together (an Act section and the regulation
  that implements it) rely on both happening to rank.
* *Chunk boundaries can remove scope words* (the `cmca-05` example), which no hit metric here
  detects. Structure-aware chunks keep clauses whole but are not a cure (end-to-end 19/22 vs
  20/22 for fixed windows).

**Generation**
* *Residual hallucination risk.* The model can cite a correct passage and still summarise it
  imperfectly (incomplete scope, a dropped condition, a merged list). The checks catch invented
  numbers and invalid citations, not subtle misreadings; the grounded-facts test checks the
  facts a grader pattern recognises, not every clause.
* *Model dependence.* Answers in this repository came from fallback models; a different model
  may phrase, cite and refuse differently. Temperature 0 does not guarantee bit-identical
  hosted output, so tests assert patterns across questions, not exact wording.
* *Free tier.* Latency is dominated by overload retries; quotas can stop a bulk run.

**Evaluation**
* 22 questions: one is 4.5 percentage points, so most comparisons between top configurations
  are not significant (the bootstrap says so). The author wrote the questions and the chunker,
  and the threshold was calibrated on 12 negatives: indicative, not definitive.
* The "foreign law" count in `docs/RESULTS.md` is a keyword heuristic.

**Not legal advice.** The output is a pointer to a passage. Read the cited text.

## 10. What I would do next

1. A re-ranker or hybrid (BM25 + dense) to recover `dpa-07`/`reg-05`-type misses.
2. Retrieve by *section* then expand to neighbouring and cross-referenced provisions.
3. Parent-document retrieval: embed small chunks, send the whole section to the model.
4. A larger, independently written question set with held-out questions for threshold tuning.
5. Version-aware corpus (amendment dates) and an explicit "as at" in every answer.
