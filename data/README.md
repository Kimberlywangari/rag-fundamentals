# Source documents

Three Kenyan legal texts, kept exactly as obtained (no edits) in [`raw/`](raw/). Everything the
pipeline derives from them (cleaned text, chunks, vectors, OCR repairs) is regenerated and is
**not** committed: see `chroma_db/` and `.llm_cache/` in `.gitignore`.

| File | Instrument | Version in this file | Pages | Producer | Identifier printed in the file |
|---|---|---|---:|---|---|
| `Data Protection Act.pdf` | Data Protection Act, 2019 (Cap. 411C) | legislation as at 31 December 2022 | 32 | Apache FOP | `/akn/ke/act/2019/24/eng@2022-12-31` |
| `Computer Misuse and Cybercrimes.pdf` | Computer Misuse and Cybercrimes Act, 2018 (Cap. 79C) | legislation as at 4 November 2025 (includes Act No. 17 of 2025 amendments) | 37 | Apache FOP | `/akn/ke/act/2018/5/eng@2025-11-04` |
| `THE-DATA-PROTECTION-GENERAL-REGULATIONS-2021-1.pdf` | Data Protection (General) Regulations, 2021 | Kenya Gazette Supplement No. 236, 31 December 2021 (Legislative Supplement No. 106, Legal Notice No. 263) | 45 | Adobe PDF Library | none (a Gazette page) |

Why the version dates matter: the two Acts are *consolidated* texts at different dates, so this
system answers "what do these three documents say", not "what is the law today". A later
amendment, or a court order suspending a provision, is invisible to it.

## Licence and attribution

* The two Acts are published by **Kenya Law and Laws.Africa** (<https://kenyalaw.org>). Each PDF
  states, on its second page: *"There is no copyright on the legislative content of this
  document. This PDF copy is licensed under a Creative Commons Attribution NonCommercial
  ShareAlike 4.0 License (CC BY-NC-SA 4.0)."* Credit the creators, use them for non-commercial
  purposes only, and share adaptations on the same terms.
* The Regulations are a page of the Kenya Gazette (a government publication) and carry no
  licence statement in the file.
* This repository is an educational project. It is not legal advice and is not a substitute for
  the official texts.

## Checking the files

Page counts, text-selectability and extraction problems are reported by `python -m src.ingest`.
The 22 evaluation questions in `eval/questions.json` each name a page in these exact files, and
`python -m eval.eval_set` verifies that the quoted passage is really on that page.
Page numbers throughout the project are **PDF page indexes** (1-based, as a viewer shows them),
not the page numbers printed on the paper (the Regulations print 1979 on PDF page 3).
