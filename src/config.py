"""Single place for every tunable value in the pipeline.

Other modules import named constants from here instead of hard-coding numbers,
so a chunking or model change is a one-line edit that shows up in git history.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
# These two folders are listed in .gitignore: they are rebuilt, never committed.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LLM_CACHE_DIR = PROJECT_ROOT / ".llm_cache"
EVAL_DIR = PROJECT_ROOT / "eval"

# Human-readable names used in prompts, citations and demo output.
DOCUMENT_TITLES = {
    "Computer Misuse and Cybercrimes.pdf": "Computer Misuse and Cybercrimes Act (Cap. 79C)",
    "Data Protection Act.pdf": "Data Protection Act, 2019 (Cap. 411C)",
    "THE-DATA-PROTECTION-GENERAL-REGULATIONS-2021-1.pdf": "Data Protection (General) Regulations, 2021",
}

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, 384-dim, max 256 word-pieces per input
COLLECTION_NAME = "kenya_law"

# --- Chunking: chosen by experiment, see eval/chunking_results.md -------------
# fixed 1000/200 had the best MRR (0.724) and hit@5 (0.91) of 28 configurations. 1000 chars is
# the largest size fully inside the embedding model's 256-token input (0% truncated; 1200+
# truncates 15-99%); 20% overlap restores 100% answer coverage (0 overlap: 95%) and is clearly
# better than none (+0.22 MRR, 95% CI [+0.07, +0.38]). "structure" is a statistically
# indistinguishable alternative (MRR 0.653) that adds section labels.
CHUNK_STRATEGY = "fixed"  # "fixed" | "structure"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
STRUCTURE_MAX_SIZE = 1200  # characters; best structure variant (MRR 0.653, 3% truncated)

# --- Retrieval ---------------------------------------------------------------
TOP_K = 5
# Guard: if the best chunk scores below this cosine similarity the LLM is not called and the
# answer is "insufficient evidence". Calibrated by `python -m eval.calibrate_threshold`:
# answerable questions score >= 0.570 (22 of 22 pass at 0.50, margin 0.07); clearly off-topic
# queries score < 0.50 (7 of 12 calibration negatives blocked). Near-domain negatives (another
# country's privacy law) score up to 0.61, ABOVE some valid questions, so no threshold can
# catch them: the prompt rule is the defence for those. Kept low to avoid false refusals.
MIN_SIMILARITY = 0.50

# --- Generation (Gemini) -----------------------------------------------------
# Best free-tier model per https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-12).
GEMINI_MODEL = "gemini-3.8-flash"
# Newest free-tier models are sometimes overloaded (HTTP 503/504, observed 2026-10-12).
# generate.py tries these in order when the primary keeps failing.
GEMINI_FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.1-flash-lite")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


def get_gemini_api_key() -> str:
    """Load the key from `.env` / the environment. Never prints or logs the value."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{GEMINI_API_KEY_ENV} is not set. Copy .env.example to .env and add your key."
        )
    return key
