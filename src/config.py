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

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, 384-dim, max 256 word-pieces per input
COLLECTION_NAME = "kenya_law"

# --- Chunking (PROVISIONAL: replaced with evidence-based values in piece 10) --
CHUNK_STRATEGY = "fixed"  # "fixed" | "structure"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# --- Retrieval ---------------------------------------------------------------
TOP_K = 5

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
