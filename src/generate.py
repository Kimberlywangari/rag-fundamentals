"""Grounded answer generation with Gemini.

Three layers keep the model from making things up:

1. Prompt layer   - the model is told to answer ONLY from the numbered passages, to cite them,
                    and to reply "INSUFFICIENT EVIDENCE: ..." when they do not contain the answer.
2. Retrieval guard - if even the best retrieved chunk is below ``MIN_SIMILARITY`` the LLM is not
                    called at all (cheap, deterministic, and immune to prompt-following slips).
3. Verification   - citations are parsed and checked against the passages that were supplied,
                    so a made-up citation is detected and reported rather than trusted.

No layer is perfect, see docs/TECHNICAL_NOTES.md for the residual risk.

Reliability: free-tier models return 503 ("high demand") and 429 (rate limit) regularly, so
calls use bounded exponential backoff, then fall back to the next model in
``config.GEMINI_FALLBACK_MODELS``.  Every successful response is cached on disk (git-ignored),
which also makes tests and demos repeatable and keeps free-tier quota for new questions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src import config
from src.retrieve import RetrievedChunk

INSUFFICIENT_PREFIX = "INSUFFICIENT EVIDENCE"

SYSTEM_PROMPT = f"""You are a careful research assistant for Kenyan data-protection and \
cybercrime law. You answer questions using ONLY the numbered context passages supplied in the \
user message.

Rules:
1. Use only facts explicitly stated in the passages. Do not use outside knowledge, even if you \
believe you know the law, and do not guess.
2. Cite the passage number(s) that support each statement in square brackets, e.g. [1] or [2][3].
3. Quote numbers, time limits and amounts exactly as the passages state them.
4. If the passages do not contain enough information to answer the question, reply with exactly \
one line in this form and nothing else:
{INSUFFICIENT_PREFIX}: <one short sentence saying what is missing>
5. Keep the answer short and direct."""

BASE_PROMPT = "Answer the question concisely and factually.\n\nQuestion: {question}"

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    text: str
    sufficient: bool  # False => "insufficient evidence" (from the model or from the guard)
    model: str = ""  # model that actually produced the text ("" if no LLM call was made)
    cited: list[int] = field(default_factory=list)  # passage numbers cited, e.g. [1, 3]
    invalid_citations: list[int] = field(default_factory=list)  # cited numbers that do not exist
    guard_triggered: bool = False  # True if the similarity guard blocked the LLM call
    from_cache: bool = False
    attempts: int = 0  # API calls made (0 for cache hits / guard)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered passages with enough provenance for the model (and the reader) to cite."""
    blocks = []
    for c in chunks:
        title = config.DOCUMENT_TITLES.get(c.source, c.source.removesuffix(".pdf"))
        label = title + (f", {c.section}" if c.section else "")
        text = " ".join(c.text.split())
        blocks.append(f"[{c.rank}] ({label}, p.{c.pages})\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"CONTEXT PASSAGES:\n\n{format_context(chunks)}\n\nQUESTION: {question}"


def parse_citations(text: str, n_passages: int) -> tuple[list[int], list[int]]:
    """Return (valid cited numbers, invalid cited numbers), each sorted and de-duplicated."""
    nums = {int(n) for grp in _CITATION.findall(text) for n in re.split(r"\s*,\s*", grp)}
    return sorted(n for n in nums if 1 <= n <= n_passages), sorted(
        n for n in nums if not 1 <= n <= n_passages
    )


def is_insufficient(text: str) -> bool:
    return text.strip().upper().startswith(INSUFFICIENT_PREFIX)


class GenerationError(RuntimeError):
    """All models and retries were exhausted."""


class Generator:
    """Calls Gemini with retry/backoff, model fallback and a local response cache."""

    def __init__(
        self,
        client=None,
        models: tuple[str, ...] | None = None,
        cache_dir: Path | None = None,
        max_attempts: int = 3,
        base_delay: float = 2.0,
        request_timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = client
        self.models = models or (config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS)
        self.cache_dir = Path(cache_dir) if cache_dir else config.LLM_CACHE_DIR / "gemini"
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.request_timeout_s = request_timeout_s
        self._sleep = sleep

    # -- client / API -------------------------------------------------------------
    @property
    def client(self):
        """Created lazily and kept on ``self``: a client that is garbage-collected mid-request
        raises "Cannot send a request, as the client has been closed"."""
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=config.get_gemini_api_key(),
                http_options=types.HttpOptions(
                    timeout=int(self.request_timeout_s * 1000),
                    # we do our own, bounded retry; the SDK default can hang for minutes
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    @staticmethod
    def _status(exc: Exception) -> int | None:
        return getattr(exc, "code", None) or getattr(exc, "status_code", None)

    def _call_with_retry(self, model: str, system: str | None, user: str) -> tuple[str, int]:
        """One model, bounded exponential backoff on 429/5xx/timeouts. Returns (text, attempts)."""
        from google.genai import types

        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0.0)
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.client.models.generate_content(model=model, contents=user, config=cfg)
                text = (resp.text or "").strip()
                if not text:
                    raise GenerationError("empty response (possibly blocked by safety filters)")
                return text, attempt
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                status = self._status(exc)
                retryable = status in (429, 500, 502, 503, 504) or status is None
                if isinstance(exc, GenerationError) or not retryable:
                    raise
                if attempt < self.max_attempts:
                    delay = self.base_delay * (2 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
                    self._sleep(delay)
        raise GenerationError(f"{model}: gave up after {self.max_attempts} attempts: {last}") from last

    def _generate(self, system: str | None, user: str, use_cache: bool) -> tuple[str, str, int, bool]:
        """Return (text, model_used, api_attempts, from_cache)."""
        key = hashlib.sha256(
            json.dumps([self.models, system, user], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if use_cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["text"], data["model"], 0, True

        total_attempts, errors = 0, []
        for model in self.models:
            try:
                text, attempts = self._call_with_retry(model, system, user)
                total_attempts += attempts
                if use_cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps({"text": text, "model": model}, ensure_ascii=False), encoding="utf-8"
                    )
                return text, model, total_attempts, False
            except Exception as exc:  # noqa: BLE001 - try the next model
                total_attempts += self.max_attempts
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
        raise GenerationError("all models failed: " + " | ".join(errors))

    # -- public API ---------------------------------------------------------------
    def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        min_score: float | None = None,
        use_cache: bool = True,
    ) -> Answer:
        """RAG answer: ``chunks`` are the only evidence the model may use."""
        threshold = config.MIN_SIMILARITY if min_score is None else min_score
        if not chunks or chunks[0].score < threshold:
            best = chunks[0].score if chunks else 0.0
            return Answer(
                text=f"{INSUFFICIENT_PREFIX}: the closest passage in the documents has similarity "
                f"{best:.2f}, below the {threshold:.2f} needed to answer reliably.",
                sufficient=False,
                guard_triggered=True,
            )
        text, model, attempts, cached = self._generate(
            SYSTEM_PROMPT, build_user_prompt(question, chunks), use_cache
        )
        valid, invalid = parse_citations(text, len(chunks))
        return Answer(
            text=text,
            sufficient=not is_insufficient(text),
            model=model,
            cited=valid,
            invalid_citations=invalid,
            from_cache=cached,
            attempts=attempts,
        )

    def answer_without_context(self, question: str, use_cache: bool = True) -> Answer:
        """Base-LLM answer: no retrieval, no grounding rules (the comparison baseline)."""
        text, model, attempts, cached = self._generate(
            None, BASE_PROMPT.format(question=question), use_cache
        )
        return Answer(text=text, sufficient=True, model=model, from_cache=cached, attempts=attempts)
