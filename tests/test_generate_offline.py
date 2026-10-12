"""Offline tests for the grounding layers, retry/backoff, model fallback and caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import config
from src.generate import (
    INSUFFICIENT_PREFIX,
    GenerationError,
    Generator,
    build_user_prompt,
    is_insufficient,
    parse_citations,
)
from src.retrieve import RetrievedChunk


def chunk(rank: int, score: float, text: str = "A data controller shall act.") -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank, score=score, distance=1 - score, text=text,
        source="Data Protection Act.pdf", page_start=22, page_end=22, section="", chunk_id=f"c{rank}",
    )


class ApiError(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class ScriptedClient:
    """Stands in for genai.Client: replays a list of replies/exceptions per call."""

    def __init__(self, script):
        self.calls: list[str] = []
        self.models = self
        self._script = list(script)

    def generate_content(self, model, contents, config):  # noqa: A002 - mirrors the SDK
        self.calls.append(model)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def make_generator(script, tmp_path, models=("m1", "m2"), max_attempts=3):
    client = ScriptedClient(script)
    sleeps: list[float] = []
    gen = Generator(client=client, models=models, cache_dir=tmp_path, max_attempts=max_attempts,
                    base_delay=1.0, sleep=sleeps.append)
    return gen, client, sleeps


# ---------------------------------------------------------------- prompt + parsing
def test_prompt_numbers_passages_and_includes_question():
    prompt = build_user_prompt("How long?", [chunk(1, 0.9, "alpha"), chunk(2, 0.8, "beta")])
    assert "[1] (" in prompt and "[2] (" in prompt
    assert "alpha" in prompt and "beta" in prompt
    assert prompt.rstrip().endswith("QUESTION: How long?")
    assert "p.22" in prompt  # page provenance reaches the model


def test_parse_citations_separates_valid_and_invalid():
    assert parse_citations("Yes [1]. Also [3] and [1, 2].", n_passages=3) == ([1, 2, 3], [])
    assert parse_citations("See [2] and [9].", n_passages=3) == ([2], [9])
    assert parse_citations("no citations", n_passages=3) == ([], [])


def test_insufficient_detection():
    assert is_insufficient(f"{INSUFFICIENT_PREFIX}: nothing about tax.")
    assert is_insufficient(f"  {INSUFFICIENT_PREFIX.lower()}: x")
    assert not is_insufficient("The period is 72 hours [1].")


# ----------------------------------------------------------------------- the guard
def test_guard_blocks_low_similarity_without_calling_the_llm(tmp_path):
    gen, client, _ = make_generator([], tmp_path)  # any API call would crash: empty script
    ans = gen.answer("What is the corporate tax rate?", [chunk(1, 0.31)], min_score=0.5)
    assert ans.guard_triggered and not ans.sufficient
    assert ans.text.startswith(INSUFFICIENT_PREFIX)
    assert client.calls == [] and ans.attempts == 0


def test_guard_blocks_when_nothing_was_retrieved(tmp_path):
    gen, client, _ = make_generator([], tmp_path)
    assert gen.answer("anything", [], min_score=0.5).guard_triggered
    assert client.calls == []


def test_guard_lets_good_matches_through_and_checks_citations(tmp_path):
    gen, client, _ = make_generator(["It is seventy-two hours [1]. Also see [7]."], tmp_path)
    ans = gen.answer("How many hours?", [chunk(1, 0.8), chunk(2, 0.7)], min_score=0.5)
    assert not ans.guard_triggered and ans.sufficient
    assert ans.cited == [1] and ans.invalid_citations == [7]  # fabricated citation is flagged
    assert client.calls == ["m1"] and ans.model == "m1"


def test_model_refusal_is_reported_as_insufficient(tmp_path):
    gen, _, _ = make_generator([f"{INSUFFICIENT_PREFIX}: the passages do not mention GDPR."], tmp_path)
    ans = gen.answer("GDPR fine?", [chunk(1, 0.6)], min_score=0.5)
    assert not ans.sufficient and not ans.guard_triggered


def test_default_threshold_comes_from_config(tmp_path):
    gen, client, _ = make_generator([], tmp_path)
    just_below = chunk(1, config.MIN_SIMILARITY - 0.01)
    assert gen.answer("q", [just_below]).guard_triggered
    assert client.calls == []


# ------------------------------------------------------------- retry / fallback / cache
def test_retries_503_with_exponential_backoff_then_succeeds(tmp_path):
    gen, client, sleeps = make_generator([ApiError(503), ApiError(503), "ok [1]"], tmp_path)
    ans = gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    assert ans.text == "ok [1]" and ans.attempts == 3
    assert client.calls == ["m1", "m1", "m1"]
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # backoff grows


def test_rate_limit_429_is_retried(tmp_path):
    gen, client, _ = make_generator([ApiError(429), "fine [1]"], tmp_path)
    assert gen.answer("q", [chunk(1, 0.9)], min_score=0.5).text == "fine [1]"
    assert client.calls == ["m1", "m1"]


def test_falls_back_to_next_model_after_exhausting_retries(tmp_path):
    script = [ApiError(503)] * 3 + ["from fallback [1]"]
    gen, client, _ = make_generator(script, tmp_path, max_attempts=3)
    ans = gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    assert ans.model == "m2" and ans.text == "from fallback [1]"
    assert client.calls == ["m1", "m1", "m1", "m2"]


def test_client_errors_are_not_retried_but_do_fall_back(tmp_path):
    gen, client, sleeps = make_generator([ApiError(404), "ok [1]"], tmp_path)
    ans = gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    assert client.calls == ["m1", "m2"] and ans.model == "m2" and sleeps == []


def test_raises_when_every_model_fails(tmp_path):
    gen, _, _ = make_generator([ApiError(503)] * 6, tmp_path, max_attempts=3)
    with pytest.raises(GenerationError):
        gen.answer("q", [chunk(1, 0.9)], min_score=0.5)


def test_second_identical_call_is_served_from_the_cache(tmp_path):
    gen, client, _ = make_generator(["cached answer [1]"], tmp_path)
    first = gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    second = gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    assert not first.from_cache and second.from_cache
    assert second.text == first.text and second.attempts == 0
    assert len(client.calls) == 1


def test_cache_can_be_bypassed(tmp_path):
    gen, client, _ = make_generator(["one [1]", "two [1]"], tmp_path)
    gen.answer("q", [chunk(1, 0.9)], min_score=0.5)
    again = gen.answer("q", [chunk(1, 0.9)], min_score=0.5, use_cache=False)
    assert again.text == "two [1]" and len(client.calls) == 2


def test_base_answer_uses_no_context(tmp_path):
    gen, client, _ = make_generator(["72 hours."], tmp_path)
    ans = gen.answer_without_context("How many hours?")
    assert ans.text == "72 hours." and ans.cited == [] and not ans.guard_triggered
