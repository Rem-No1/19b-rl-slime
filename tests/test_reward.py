from __future__ import annotations

import asyncio
from types import SimpleNamespace

from slime_hooks.reward import apply_overlong_penalty, check_nonzero_acc_std, compute_reward, extract_braced_value


def test_nested_box_extraction() -> None:
    assert extract_braced_value(r"work \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_overlong_penalty_matches_verl(monkeypatch) -> None:
    monkeypatch.setenv("DAPO_OVERLONG_BUFFER_LEN", "20")
    monkeypatch.setenv("DAPO_OVERLONG_PENALTY_FACTOR", "1.0")
    score, penalty = apply_overlong_penalty(SimpleNamespace(rollout_max_response_len=100), 1.0, 90)
    assert score == 0.5
    assert penalty == -0.5


def test_overlong_penalty_defaults_to_half_strength(monkeypatch) -> None:
    monkeypatch.setenv("DAPO_OVERLONG_BUFFER_LEN", "20")
    monkeypatch.delenv("DAPO_OVERLONG_PENALTY_FACTOR", raising=False)
    score, penalty = apply_overlong_penalty(SimpleNamespace(rollout_max_response_len=100), 1.0, 100)
    assert score == 0.5
    assert penalty == -0.5


def test_overlong_penalty_starts_at_20480_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DAPO_OVERLONG_BUFFER_LEN", raising=False)
    monkeypatch.delenv("DAPO_OVERLONG_PENALTY_FACTOR", raising=False)
    score_at_boundary, penalty_at_boundary = apply_overlong_penalty(
        SimpleNamespace(rollout_max_response_len=24576), 1.0, 20480
    )
    score, penalty = apply_overlong_penalty(SimpleNamespace(rollout_max_response_len=24576), 1.0, 22528)
    assert score_at_boundary == 1.0
    assert penalty_at_boundary == 0.0
    assert score == 0.75
    assert penalty == -0.25


def test_compute_reward_returns_acc_and_shaped_score(monkeypatch) -> None:
    monkeypatch.setenv("DAPO_OVERLONG_BUFFER_LEN", "0")
    sample = SimpleNamespace(response=r"reasoning \boxed{42}", label="42", response_length=5)
    reward = asyncio.run(compute_reward(SimpleNamespace(rollout_max_response_len=100), sample))
    assert reward["score"] == 1.0
    assert reward["acc"] == 1.0
    assert reward["format_ok"] is True


def test_dynamic_filter_uses_acc_not_shaped_score() -> None:
    samples = [
        SimpleNamespace(reward={"score": -0.2, "acc": 0.0}),
        SimpleNamespace(reward={"score": -0.8, "acc": 0.0}),
    ]
    output = check_nonzero_acc_std(SimpleNamespace(), samples)
    assert (output.keep if hasattr(output, "keep") else output) is False
