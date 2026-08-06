"""DAPO math reward compatible with the previous verl experiment."""

from __future__ import annotations

import os
import re
from typing import Any


def extract_braced_value(text: str, marker: str = r"\boxed{") -> str | None:
    start = text.rfind(marker)
    if start < 0:
        return None
    position = start + len(marker)
    depth = 1
    chars: list[str] = []
    while position < len(text):
        char = text[position]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        position += 1
    return None


def extract_final_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL).strip()
    boxed = extract_braced_value(text)
    if boxed is not None:
        return boxed
    matches = list(re.finditer(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*(.+)", text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def normalize_answer(text: str) -> str:
    answer = extract_final_answer(text).strip().rstrip(".。")
    for _ in range(2):
        answer = re.sub(r"^\$+|\$+$", "", answer).strip()
        answer = re.sub(r"^\\\(|\\\)$", "", answer).strip()
        answer = re.sub(r"^\\\[|\\\]$", "", answer).strip()
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    answer = answer.replace(",", "")
    return re.sub(r"\s+", "", answer).lower()


def is_correct(response: str, label: str) -> bool:
    """Use slime's symbolic verifier first, then the previous exact fallback."""

    correct = False
    try:
        from slime.rollout.rm_hub.math_utils import grade_answer_verl

        correct = bool(grade_answer_verl(response, label))
    except Exception:
        # The fallback is intentional: malformed symbolic expressions should
        # score zero unless their normalized final answers match exactly.
        correct = False
    return correct or normalize_answer(response) == normalize_answer(label)


def apply_overlong_penalty(args: Any, score: float, response_length: int) -> tuple[float, float]:
    """Apply the linear penalty used by verl's DAPORewardManager."""

    buffer_len = int(os.environ.get("DAPO_OVERLONG_BUFFER_LEN", "4096"))
    factor = float(os.environ.get("DAPO_OVERLONG_PENALTY_FACTOR", "1.0"))
    max_response_len = int(getattr(args, "rollout_max_response_len", 0) or 0)
    if buffer_len <= 0 or max_response_len <= 0:
        return score, 0.0
    if buffer_len > max_response_len:
        raise ValueError(
            f"DAPO_OVERLONG_BUFFER_LEN={buffer_len} exceeds rollout_max_response_len={max_response_len}"
        )
    expected_len = max_response_len - buffer_len
    exceed_len = response_length - expected_len
    penalty = min(-exceed_len / buffer_len * factor, 0.0)
    return score + penalty, penalty


async def compute_reward(args: Any, sample: Any, **kwargs: Any) -> dict[str, Any]:
    """Return shaped training score plus unshaped accuracy for filtering/eval."""

    del kwargs
    response = str(sample.response)
    label = str(sample.label or "")
    correct = is_correct(response, label)
    score, overlong_penalty = apply_overlong_penalty(
        args,
        1.0 if correct else 0.0,
        int(sample.response_length),
    )
    return {
        "score": float(score),
        "acc": float(correct),
        "pred": extract_final_answer(response),
        "format_ok": extract_braced_value(response) is not None,
        "overlong_penalty": float(overlong_penalty),
    }


def check_nonzero_acc_std(args: Any, samples: list[Any], **kwargs: Any) -> Any:
    """DAPO dynamic filter based on raw correctness, not shaped score."""

    del args, kwargs
    accuracies = []
    for sample in samples:
        reward = sample.reward
        acc = reward.get("acc", 0.0) if isinstance(reward, dict) else reward
        accuracies.append(float(acc))
    keep = max(accuracies) - min(accuracies) > 1e-6
    reason = None if keep else f"zero_std_acc_{accuracies[0]:.1f}"
    try:
        from slime.rollout.filter_hub.base_types import DynamicFilterOutput

        return DynamicFilterOutput(keep=keep, reason=reason)
    except ImportError:
        # slime accepts the legacy boolean result as well. This also keeps the
        # function independently unit-testable before slime is installed.
        return keep

