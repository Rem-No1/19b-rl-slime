from __future__ import annotations

import pytest

from scripts.prepare_data import split_prompt_and_label


def test_split_prompt_and_label_preserves_system_prompt() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "box it"},
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ]
    }
    prompt, label = split_prompt_and_label(row, "unit")
    assert prompt[0] == {"role": "system", "content": "box it"}
    assert label == "2"


def test_split_rejects_nonfinal_assistant() -> None:
    row = {"messages": [{"role": "assistant", "content": "2"}, {"role": "user", "content": "x"}]}
    with pytest.raises(ValueError, match="final message"):
        split_prompt_and_label(row, "unit")
