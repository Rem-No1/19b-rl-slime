from __future__ import annotations

import pytest

from scripts.preflight import validate_topology


def test_original_six_gpu_topology_is_valid() -> None:
    validate_topology(6, train_tp=2, train_pp=1, train_cp=1, train_ep=2, rollout_tp=2)


def test_two_node_sixteen_gpu_topology_is_valid() -> None:
    validate_topology(16, train_tp=2, train_pp=2, train_cp=1, train_ep=2, rollout_tp=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ngpus": 16, "train_tp": 2, "train_pp": 3, "train_cp": 1, "train_ep": 2, "rollout_tp": 2}, "TP \\* PP \\* CP"),
        ({"ngpus": 16, "train_tp": 4, "train_pp": 1, "train_cp": 1, "train_ep": 2, "rollout_tp": 2}, "query groups"),
        ({"ngpus": 16, "train_tp": 2, "train_pp": 1, "train_cp": 1, "train_ep": 3, "rollout_tp": 2}, "EP \\* PP"),
    ],
)
def test_invalid_topology_is_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_topology(**kwargs)
