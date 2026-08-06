from __future__ import annotations

from types import SimpleNamespace

import torch

import slime_hooks.qwen35_per_expert as compat


class Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def __contains__(self, key):
        return key in self.tensors

    def get_tensor(self, key):
        return self.tensors[key]


def setup_function() -> None:
    compat._ORIGINAL_LOADER = lambda name, reader, config: reader.get_tensor(name)


def test_numbered_fc1_loads_individual_gate_and_up() -> None:
    prefix = "model.language_model.layers.0.mlp.experts.1"
    reader = Reader(
        {
            f"{prefix}.gate_proj.weight": torch.full((1, 3), 1.0),
            f"{prefix}.up_proj.weight": torch.full((1, 3), 2.0),
        }
    )
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=2))
    tensor = compat.qwen35_per_expert_tensor(
        "module.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight1", reader, config
    )
    assert tuple(tensor.shape) == (2, 3)
    torch.testing.assert_close(tensor[0], torch.full((3,), 1.0))
    torch.testing.assert_close(tensor[1], torch.full((3,), 2.0))


def test_grouped_fc2_stacks_experts() -> None:
    prefix = "model.language_model.layers.0.mlp.experts"
    reader = Reader(
        {
            f"{prefix}.0.down_proj.weight": torch.full((3, 1), 1.0),
            f"{prefix}.1.down_proj.weight": torch.full((3, 1), 2.0),
        }
    )
    config = SimpleNamespace(text_config=SimpleNamespace(num_experts=2))
    tensor = compat.qwen35_per_expert_tensor(
        "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight", reader, config
    )
    assert tuple(tensor.shape) == (2, 3, 1)
    torch.testing.assert_close(tensor[1], torch.full((3, 1), 2.0))

