"""Load per-expert Qwen3.5 MoE HF checkpoints with slime's native loader."""

from __future__ import annotations

import re
from typing import Any, Callable

import torch


_EXPERT_RE = re.compile(
    r"^decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\.linear_fc(?P<projection>[12])"
    r"(?:\.weight)?(?P<expert>\d+)?$"
)
_ORIGINAL_LOADER: Callable[[str, Any, Any], torch.Tensor] | None = None


def _strip_name(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name.removeprefix("language_model.")


def _expert_key_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts"


def _load_one_expert(reader: Any, prefix: str, projection: str, expert_id: int) -> torch.Tensor:
    expert_prefix = f"{prefix}.{expert_id}"
    if projection == "1":
        gate = reader.get_tensor(f"{expert_prefix}.gate_proj.weight")
        up = reader.get_tensor(f"{expert_prefix}.up_proj.weight")
        if gate.shape != up.shape:
            raise ValueError(f"gate/up shape mismatch for expert {expert_id}: {gate.shape} != {up.shape}")
        return torch.cat((gate, up), dim=0)
    return reader.get_tensor(f"{expert_prefix}.down_proj.weight")


def _load_all_experts(reader: Any, prefix: str, projection: str, num_experts: int) -> torch.Tensor:
    first = _load_one_expert(reader, prefix, projection, 0)
    fused = first.new_empty((num_experts, *first.shape))
    fused[0].copy_(first)
    for expert_id in range(1, num_experts):
        tensor = _load_one_expert(reader, prefix, projection, expert_id)
        if tensor.shape != first.shape:
            raise ValueError(
                f"inconsistent shape for expert {expert_id}: {tuple(tensor.shape)} != {tuple(first.shape)}"
            )
        fused[expert_id].copy_(tensor)
    return fused


def qwen35_per_expert_tensor(name: str, reader: Any, hf_config: Any) -> torch.Tensor:
    """Delegate normal tensors and synthesize only missing expert tensors."""

    if _ORIGINAL_LOADER is None:
        raise RuntimeError("install_qwen35_per_expert_loader must run before loading weights")

    match = _EXPERT_RE.fullmatch(_strip_name(name))
    if match is None:
        return _ORIGINAL_LOADER(name, reader, hf_config)

    layer = int(match.group("layer"))
    projection = match.group("projection")
    expert = match.group("expert")
    prefix = _expert_key_prefix(layer)
    fused_suffix = "gate_up_proj" if projection == "1" else "down_proj"

    # A standard fused checkpoint should continue through slime's native path.
    if f"{prefix}.{fused_suffix}" in reader:
        return _ORIGINAL_LOADER(name, reader, hf_config)

    text_config = getattr(hf_config, "text_config", hf_config)
    num_experts = int(text_config.num_experts)
    if expert is not None:
        expert_id = int(expert)
        if not 0 <= expert_id < num_experts:
            raise ValueError(f"expert id {expert_id} outside [0, {num_experts})")
        return _load_one_expert(reader, prefix, projection, expert_id)
    return _load_all_experts(reader, prefix, projection, num_experts)


def install_qwen35_per_expert_loader(args: Any) -> None:
    """Install the loader through slime's --custom-megatron-init-path hook."""

    del args
    global _ORIGINAL_LOADER
    from slime.backends.megatron_utils import hf_to_megatron

    current = hf_to_megatron._LOADERS["qwen3_5_moe"]
    if current is qwen35_per_expert_tensor:
        return
    _ORIGINAL_LOADER = current
    hf_to_megatron._LOADERS["qwen3_5_moe"] = qwen35_per_expert_tensor

