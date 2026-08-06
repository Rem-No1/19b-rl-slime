#!/usr/bin/env python3
"""Cheap model/data/topology checks before starting Ray or allocating weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_and_validate(path: Path, expected_source: str) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("prompt"), list) or not row["prompt"]:
                raise ValueError(f"{path}:{line_number}: prompt must be nonempty messages")
            if not str(row.get("label", "")).strip():
                raise ValueError(f"{path}:{line_number}: label is empty")
            if (row.get("metadata") or {}).get("source_name") != expected_source:
                raise ValueError(f"{path}:{line_number}: unexpected source_name")
            count += 1
    return count


def checkpoint_keys(model_path: Path) -> set[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        return set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate the model checkpoint") from exc
    keys: set[str] = set()
    for path in model_path.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def validate_topology(
    ngpus: int,
    train_tp: int,
    train_pp: int,
    train_cp: int,
    train_ep: int,
    rollout_tp: int,
) -> None:
    """Validate this model's Megatron/SGLang parallel dimensions."""
    sizes = (train_tp, train_pp, train_cp, train_ep, rollout_tp)
    if ngpus <= 0 or any(size <= 0 for size in sizes):
        raise ValueError("GPU and parallel sizes must be positive")
    dense_model_parallel = train_tp * train_pp * train_cp
    if ngpus % dense_model_parallel:
        raise ValueError("TP * PP * CP must divide total actor GPU count")
    if ngpus % rollout_tp:
        raise ValueError("rollout TP must divide total actor GPU count in colocate mode")
    if ngpus % (train_ep * train_pp):
        raise ValueError("EP * PP must divide total actor GPU count")
    if 128 % train_ep:
        raise ValueError("train EP must divide 128 experts")
    if 40 % train_pp:
        raise ValueError("train PP must divide 40 transformer layers")
    if 16 % train_tp or 2 % train_tp:
        raise ValueError("train TP must divide 16 attention heads and 2 query groups")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--slime-root", type=Path, required=True)
    parser.add_argument("--ngpus", type=int, required=True)
    parser.add_argument("--train-tp", type=int, required=True)
    parser.add_argument("--train-pp", type=int, default=1)
    parser.add_argument("--train-cp", type=int, default=1)
    parser.add_argument("--train-ep", type=int, required=True)
    parser.add_argument("--rollout-tp", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.slime_root):
        if not path.is_dir():
            raise SystemExit(f"[preflight error] directory not found: {path}")
    for path in (args.train_data, args.eval_data, args.slime_root / "train.py"):
        if not path.is_file():
            raise SystemExit(f"[preflight error] file not found: {path}")
    try:
        validate_topology(
            args.ngpus,
            args.train_tp,
            args.train_pp,
            args.train_cp,
            args.train_ep,
            args.rollout_tp,
        )
    except ValueError as exc:
        raise SystemExit(f"[preflight error] {exc}") from exc

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    expected = {
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 248320,
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise SystemExit(f"[preflight error] model {key}={text.get(key)!r}, expected {value!r}")

    keys = checkpoint_keys(args.model)
    per_expert = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
    fused = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    if per_expert not in keys and fused not in keys:
        raise SystemExit("[preflight error] unsupported/partial expert checkpoint layout")
    layout = "per_expert_hook" if per_expert in keys else "pre_fused"
    if layout == "per_expert_hook":
        missing = [
            f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
            for layer in range(40)
            for expert in range(128)
            for projection in ("gate", "up", "down")
            if f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight" not in keys
        ]
        if missing:
            raise SystemExit(
                f"[preflight error] incomplete per-expert layout: missing {len(missing)} keys; first={missing[0]}"
            )

    train_rows = count_and_validate(args.train_data, "big_math_rl_verified")
    eval_rows = count_and_validate(args.eval_data, "aime25")
    print(
        "[preflight ok] "
        f"model=qwen3.5-moe-19b experts=128 layout={layout} "
        f"train_rows={train_rows} eval_rows={eval_rows} "
        f"gpus={args.ngpus} train_tp={args.train_tp} train_pp={args.train_pp} train_cp={args.train_cp} "
        f"train_ep={args.train_ep} rollout_tp={args.rollout_tp}"
    )


if __name__ == "__main__":
    main()
