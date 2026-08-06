#!/usr/bin/env python3
"""Convert the requested Big-Math and AIME25 chat JSONL files for slime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_TRAIN_ROOT = Path("/mnt/data/user01/LLMData/train/math/Big-Math-RL-Verified ")
DEFAULT_TRAIN_FILES = (
    DEFAULT_TRAIN_ROOT / "rl_buckets_0.05/solve_rate_0.00_0.05_with_system_prompt.jsonl",
    DEFAULT_TRAIN_ROOT / "rl_buckets_0.05/solve_rate_0.05_0.10_with_system_prompt.jsonl",
)
DEFAULT_VAL_FILE = Path("/mnt/data/user01/LLMData/val/AIME25/AIME25.jsonl")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def split_prompt_and_label(record: dict, source: str) -> tuple[list[dict], str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source}: messages must be a nonempty list")
    assistant_indices = [i for i, message in enumerate(messages) if message.get("role") == "assistant"]
    if not assistant_indices or assistant_indices[-1] != len(messages) - 1:
        raise ValueError(f"{source}: final message must be the gold assistant answer")
    prompt = messages[: assistant_indices[-1]]
    if not prompt or not any(message.get("role") == "user" for message in prompt):
        raise ValueError(f"{source}: prompt must contain a user message")
    for message in prompt:
        if message.get("role") not in {"system", "user"}:
            raise ValueError(f"{source}: unsupported prompt role {message.get('role')!r}")
        if not str(message.get("content", "")).strip():
            raise ValueError(f"{source}: empty prompt content")
    label = str(messages[-1].get("content", "")).strip()
    if not label:
        raise ValueError(f"{source}: empty gold answer")
    return prompt, label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-files", nargs="+", type=Path, default=list(DEFAULT_TRAIN_FILES))
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-samples-per-file", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_paths = [path.expanduser().resolve() for path in args.train_files]
    val_path = args.val_file.expanduser().resolve()
    for path in [*train_paths, val_path]:
        if not path.is_file():
            raise SystemExit(f"[data error] file not found: {path}")

    output_dir = args.output_dir.expanduser().resolve()
    train_output = output_dir / "big_math_dapo_train.jsonl"
    val_output = output_dir / "aime25_val.jsonl"
    manifest_output = output_dir / "manifest.json"
    existing = [path for path in (train_output, val_output, manifest_output) if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"[data error] outputs already exist; pass --force to replace: {existing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_counts: dict[str, int] = {}
    system_prompts: set[str] = set()
    sample_ids: set[str] = set()
    train_rows = 0
    with train_output.open("w", encoding="utf-8") as output:
        for path in train_paths:
            bucket = path.stem.replace("_with_system_prompt", "")
            count = 0
            for line_number, record in iter_jsonl(path):
                if args.max_samples_per_file >= 0 and count >= args.max_samples_per_file:
                    break
                prompt, label = split_prompt_and_label(record, f"{path}:{line_number}")
                metadata = record.get("metadata") or {}
                sample_id = str(metadata.get("_sample_id") or f"{path.name}:{line_number}")
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample id: {sample_id}")
                sample_ids.add(sample_id)
                for message in prompt:
                    if message.get("role") == "system":
                        system_prompts.add(str(message["content"]))
                row = {
                    "prompt": prompt,
                    "label": label,
                    "metadata": {
                        "source_name": "big_math_rl_verified",
                        "sample_id": sample_id,
                        "source": str(metadata.get("source", "")),
                        "bucket": bucket,
                        "solve_rate": float(metadata.get("llama8b_solve_rate", -1.0)),
                        "source_row": int(metadata.get("_source_row", -1)),
                    },
                }
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                train_rows += 1
            bucket_counts[bucket] = count

    if len(system_prompts) != 1:
        raise ValueError(f"expected one shared system prompt, got {len(system_prompts)}")
    system_prompt = next(iter(system_prompts))

    val_rows = 0
    with val_output.open("w", encoding="utf-8") as output:
        for line_number, record in iter_jsonl(val_path):
            prompt, label = split_prompt_and_label(record, f"{val_path}:{line_number}")
            if not any(message.get("role") == "system" for message in prompt):
                prompt = [{"role": "system", "content": system_prompt}, *prompt]
            row = {
                "prompt": prompt,
                "label": label,
                "metadata": {"source_name": "aime25", "index": val_rows},
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            val_rows += 1

    manifest = {
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "bucket_counts": bucket_counts,
        "unique_sample_ids": len(sample_ids),
        "system_prompt": system_prompt,
        "max_samples_per_file": args.max_samples_per_file,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in train_paths],
        "validation_input": {"path": str(val_path), "sha256": sha256(val_path)},
        "outputs": {"train": str(train_output), "validation": str(val_output)},
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] train={train_rows} -> {train_output}")
    print(f"[done] validation={val_rows} -> {val_output}")
    print(f"[done] manifest -> {manifest_output}")


if __name__ == "__main__":
    main()

