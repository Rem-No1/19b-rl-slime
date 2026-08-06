#!/usr/bin/env python3
"""Verify a dedicated Ray cluster and shared project paths on every node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def probe_node(config: dict[str, str]) -> dict[str, Any]:
    model = Path(config["model"])
    train_data = Path(config["train_data"])
    eval_data = Path(config["eval_data"])
    project_root = Path(config["project_root"])
    slime_root = Path(config["slime_root"])
    output_root = Path(config["output_root"])
    signature_paths = {
        "model_config": model / "config.json",
        "train_data": train_data,
        "eval_data": eval_data,
        "slime_train": slime_root / "train.py",
        "project_reward": project_root / "slime_hooks" / "reward.py",
        "project_checkpoint_loader": project_root / "slime_hooks" / "qwen35_per_expert.py",
        "tis_config": project_root / "configs" / "tis_token_batch_normalized.yaml",
    }
    model_index = model / "model.safetensors.index.json"
    if model_index.is_file():
        signature_paths["model_index"] = model_index
    missing = [str(path) for path in signature_paths.values() if not path.is_file()]
    model_shards: dict[str, int | None] = {}
    if model_index.is_file():
        index = json.loads(model_index.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    else:
        shard_names = sorted(path.name for path in model.glob("*.safetensors"))
    if not shard_names:
        missing.append(str(model / "*.safetensors"))
    for shard_name in shard_names:
        shard_path = model / shard_name
        model_shards[shard_name] = shard_path.stat().st_size if shard_path.is_file() else None
        if not shard_path.is_file():
            missing.append(str(shard_path))
    signatures = {
        name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in signature_paths.items()
        if path.is_file()
    }
    nvidia = subprocess.run(
        ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
    )
    gpu_lines = [line for line in nvidia.stdout.splitlines() if line.strip().startswith("GPU ")]
    writable_parent = nearest_existing_parent(output_root)
    shared_probe = Path(config["shared_probe"])
    try:
        shared_probe_matches = shared_probe.read_text(encoding="utf-8") == config["shared_token"]
    except OSError:
        shared_probe_matches = False
    return {
        "hostname": socket.gethostname(),
        "node_ip": os.environ.get("RAY_NODE_IP_ADDRESS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_count": len(gpu_lines),
        "nvidia_smi_ok": nvidia.returncode == 0,
        "missing": missing,
        "signatures": signatures,
        "model_shards": model_shards,
        "slime_revision": git_revision(slime_root),
        "output_parent": str(writable_parent),
        "output_parent_writable": os.access(writable_parent, os.W_OK),
        "shared_probe_matches": shared_probe_matches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--slime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_nodes <= 0 or args.gpus_per_node <= 0:
        raise SystemExit("[cluster preflight error] node and GPU counts must be positive")

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.address, logging_level="ERROR")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    if len(nodes) != args.expected_nodes:
        raise SystemExit(
            f"[cluster preflight error] alive Ray nodes={len(nodes)}, expected={args.expected_nodes}"
        )
    for node in nodes:
        advertised = int(node["Resources"].get("GPU", 0))
        if advertised != args.gpus_per_node:
            raise SystemExit(
                "[cluster preflight error] "
                f"node {node['NodeManagerAddress']} advertises {advertised} GPUs, "
                f"expected {args.gpus_per_node}"
            )

    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"[cluster preflight error] cannot create output root {args.output_root}: {exc}"
        ) from exc
    shared_token = uuid.uuid4().hex
    shared_probe = args.output_root / f".slime_cluster_preflight_{shared_token}"
    shared_probe.write_text(shared_token, encoding="utf-8")
    config = {
        "model": str(args.model),
        "train_data": str(args.train_data),
        "eval_data": str(args.eval_data),
        "project_root": str(args.project_root),
        "slime_root": str(args.slime_root),
        "output_root": str(args.output_root),
        "shared_probe": str(shared_probe),
        "shared_token": shared_token,
    }
    try:
        remote_probe = ray.remote(num_cpus=0.01)(probe_node)
        refs = [
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
            ).remote(config)
            for node in nodes
        ]
        results = ray.get(refs)
    finally:
        shared_probe.unlink(missing_ok=True)
        ray.shutdown()

    failures: list[str] = []
    for node, result in zip(nodes, results, strict=True):
        label = f"{result['hostname']}({node['NodeManagerAddress']})"
        if result["missing"]:
            failures.append(f"{label}: missing paths: {result['missing']}")
        if not result["nvidia_smi_ok"] or result["physical_gpu_count"] < args.gpus_per_node:
            failures.append(
                f"{label}: physical GPUs={result['physical_gpu_count']}, expected at least {args.gpus_per_node}"
            )
        if not result["output_parent_writable"]:
            failures.append(f"{label}: output parent is not writable: {result['output_parent']}")
        if not result["shared_probe_matches"]:
            failures.append(f"{label}: output root is not the same shared filesystem as head")
        if not result["slime_revision"]:
            failures.append(f"{label}: cannot determine slime Git revision")

    baseline = results[0]
    for node, result in zip(nodes[1:], results[1:], strict=True):
        label = f"{result['hostname']}({node['NodeManagerAddress']})"
        if result["signatures"] != baseline["signatures"]:
            failures.append(f"{label}: model metadata/data/source signatures differ from head")
        if result["model_shards"] != baseline["model_shards"]:
            failures.append(f"{label}: model shard names or sizes differ from head")
        if result["slime_revision"] != baseline["slime_revision"]:
            failures.append(f"{label}: slime Git revision differs from head")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("[cluster preflight error] " + "\n[cluster preflight error] ".join(failures))
    print(
        "[cluster preflight ok] "
        f"nodes={args.expected_nodes} gpus_per_node={args.gpus_per_node} "
        f"total_gpus={args.expected_nodes * args.gpus_per_node} shared_paths=consistent"
    )


if __name__ == "__main__":
    main()
