from __future__ import annotations

from pathlib import Path

from scripts.cluster_preflight import nearest_existing_parent, sha256_file


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"slime multinode\n")
    assert sha256_file(path) == "3c1cb1ac2c609d0587bf09990bb7b61a1ceda07e0f28f8e8663d22b197662300"


def test_nearest_existing_parent_handles_future_output_tree(tmp_path: Path) -> None:
    assert nearest_existing_parent(tmp_path / "new" / "checkpoints") == tmp_path
