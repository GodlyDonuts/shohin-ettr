from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from build_pcf1_custody import PCF1CustodyError, _exact_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "model"
    (root / ".cache/huggingface").mkdir(parents=True)
    (root / ".cache/huggingface/.gitignore").write_text("x\n")
    (root / "config.json").write_text("{}\n")
    entries = {
        ".cache/huggingface/.gitignore": _sha256(
            root / ".cache/huggingface/.gitignore"
        ),
        "config.json": _sha256(root / "config.json"),
    }
    return root, entries


def _manifest(path: Path, entries: list[tuple[str, str]]) -> str:
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in entries))
    return _sha256(path)


@pytest.mark.parametrize("prefix", ["", "./"])
def test_exact_manifest_accepts_canonical_or_single_find_prefix(
    tmp_path: Path, prefix: str
) -> None:
    root, entries = _tree(tmp_path)
    manifest = tmp_path / "SHA256SUMS"
    digest = _manifest(
        manifest, [(prefix + name, value) for name, value in sorted(entries.items())]
    )
    result = _exact_manifest(
        root=root,
        manifest_path=manifest,
        expected_sha256=digest,
        label="model",
    )
    assert result["file_count"] == 2
    assert result["tree_sha256"]


@pytest.mark.parametrize(
    "names",
    [
        ["././config.json", "./.cache/huggingface/.gitignore"],
        ["../config.json", ".cache/huggingface/.gitignore"],
        ["config.json", "./config.json", ".cache/huggingface/.gitignore"],
        ["config.json", ".cache/./huggingface/.gitignore"],
    ],
)
def test_exact_manifest_rejects_alias_traversal_and_duplicate_canonical_names(
    tmp_path: Path, names: list[str]
) -> None:
    root, entries = _tree(tmp_path)
    manifest = tmp_path / "SHA256SUMS"
    rows = []
    for name in names:
        canonical = name.removeprefix("./")
        digest = entries.get(canonical, next(iter(entries.values())))
        rows.append((name, digest))
    manifest_sha256 = _manifest(manifest, rows)
    with pytest.raises(PCF1CustodyError, match="manifest entry is unsafe"):
        _exact_manifest(
            root=root,
            manifest_path=manifest,
            expected_sha256=manifest_sha256,
            label="model",
        )
