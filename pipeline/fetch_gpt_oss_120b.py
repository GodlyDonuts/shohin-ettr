#!/usr/bin/env python3
"""Acquire the exact single-H100 GPT-OSS-120B Transformers projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
SCHEMA = "shohin-gpt-oss-120b-acquisition-v1"
MODEL_FILES: dict[str, tuple[int, str]] = {
    ".gitattributes": (
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "LICENSE": (
        11357,
        "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
    ),
    "README.md": (
        7111,
        "92b0408cf5dce04e4c5e4e5f2be0361da710d3650baefd7d7519badd1c687919",
    ),
    "USAGE_POLICY": (
        201,
        "fc48d386a7a7ff8b066f743cfe62df683ab16892450f5bb7357bb4de261cd037",
    ),
    "chat_template.jinja": (
        16738,
        "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
    ),
    "config.json": (
        2089,
        "933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566",
    ),
    "generation_config.json": (
        177,
        "f9970ada892d2d1f72e3ed0a6535ccebadd11897318794ca671d8c7014c957da",
    ),
    "model-00000-of-00014.safetensors": (
        4_625_017_896,
        "695218884684c611fe08a74751ee443f971e9bd9bc062edba822da3fe45969b7",
    ),
    "model-00001-of-00014.safetensors": (
        4_115_586_736,
        "a881aa5f561b26a22b14a8262aa61849ace349ffd73d74769e030ac90a1fcf8a",
    ),
    "model-00002-of-00014.safetensors": (
        4_625_017_888,
        "022478dd04398c5bdb545a5be0a6437ecc2eb53d1dbd29edafcfff4b3ddf0a41",
    ),
    "model-00003-of-00014.safetensors": (
        4_115_586_752,
        "47aee9e7b9d5bedb215042c01ccededd9bd9c30b0dddea862dc2506b9d6c74de",
    ),
    "model-00004-of-00014.safetensors": (
        4_625_017_896,
        "f6c2752acda607b1d5ca52df9e75c1b9b2761e6875ff10c9bd6ddac473c0262e",
    ),
    "model-00005-of-00014.safetensors": (
        4_115_586_696,
        "0c8dd401544c31cb93b8459eee7da20ea2a07626a59455d7d92b85257df9b46c",
    ),
    "model-00006-of-00014.safetensors": (
        4_625_017_856,
        "28d839f2e027985a8b14e45f2323798862eddb7770ee9800ea6b7c803abee489",
    ),
    "model-00007-of-00014.safetensors": (
        4_060_267_176,
        "c8958c5f183c04f6ea959cfd90562b5128124154b2bbf979b8a22b9405b30ed8",
    ),
    "model-00008-of-00014.safetensors": (
        4_625_017_896,
        "bf1f2a88868ffc37d520dcf77d26f0e823710b5e682d473ff10f6974fa3b7517",
    ),
    "model-00009-of-00014.safetensors": (
        4_170_906_304,
        "f72d34a4004241b45c332b61f8ffa124e9a913bc1ab442b66e717d3e94e741ce",
    ),
    "model-00010-of-00014.safetensors": (
        4_625_017_896,
        "f48c867c2cb0a44bfc2f8768cb98e4aec9a350946fceacfebdcad5d32ad4a471",
    ),
    "model-00011-of-00014.safetensors": (
        4_115_586_752,
        "a06851b2cfd35f48722f823bc1ab8f7bcb4a878a5b8e975f4d3544f230454eeb",
    ),
    "model-00012-of-00014.safetensors": (
        4_064_660_808,
        "3af33667c307e20ae2a7648ea52653de46dd0171601ec5c696e47a2f5d5bf1e4",
    ),
    "model-00013-of-00014.safetensors": (
        4_625_017_896,
        "bcbcb74b043e071d1e05471d500d74dcf661175e00878ed302ccdf1801a75aef",
    ),
    "model-00014-of-00014.safetensors": (
        4_115_586_736,
        "54b1be1609696c307cc5ca117b1fa54feaddebffa04e9c2db117652a01964230",
    ),
    "model.safetensors.index.json": (
        54511,
        "ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e",
    ),
    "special_tokens_map.json": (
        98,
        "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293",
    ),
    "tokenizer.json": (
        27_868_174,
        "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    ),
    "tokenizer_config.json": (
        4200,
        "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
    ),
}
MODEL_BYTES = sum(size for size, _ in MODEL_FILES.values())


class GptOssAcquisitionError(RuntimeError):
    """The exact GPT-OSS acquisition contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise GptOssAcquisitionError(f"refusing existing acquisition output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_projection(
    root: Path, files: Mapping[str, tuple[int, str]] = MODEL_FILES
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise GptOssAcquisitionError("acquisition root differs")
    actual: set[str] = set()
    covered_bytes = 0
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not candidate.is_symlink():
            continue
        if not stat.S_ISREG(mode) or candidate.is_symlink():
            raise GptOssAcquisitionError("acquisition member is linked or special")
        actual.add(relative)
    if actual != set(files):
        raise GptOssAcquisitionError("acquisition membership differs")
    for relative, (expected_size, expected_sha256) in files.items():
        candidate = root / relative
        if (
            candidate.stat().st_size != expected_size
            or sha256_file(candidate) != expected_sha256
        ):
            raise GptOssAcquisitionError(f"acquisition file differs: {relative}")
        covered_bytes += expected_size
    return {
        "files": len(files),
        "covered_bytes": covered_bytes,
        "exact_membership": True,
    }


def manifest_text(
    files: Mapping[str, tuple[int, str]] = MODEL_FILES,
    revision: str = MODEL_REVISION,
) -> str:
    source_sha256 = hashlib.sha256(f"{revision}\n".encode()).hexdigest()
    rows = [(relative, digest) for relative, (_, digest) in files.items()]
    rows.append(("SOURCE_REVISION", source_sha256))
    return "".join(f"{digest}  {relative}\n" for relative, digest in sorted(rows))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve(strict=False)
    report = args.report.resolve(strict=False)
    if (
        output_root.exists()
        or output_root.is_symlink()
        or report.exists()
        or report.is_symlink()
    ):
        raise GptOssAcquisitionError("final acquisition output exists")
    stage = output_root.with_name(f".{output_root.name}.partial")
    if stage.is_symlink():
        raise GptOssAcquisitionError("acquisition staging root is symbolic")
    stage.mkdir(parents=True, exist_ok=True)
    for provenance_name in ("SOURCE_REVISION", "SHA256SUMS"):
        provenance = stage / provenance_name
        if provenance.exists() or provenance.is_symlink():
            if provenance.is_symlink() or not provenance.is_file():
                raise GptOssAcquisitionError("partial acquisition provenance differs")
            provenance.unlink()

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        repo_type="model",
        revision=MODEL_REVISION,
        local_dir=stage,
        allow_patterns=sorted(MODEL_FILES),
        max_workers=args.workers,
    )
    cache = stage / ".cache"
    if cache.exists():
        if cache.is_symlink() or not cache.is_dir():
            raise GptOssAcquisitionError("acquisition cache differs")
        shutil.rmtree(cache)
    projection = verify_projection(stage)
    _atomic_text(stage / "SOURCE_REVISION", f"{MODEL_REVISION}\n")
    sums = manifest_text()
    _atomic_text(stage / "SHA256SUMS", sums)
    manifest_sha256 = hashlib.sha256(sums.encode()).hexdigest()
    os.replace(stage, output_root)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_files": projection["files"],
        "model_bytes": projection["covered_bytes"],
        "projection_excludes": ["metal/**", "original/**"],
        "manifest_sha256": manifest_sha256,
        "manifest_entries": len(MODEL_FILES) + 1,
        "exact_membership": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise GptOssAcquisitionError("download worker count differs")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
