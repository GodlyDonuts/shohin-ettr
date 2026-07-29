import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline.test_verify_tokenized_shards import build_corpus
from pipeline.tokenize_shards import canonical_payload_sha256


JOBS = Path(__file__).with_name("jobs")


def _script(name: str) -> str:
    return (JOBS / name).read_text(encoding="ascii")


def test_cross_source_audits_are_priority_ordered_and_fail_closed() -> None:
    for name, module, removal in (
        (
            "audit_cross_source_exact_dedup.sbatch",
            "audit_cross_source_exact_dedup.py",
            "exact_duplicate_removals.jsonl.zst",
        ),
        (
            "audit_cross_source_near_dedup.sbatch",
            "audit_cross_source_near_dedup.py",
            "near_duplicate_removals.jsonl.zst",
        ),
    ):
        script = _script(name)
        assert module in script
        assert removal in script
        assert "CORPUS_SPEC_FILE_SHA256" in script
        assert "RUNTIME_SHA256SUMS_SHA256" in script
        assert 'arguments+=(--corpus "$spec")' in script
        assert '".partial"' in script
        assert "--skip-external-input-verification" not in script
        assert "chmod 0444" in script


def test_residual_jobs_bind_audit_and_publish_fresh_outputs() -> None:
    exact = _script("materialize_cross_source_exact_residual.sbatch")
    near = _script("materialize_cross_source_near_residual.sbatch")
    assert "--dedup-dir" in exact
    assert "--near-dir" in near
    assert "--source-selection-code" in near
    assert "SOURCE_SELECTION_CODE_SHA256" in near
    for script in (exact, near):
        assert "RUNTIME_SHA256SUMS_SHA256" in script
        assert '".partial"' in script
        assert '! -e "$OUTPUT_DIR"' in script
        assert "documents.jsonl.zst" in script
        assert "chmod 0444" in script


def test_holdout_job_runs_creator_then_independent_verifier() -> None:
    script = _script("materialize_v3_holdout_split.sbatch")
    assert "materialize_v3_holdout_split.py" in script
    assert "verify_v3_holdout_split.py" in script
    assert "SOURCE_SELECTION_CODE_SHA256" in script
    assert "--document-validation-bps" in script
    assert "--domain-validation-bps" in script
    assert "partition_verified" in script
    assert "set -o noclobber" in script
    assert "chmod 0444" in script


def test_exact_audit_launcher_executes_complete_small_contract(
    tmp_path: Path,
) -> None:
    if not Path("/usr/bin/sha256sum").is_file():
        pytest.skip("Slurm launcher integration requires the Stokes GNU path")
    repo = Path(__file__).parents[1]
    first_root = tmp_path / "first-root"
    second_root = tmp_path / "second-root"
    first_root.mkdir()
    second_root.mkdir()
    first, selection = build_corpus(first_root, schema="v3")
    second, _ = build_corpus(second_root, schema="v3")
    for corpus in (first, second):
        manifest_path = corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest["filters"] = {"exact_dedup": True}
        manifest.pop("payload_sha256")
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    auditor = repo / "pipeline" / "audit_cross_source_exact_dedup.py"
    runtime_receipt = tmp_path / "SHA256SUMS"
    runtime_receipt.write_text(
        f"{hashlib.sha256(auditor.read_bytes()).hexdigest()}  "
        "pipeline/audit_cross_source_exact_dedup.py\n",
        encoding="ascii",
    )
    specs = tmp_path / "corpora.txt"
    specs.write_text(
        f"incumbent={first}::{selection}\n"
        f"challenger={second}::{selection}\n",
        encoding="ascii",
    )
    output = tmp_path / "exact-output"
    environment = {
        **os.environ,
        "CODE_ROOT": str(repo),
        "CORPUS_SPEC_FILE": str(specs),
        "CORPUS_SPEC_FILE_SHA256": hashlib.sha256(
            specs.read_bytes()
        ).hexdigest(),
        "OUTPUT_DIR": str(output),
        "PY": sys.executable,
        "RUNTIME_SHA256SUMS": str(runtime_receipt),
        "RUNTIME_SHA256SUMS_SHA256": hashlib.sha256(
            runtime_receipt.read_bytes()
        ).hexdigest(),
    }
    completed = subprocess.run(
        [
            "bash",
            str(JOBS / "audit_cross_source_exact_dedup.sbatch"),
        ],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "report.json").is_file()
    assert (output / "exact_duplicate_removals.jsonl.zst").is_file()
    assert not os.access(output / "report.json", os.W_OK)
