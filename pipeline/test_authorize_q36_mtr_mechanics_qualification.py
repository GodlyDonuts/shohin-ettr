from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import authorize_q36_mtr_mechanics_qualification as module
from authorize_q36_mtr_mechanics_qualification import (
    Q36MTRMechanicsQualificationError,
    authorize,
    sha256_file,
    verify_authorization,
)
from q36_mtr_roles import MODEL_REVISION

COMMIT = "a" * 40


def _manifest(root: Path, members: dict[str, str]) -> Path:
    root.mkdir()
    rows = []
    for relative, content in members.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rows.append(f"{sha256_file(path)}  {relative}")
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")
    return manifest


def _terminal(path: Path, run_id: str, source_commit: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": module.TERMINAL_SCHEMA,
                "status": "terminal_infrastructure_failure",
                "run_id": run_id,
                "source_commit": source_commit,
                "formal_scientific_result": None,
                "capability_rows_scored": 0,
                "development_assessor_reads": 0,
                "sealed_holdout_accesses": 0,
                "protected_product_accesses": 0,
                "public_accesses": 0,
                "automatic_retry_authorized": False,
                "automatic_successor_authorized": False,
                "stop_after_terminal": True,
                "infrastructure_diagnosis": {
                    "class": "fixture",
                    "scientific_gate_entered": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    runtime = tmp_path / "runtime"
    runtime_manifest = _manifest(
        runtime,
        {
            "runtime.json": json.dumps(
                {
                    "schema": "shohin-q36-mtr-runtime-v1",
                    "status": "complete",
                    "source_commit": COMMIT,
                    "model_acquisition_capability": False,
                }
            )
            + "\n",
            "train/hf_q36_mtr_mechanics.py": "mechanics\n",
            "train/jobs/q36_mtr_mechanics_qualification.sbatch": "wrapper\n",
        },
    )
    model = tmp_path / "model"
    model_manifest = _manifest(
        model,
        {
            "config.json": "{}\n",
            "SOURCE_REVISION": MODEL_REVISION + "\n",
        },
    )
    monkeypatch.setattr(
        module, "MODEL_CONFIG_SHA256", sha256_file(model / "config.json")
    )
    monkeypatch.setattr(module, "MODEL_MANIFEST_SHA256", sha256_file(model_manifest))
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-environment-v1",
                "status": "pass",
                "model_revision": MODEL_REVISION,
                "model_config_sha256": sha256_file(model / "config.json"),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "environment_tree_sha256": "b" * 64,
                "scientific_rows_read": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    b1 = tmp_path / "b1.jsonl"
    b1.write_text("b1\n", encoding="utf-8")
    monkeypatch.setattr(module, "B1_SHA256", sha256_file(b1))
    terminals = [
        _terminal(tmp_path / "terminal-first.json", "first", "1" * 40),
        _terminal(tmp_path / "terminal-second.json", "second", "2" * 40),
    ]
    monkeypatch.setattr(
        module, "PRIOR_TERMINAL_SHA256S", {sha256_file(path) for path in terminals}
    )
    return argparse.Namespace(
        run_id="q36-mechanics-qualification-r1",
        source_commit=COMMIT,
        runtime_root=runtime,
        runtime_manifest=runtime_manifest,
        model_root=model,
        model_manifest=model_manifest,
        environment_receipt=environment,
        b1=b1,
        prior_terminal=terminals,
        output_root=tmp_path / "qualification-output",
        output=tmp_path / "authorization.json",
    )


def _verify(args: argparse.Namespace, digest: str | None = None):
    return verify_authorization(
        args.output,
        digest or sha256_file(args.output),
        COMMIT,
        args.run_id,
        args.output_root,
        args.runtime_manifest,
        args.model_manifest,
        args.environment_receipt,
        args.b1,
        args.prior_terminal,
    )


def test_authorization_is_mechanics_only_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    result = authorize(args)
    assert result["status"] == "authorized"
    assert result["h100_allocations_authorized"] == 1
    assert result["mechanics_qualification_authorized"] is True
    assert result["scientific_graph_authorized"] is False
    assert result["capability_scoring_authorized"] is False
    assert result["assessor_access_authorized"] is False
    assert result["submission_capability"] is False
    assert len(result["prior_terminal_receipts"]) == 2
    assert json.loads(args.output.read_text()) == result
    assert args.output.stat().st_mode & 0o222 == 0
    with pytest.raises(Q36MTRMechanicsQualificationError, match="exists"):
        authorize(args)


def test_runtime_verifier_binds_identity_and_fresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorize(args)
    digest = sha256_file(args.output)
    result = _verify(args, digest)
    assert result["one_shot"] is True
    args.output_root.mkdir()
    with pytest.raises(Q36MTRMechanicsQualificationError, match="output root"):
        _verify(args, digest)


def test_verifier_rejects_writable_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorize(args)
    args.output.chmod(0o644)
    with pytest.raises(Q36MTRMechanicsQualificationError):
        _verify(args)


def test_terminal_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    terminal = args.prior_terminal[0]
    value = json.loads(terminal.read_text())
    value["capability_rows_scored"] = 1
    terminal.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRMechanicsQualificationError):
        authorize(args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    "field",
    (
        "scientific_graph_authorized",
        "capability_scoring_authorized",
        "assessor_access_authorized",
        "automatic_retry_authorized",
        "automatic_successor_authorized",
        "submission_capability",
    ),
)
def test_verifier_rejects_privilege_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorize(args)
    value = json.loads(args.output.read_text())
    value[field] = True
    args.output.chmod(0o644)
    args.output.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRMechanicsQualificationError):
        _verify(args)


def test_verifier_rejects_post_authorization_terminal_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorize(args)
    terminal = args.prior_terminal[0]
    value = json.loads(terminal.read_text())
    value["run_id"] = "substituted"
    terminal.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRMechanicsQualificationError):
        _verify(args)


@pytest.mark.parametrize(
    "field",
    ("runtime_manifest", "model_manifest", "environment_receipt", "b1"),
)
def test_verifier_rejects_custody_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorize(args)
    original = getattr(args, field)
    substitute = tmp_path / f"substitute-{original.name}"
    substitute.write_bytes(original.read_bytes())
    values = {
        "runtime_manifest": args.runtime_manifest,
        "model_manifest": args.model_manifest,
        "environment_receipt": args.environment_receipt,
        "b1": args.b1,
    }
    values[field] = substitute
    with pytest.raises(Q36MTRMechanicsQualificationError):
        verify_authorization(
            args.output,
            sha256_file(args.output),
            COMMIT,
            args.run_id,
            args.output_root,
            values["runtime_manifest"],
            values["model_manifest"],
            values["environment_receipt"],
            values["b1"],
            args.prior_terminal,
        )


def test_wrapper_is_one_h100_no_score_no_dispatch() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "train/jobs/q36_mtr_mechanics_qualification.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert 'authorize_q36_mtr_mechanics_qualification.py" verify' in source
    assert "q36_require_authorization" not in source
    assert "ASSESSOR" not in source
    assert "score_completion" not in source
    assert "sbatch " not in source
    assert source.index("export OMP_NUM_THREADS=1") < source.index(
        'authorize_q36_mtr_mechanics_qualification.py" verify'
    )
