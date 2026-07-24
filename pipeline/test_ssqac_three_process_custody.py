from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline.ssqac_three_process_custody import (
    CustodyError,
    run_three_process_custody,
    verify_three_process_custody,
)


def _rewrite_json(path: Path, value: object) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)


def test_three_process_custody_passes_and_proves_serial_deletion(tmp_path: Path) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    receipt = verify_three_process_custody(run.root)

    assert run.receipt_path.stat().st_mode & 0o222 == 0
    assert receipt["mechanics_only"] is True
    assert receipt["reasoning_claim"] is False
    assert receipt["network_access"] is False
    assert receipt["process_order"] == ["compiler", "candidate", "assessor"]
    assert len({entry["pid"] for entry in receipt["processes"]}) == 3
    assert all(entry["returncode"] == 0 for entry in receipt["processes"])
    assert not (run.root / "compiler_workspace").exists()
    assert receipt["compiler"]["source_unlinked_before_candidate"] is True
    assert receipt["compiler"]["workspace_unlinked_before_candidate"] is True
    assert (
        receipt["compiler"]["deletion_completed_ns"]
        <= receipt["processes"][1]["parent_observed_start_ns"]
    )
    assert (
        receipt["processes"][1]["parent_observed_end_ns"]
        <= receipt["assessor"]["created_ns"]
    )


def test_candidate_receives_only_sealed_artifact_runtime_and_manifest(
    tmp_path: Path,
) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    receipt = run.receipt
    assert receipt["candidate"]["data_payload_files"] == [
        "primitive_runtime.json",
        "sealed_algebra.json",
    ]
    assert receipt["candidate"]["control_manifest_file"] == "manifest.json"
    assert receipt["candidate"]["delivered_files"] == [
        "manifest.json",
        "primitive_runtime.json",
        "sealed_algebra.json",
    ]
    assert receipt["candidate"]["read_files"] == [
        "input/manifest.json",
        "input/primitive_runtime.json",
        "input/sealed_algebra.json",
    ]
    assert receipt["candidate"]["forbidden_scan"][
        "forbidden_filenames_absent"
    ] is True
    assert receipt["candidate"]["forbidden_scan"]["forbidden_content_absent"] is True
    assert receipt["candidate"]["forbidden_scan"]["raw_source_bytes_absent"] is True
    assert receipt["candidate"]["assessor_workspace_absent_through_exit"] is True


def test_final_mechanics_result_is_trivial_and_not_a_reasoning_claim(
    tmp_path: Path,
) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    result = json.loads(
        (run.root / "candidate/output/candidate_result.json").read_text(
            encoding="ascii"
        )
    )
    assessment = json.loads(
        (run.root / "assessor/output/assessment.json").read_text(encoding="ascii")
    )
    assert result["result"] == {
        "executed_primitives": 1,
        "halted": True,
        "tensor": [[1, 0], [0, 1]],
    }
    assert result["reasoning_claim"] is False
    assert assessment["passed"] is True
    assert assessment["reasoning_claim"] is False
    assert run.receipt["boundary"]["promotion_eligible"] is False
    assert run.receipt["boundary"]["kernel_namespace_isolation_claimed"] is False


@pytest.mark.parametrize("role", ["compiler", "candidate", "assessor"])
def test_subprocess_failure_fails_closed(tmp_path: Path, role: str) -> None:
    with pytest.raises(CustodyError, match=rf"{role} subprocess failed"):
        run_three_process_custody(
            tmp_path / role,
            _fault_role=role,
        )


def test_extra_candidate_file_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase == "candidate_staged":
            (directory / "extra.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(CustodyError, match="closed-world file set differs"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_candidate_symlink_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase != "candidate_staged":
            return
        target = directory / "primitive_runtime.json"
        target.unlink()
        target.symlink_to(directory / "sealed_algebra.json")

    with pytest.raises(CustodyError, match="symlink"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_altered_candidate_hash_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase != "candidate_staged":
            return
        path = directory / "sealed_algebra.json"
        value = json.loads(path.read_text(encoding="ascii"))
        value["tensor"][0][0] = 99
        _rewrite_json(path, value)

    with pytest.raises(CustodyError, match="manifest hash or size differs"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_forbidden_candidate_filename_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase == "candidate_staged":
            (directory / "gold_verifier.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(CustodyError, match="closed-world file set differs"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_forbidden_candidate_content_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase != "candidate_staged":
            return
        path = directory / "sealed_algebra.json"
        value = json.loads(path.read_text(encoding="ascii"))
        value["query_answer"] = 17
        _rewrite_json(path, value)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        payload = path.read_bytes()
        import hashlib

        manifest["files"]["sealed_algebra.json"] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        _rewrite_json(manifest_path, manifest)

    with pytest.raises(CustodyError, match="forbidden content"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_extra_assessor_file_fails_closed(tmp_path: Path) -> None:
    def inject(phase: str, directory: Path) -> None:
        if phase == "assessor_staged":
            (directory / "extra.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(CustodyError, match="closed-world file set differs"):
        run_three_process_custody(tmp_path / "custody", _phase_hook=inject)


def test_post_run_file_tamper_is_rejected(tmp_path: Path) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    result_path = run.root / "candidate/output/candidate_result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))
    result["result"]["tensor"][0][0] = 2
    _rewrite_json(result_path, result)
    with pytest.raises(CustodyError):
        verify_three_process_custody(run.root)


def test_post_run_extra_file_is_rejected(tmp_path: Path) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    extra = run.root / "unexpected.json"
    extra.write_text("{}\n", encoding="ascii")
    extra.chmod(0o444)
    with pytest.raises(CustodyError, match="final tree file closure differs"):
        verify_three_process_custody(run.root)


def test_post_run_symlink_is_rejected(tmp_path: Path) -> None:
    run = run_three_process_custody(tmp_path / "custody")
    link = run.root / "leak"
    os.symlink(run.root / "candidate/input/sealed_algebra.json", link)
    with pytest.raises(CustodyError, match="invalid file in final tree"):
        verify_three_process_custody(run.root)


def test_nonempty_root_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    (root / "existing").write_text("not empty", encoding="ascii")
    with pytest.raises(CustodyError, match="absent or an empty"):
        run_three_process_custody(root)
