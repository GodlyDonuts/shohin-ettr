import json
from pathlib import Path

import pytest

from pipeline.audit_v3_sensitive_content import audit_sensitive_content
from pipeline.materialize_sensitive_residual import (
    SensitiveResidualError,
    materialize_sensitive_residual,
)
from pipeline.test_audit_v3_sensitive_content import _corpus
from pipeline.tokenize_shards import sha256_file
from pipeline.verify_tokenized_shards import verify_manifest


SELECTION_CODE = Path(__file__).with_name("materialize_sensitive_residual.py")
JOB = (
    Path(__file__).parent / "jobs" / "materialize_sensitive_residual.sbatch"
).read_text()


def _prepare(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, source_selection = _corpus(tmp_path)
    audit = tmp_path / "audit"
    audit_sensitive_content(
        corpus_dir=source,
        selection_code=source_selection,
        output_dir=audit,
    )
    return source, source_selection, audit


def test_sensitive_removals_are_applied_and_reverified(tmp_path: Path) -> None:
    source, source_selection, audit = _prepare(tmp_path)
    output = tmp_path / "residual"
    result = materialize_sensitive_residual(
        source_dir=source,
        source_selection_code=source_selection,
        audit_dir=audit,
        selection_code=SELECTION_CODE,
        output_dir=output,
        shard_tokens=1,
    )
    assert result["documents"] == 1
    assert result["dropped_documents"] == 2
    assert result["removed_category_documents"] == {
        "aws_access_key": 1,
        "credential_assignment": 1,
    }
    verification = verify_manifest(
        output,
        selection_code=SELECTION_CODE,
        require_external_inputs=True,
    )
    assert verification["document_rows"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dropped_sensitive_content"] == 2
    assert manifest["sensitive_content_residual"][
        "audit_report_payload_sha256"
    ] == json.loads((audit / "report.json").read_text())["payload_sha256"]


def test_tampered_findings_fail_before_output(tmp_path: Path) -> None:
    source, source_selection, audit = _prepare(tmp_path)
    findings = audit / "sensitive_findings.jsonl.zst"
    findings.chmod(0o600)
    findings.write_bytes(b"tampered")
    output = tmp_path / "residual"
    with pytest.raises(SensitiveResidualError, match="findings receipt"):
        materialize_sensitive_residual(
            source_dir=source,
            source_selection_code=source_selection,
            audit_dir=audit,
            selection_code=SELECTION_CODE,
            output_dir=output,
        )
    assert not output.exists()


def test_report_source_binding_fails_closed(tmp_path: Path) -> None:
    source, source_selection, audit = _prepare(tmp_path)
    report_path = audit / "report.json"
    report_path.chmod(0o600)
    report = json.loads(report_path.read_text())
    report["corpus"]["manifest_payload_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    output = tmp_path / "residual"
    with pytest.raises(SensitiveResidualError, match="audit contract"):
        materialize_sensitive_residual(
            source_dir=source,
            source_selection_code=source_selection,
            audit_dir=audit,
            selection_code=SELECTION_CODE,
            output_dir=output,
        )
    assert not output.exists()


def test_source_selection_substitution_fails_closed(tmp_path: Path) -> None:
    source, source_selection, audit = _prepare(tmp_path)
    source_selection.write_text("substituted\n")
    assert sha256_file(source_selection) != json.loads(
        (source / "manifest.json").read_text()
    )["selection_code_sha256"]
    with pytest.raises(Exception, match="selection code"):
        materialize_sensitive_residual(
            source_dir=source,
            source_selection_code=source_selection,
            audit_dir=audit,
            selection_code=SELECTION_CODE,
            output_dir=tmp_path / "residual",
        )


def test_sensitive_residual_job_is_hash_bound_and_cpu_only() -> None:
    assert "SHA256SUMS" in JOB
    assert 'export PYTHONPATH="$SOURCE_ROOT"' in JOB
    assert "--source-selection-code" in JOB
    assert "--audit-dir" in JOB
    assert "--selection-code" in JOB
    assert "--output-dir" in JOB
    assert '"$OUT/documents.jsonl.zst"' in JOB
    assert 'find "$OUT" -type f -exec chmod 0444 {} +' in JOB
    assert 'find "$OUT" -type d -exec chmod 0555 {} +' in JOB
    assert "--gres" not in JOB
    assert "CUDA" not in JOB
