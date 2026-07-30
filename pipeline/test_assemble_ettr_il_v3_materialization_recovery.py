from __future__ import annotations

import hashlib
from pathlib import Path
import stat

import ettr_il_v2_token_native_surface

from accelerate_ettr_il_v3_materialization import materialize_task_parallel
from assemble_ettr_il_v3_materialization_recovery import (
    ASSEMBLY_SCHEMA,
    RecoveryAssemblyError,
    assemble_recovery,
)
from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH
from ettr_il_v3_protocol import canonical_json_bytes
from freeze_ettr_il_v3_protocol import build_freeze
from materialize_ettr_il_v3_corpus import (
    audit_materialization,
    build_task_manifest,
)
from test_materialize_ettr_il_v3_corpus import _selected_root


ADAPTER = Path(__file__).with_name("accelerate_ettr_il_v3_materialization.py")
ASSEMBLER = Path(__file__).with_name(
    "assemble_ettr_il_v3_materialization_recovery.py"
)
JOB = (
    Path(__file__).parent
    / "jobs"
    / "assemble_ettr_il_v3_materialization_recovery.sbatch"
)


def _fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    role: str = "main",
):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    monkeypatch.setattr(
        ettr_il_v2_token_native_surface,
        "DEFAULT_TOKENIZER_PATH",
        tmp_path / "developer-default-does-not-exist.json",
    )
    selected = _selected_root(
        tmp_path,
        role=role,
        split="train" if role == "main" else "confirmation",
    )
    source_root = Path(__file__).parent.parent
    freeze_value = build_freeze(
        source_root,
        (
            "pipeline/ettr_il_v3_protocol.py",
            "pipeline/materialize_ettr_il_v3_corpus.py",
        ),
        source_commit="d" * 40,
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_bytes(canonical_json_bytes(freeze_value))
    tasks = tmp_path / "tasks.json"
    build_task_manifest(
        selected,
        tasks,
        materializer_source_commit="d" * 40,
        materializer_freeze_sha256=str(freeze_value["freeze_sha256"]),
    )
    recovery_output = tmp_path / "recovery-output"
    recovery_reports = tmp_path / "recovery-reports"
    adapter_sha256 = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    confirmation_key = None
    if role == "sealed_confirmation":
        confirmation_key = tmp_path / "confirmation.key"
        confirmation_key.write_bytes(b"k" * 32)
        confirmation_key.chmod(0o400)
    materialize_task_parallel(
        tasks,
        selected,
        recovery_output,
        recovery_reports,
        tokenizer,
        source_root,
        freeze,
        task_index=0,
        workers=2,
        execution_source_commit="e" * 40,
        execution_code_sha256=adapter_sha256,
        execution_code=ADAPTER,
        confirmation_key_file=confirmation_key,
    )
    return (
        tokenizer,
        source_root,
        freeze,
        tasks,
        recovery_output,
        recovery_reports,
        adapter_sha256,
    )


def test_recovery_assembly_is_read_only_and_passes_global_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        tokenizer,
        source_root,
        freeze,
        tasks,
        recovery_output,
        recovery_reports,
        adapter_sha256,
    ) = _fixture(tmp_path, monkeypatch)
    assembly_sha256 = hashlib.sha256(ASSEMBLER.read_bytes()).hexdigest()
    output = tmp_path / "assembled-output"
    reports = tmp_path / "assembled-reports"
    receipt_path = tmp_path / "assembly.json"
    receipt = assemble_recovery(
        tasks,
        tmp_path / "unused-primary-output",
        tmp_path / "unused-primary-reports",
        output,
        reports,
        receipt_path,
        {0: (recovery_output, recovery_reports)},
        execution_source_commit="f" * 40,
        execution_code_sha256=assembly_sha256,
        expected_adapter_source_commit="e" * 40,
        expected_adapter_sha256=adapter_sha256,
        execution_code=ASSEMBLER,
    )
    assert receipt["schema"] == ASSEMBLY_SCHEMA
    assert receipt["replacement_indices"] == [0]
    assert receipt["role"] == "main"
    for path in (*output.rglob("*"), *reports.rglob("*"), receipt_path):
        assert not path.is_symlink()
        if path.is_file():
            assert path.stat().st_nlink == 1
        assert not path.stat().st_mode & stat.S_IWUSR
    audit = audit_materialization(
        tasks,
        output,
        reports,
        tokenizer,
        source_root,
        freeze,
        tmp_path / "audit.json",
    )
    assert audit["status"] == "pass"
    assert audit["core_rows"] == 1


def test_confirmation_recovery_assembly_passes_global_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        tokenizer,
        source_root,
        freeze,
        tasks,
        recovery_output,
        recovery_reports,
        adapter_sha256,
    ) = _fixture(tmp_path, monkeypatch, role="sealed_confirmation")
    assembly_sha256 = hashlib.sha256(ASSEMBLER.read_bytes()).hexdigest()
    output = tmp_path / "assembled-output"
    reports = tmp_path / "assembled-reports"
    receipt_path = tmp_path / "assembly.json"
    receipt = assemble_recovery(
        tasks,
        tmp_path / "unused-primary-output",
        tmp_path / "unused-primary-reports",
        output,
        reports,
        receipt_path,
        {0: (recovery_output, recovery_reports)},
        execution_source_commit="f" * 40,
        execution_code_sha256=assembly_sha256,
        expected_adapter_source_commit="e" * 40,
        expected_adapter_sha256=adapter_sha256,
        execution_code=ASSEMBLER,
    )
    assert receipt["schema"] == ASSEMBLY_SCHEMA
    assert receipt["role"] == "sealed_confirmation"
    assert receipt["replacement_indices"] == [0]
    audit = audit_materialization(
        tasks,
        output,
        reports,
        tokenizer,
        source_root,
        freeze,
        tmp_path / "audit.json",
    )
    assert audit["status"] == "pass"
    assert audit["role"] == "sealed_confirmation"
    assert audit["core_rows"] == 1


def test_recovery_assembly_rejects_wrong_adapter_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (
        _,
        _,
        _,
        tasks,
        recovery_output,
        recovery_reports,
        _,
    ) = _fixture(tmp_path, monkeypatch)
    assembly_sha256 = hashlib.sha256(ASSEMBLER.read_bytes()).hexdigest()
    try:
        assemble_recovery(
            tasks,
            tmp_path / "unused-primary-output",
            tmp_path / "unused-primary-reports",
            tmp_path / "assembled-output",
            tmp_path / "assembled-reports",
            tmp_path / "assembly.json",
            {0: (recovery_output, recovery_reports)},
            execution_source_commit="f" * 40,
            execution_code_sha256=assembly_sha256,
            expected_adapter_source_commit="e" * 40,
            expected_adapter_sha256="0" * 64,
            execution_code=ASSEMBLER,
        )
    except RecoveryAssemblyError:
        pass
    else:
        raise AssertionError("wrong recovery adapter identity was accepted")


def test_recovery_assembly_job_is_hash_bound_cpu_only_and_fresh() -> None:
    text = JOB.read_text()
    assert "SHA256SUMS" in text
    assert "--execution-code-sha256" in text
    assert "--expected-adapter-sha256" in text
    assert "RECOVERY_LAYOUT" in text
    assert '|| -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT"' in text
    assert '|| -e "$REPORTS_ROOT" || -L "$REPORTS_ROOT"' in text
    assert '|| -e "$RECEIPT" || -L "$RECEIPT"' in text
    assert "--gres" not in text
    assert "CUDA" not in text
