from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ettr_il_v2_token_native_surface
import pytest

from accelerate_ettr_il_v3_materialization import (
    ADAPTER_SCHEMA,
    CONFIRMATION_ADAPTER_SCHEMA,
    materialize_task_parallel,
)
from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH
from ettr_il_v3_protocol import canonical_json_bytes
from freeze_ettr_il_v3_protocol import build_freeze
from materialize_ettr_il_v3_corpus import (
    AUDIT_SCHEMA,
    audit_materialization,
    build_task_manifest,
    materialize_task,
)
from test_materialize_ettr_il_v3_corpus import _selected_root


ADAPTER = Path(__file__).with_name("accelerate_ettr_il_v3_materialization.py")
JOB = Path(__file__).parent / "jobs" / "accelerate_ettr_il_v3_materialization.sbatch"


def test_parallel_adapter_is_byte_identical_to_serial(tmp_path: Path, monkeypatch) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    monkeypatch.setattr(
        ettr_il_v2_token_native_surface,
        "DEFAULT_TOKENIZER_PATH",
        tmp_path / "developer-default-does-not-exist.json",
    )
    selected = _selected_root(tmp_path)
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

    serial_output = tmp_path / "serial-output"
    serial_reports = tmp_path / "serial-reports"
    materialize_task(
        tasks,
        selected,
        serial_output,
        serial_reports,
        tokenizer,
        source_root,
        freeze,
        task_index=0,
    )
    parallel_output = tmp_path / "parallel-output"
    parallel_reports = tmp_path / "parallel-reports"
    code_sha256 = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    report = materialize_task_parallel(
        tasks,
        selected,
        parallel_output,
        parallel_reports,
        tokenizer,
        source_root,
        freeze,
        task_index=0,
        workers=2,
        execution_source_commit="e" * 40,
        execution_code_sha256=code_sha256,
        execution_code=ADAPTER,
    )
    relative_output = "train/horn-atomic_transactions.jsonl.gz"
    assert (parallel_output / relative_output).read_bytes() == (
        serial_output / relative_output
    ).read_bytes()
    serial_report = json.loads((serial_reports / "task-00000.json").read_bytes())
    assert report["output_sha256"] == serial_report["output_sha256"]
    assert report["uncompressed_bytes"] == serial_report["uncompressed_bytes"]
    assert report["execution_adapter"] == {
        "schema": ADAPTER_SCHEMA,
        "source_commit": "e" * 40,
        "source_sha256": code_sha256,
        "workers": 2,
    }
    audit = audit_materialization(
        tasks,
        parallel_output,
        parallel_reports,
        tokenizer,
        source_root,
        freeze,
        tmp_path / "parallel-audit.json",
    )
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["status"] == "pass"
    assert audit["core_rows"] == 1


def test_parallel_confirmation_is_byte_identical_and_key_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(DEFAULT_TOKENIZER_PATH.read_bytes())
    monkeypatch.setattr(
        ettr_il_v2_token_native_surface,
        "DEFAULT_TOKENIZER_PATH",
        tmp_path / "developer-default-does-not-exist.json",
    )
    selected = _selected_root(
        tmp_path,
        name="confirmation-selected",
        role="sealed_confirmation",
        split="confirmation",
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
    key = tmp_path / "confirmation.key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o400)

    serial_output = tmp_path / "serial-output"
    serial_reports = tmp_path / "serial-reports"
    materialize_task(
        tasks,
        selected,
        serial_output,
        serial_reports,
        tokenizer,
        source_root,
        freeze,
        task_index=0,
        confirmation_key_file=key,
    )
    parallel_output = tmp_path / "parallel-output"
    parallel_reports = tmp_path / "parallel-reports"
    code_sha256 = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    report = materialize_task_parallel(
        tasks,
        selected,
        parallel_output,
        parallel_reports,
        tokenizer,
        source_root,
        freeze,
        task_index=0,
        workers=2,
        execution_source_commit="e" * 40,
        execution_code_sha256=code_sha256,
        execution_code=ADAPTER,
        confirmation_key_file=key,
    )
    relative_output = "confirmation/horn-atomic_transactions.jsonl.gz"
    assert (parallel_output / relative_output).read_bytes() == (
        serial_output / relative_output
    ).read_bytes()
    assert report["execution_adapter"] == {
        "role": "sealed_confirmation",
        "schema": CONFIRMATION_ADAPTER_SCHEMA,
        "source_commit": "e" * 40,
        "source_sha256": code_sha256,
        "workers": 2,
    }
    encoded_report = json.dumps(report)
    assert "confirmation_key" not in encoded_report
    assert (b"k" * 32).hex() not in encoded_report
    audit = audit_materialization(
        tasks,
        parallel_output,
        parallel_reports,
        tokenizer,
        source_root,
        freeze,
        tmp_path / "parallel-audit.json",
    )
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["status"] == "pass"
    assert audit["core_rows"] == 1

    with pytest.raises(ValueError, match="confirmation key file is required"):
        materialize_task_parallel(
            tasks,
            selected,
            tmp_path / "missing-key-output",
            tmp_path / "missing-key-reports",
            tokenizer,
            source_root,
            freeze,
            task_index=0,
            workers=2,
            execution_source_commit="e" * 40,
            execution_code_sha256=code_sha256,
            execution_code=ADAPTER,
        )


def test_parallel_job_is_isolated_hash_bound_and_cpu_only() -> None:
    text = JOB.read_text()
    assert "SHA256SUMS" in text
    assert 'export PYTHONPATH="$SOURCE_ROOT/pipeline"' in text
    assert "--execution-source-commit" in text
    assert "--execution-code-sha256" in text
    assert "--confirmation-key-file" in text
    assert '|| -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT"' in text
    assert '|| -e "$REPORTS_ROOT" || -L "$REPORTS_ROOT"' in text
    assert "--gres" not in text
    assert "CUDA" not in text
