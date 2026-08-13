from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from build_q36_mtr_custody import EVIDENCE_SCHEMA
from compare_q36_mtr import CUSTODY_SCHEMA, OUTPUT_SCHEMA
from mirror_q36_mtr_evidence import sha256_file
from q36_mtr_contract import graph_payload
from seal_q36_mtr_terminal_evidence import (
    Q36MTRTerminalEvidenceError,
    seal,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> argparse.Namespace:
    graph = _write(tmp_path / "graph.json", graph_payload("a" * 40))
    custody = _write(
        tmp_path / "custody.json",
        {
            "schema": CUSTODY_SCHEMA,
            "status": "complete",
            "run_id": "run",
            "custody_verified": True,
        },
    )
    preterminal = _write(
        tmp_path / "preterminal.json",
        {
            "schema": EVIDENCE_SCHEMA,
            "status": "complete",
            "verified": True,
            "run_id": "run",
            "source_commit": "a" * 40,
        },
    )
    result = _write(
        tmp_path / "result.json",
        {
            "schema": OUTPUT_SCHEMA,
            "status": "complete",
            "formal_result": "PASS",
            "gate_pass": True,
            "run_id": "run",
            "stop_after_gate": True,
            "automatic_retry_authorized": False,
            "automatic_confirmation_authorized": False,
            "automatic_successor_authorized": False,
            "holdout_access_authorized": False,
            "product_access_authorized": False,
            "next_action": "stop_and_preserve_evidence",
            "inputs": {
                "final_custody": {"sha256": sha256_file(custody)},
                "graph_contract": {"sha256": sha256_file(graph)},
            },
        },
    )
    authorized = tmp_path / "authorized"
    authorized.mkdir()
    return argparse.Namespace(
        graph_contract=graph,
        preterminal_evidence=preterminal,
        final_custody=custody,
        final_result=result,
        authorized_root=authorized,
        output_root=authorized / "terminal",
    )


def test_q36_terminal_result_is_copied_in_final_cpu_job(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    receipt = seal(args)
    assert receipt["formal_result"] == "PASS"
    assert receipt["stop_after_gate"] is True
    assert receipt["successor_authorized"] is False
    assert (args.output_root / "manifest.json").is_file()


def test_q36_terminal_evidence_rejects_successor_authorization(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    result = json.loads(args.final_result.read_text(encoding="utf-8"))
    result["automatic_successor_authorized"] = True
    args.final_result.write_text(json.dumps(result) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRTerminalEvidenceError):
        seal(args)
