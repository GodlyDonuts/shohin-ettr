from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from build_q36_mtr_custody import EVIDENCE_SCHEMA
from compare_q36_mtr import CUSTODY_SCHEMA, OUTPUT_SCHEMA
from mirror_q36_mtr_evidence import sha256_file
from q36_mtr_contract import graph_payload
from q36_mtr_evidence import verify_evidence_snapshot
from seal_q36_mtr_terminal_evidence import (
    Q36MTRTerminalEvidenceError,
    seal,
)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> argparse.Namespace:
    graph = _write(tmp_path / "graph.json", graph_payload("a" * 40))
    preterminal_root = tmp_path / "preterminal"
    artifacts = preterminal_root / "artifacts"
    artifacts.mkdir(parents=True)
    mirrored = artifacts / "graph_contract.json"
    mirrored.write_bytes(graph.read_bytes())
    digest = sha256_file(mirrored)
    record = {
        "name": "graph_contract",
        "primary": str(graph.resolve()),
        "mirror": str(mirrored.resolve()),
        "sha256": digest,
        "bytes": mirrored.stat().st_size,
    }
    tree_row = {"name": "graph_contract", "sha256": digest, "bytes": record["bytes"]}
    import hashlib

    tree_digest = hashlib.sha256(
        (json.dumps(tree_row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    preterminal_payload = {
        "schema": EVIDENCE_SCHEMA,
        "status": "complete",
        "verified": True,
        "run_id": "run",
        "source_commit": "a" * 40,
        "artifact_sha256s": {"graph_contract": digest},
        "artifact_count": 1,
        "artifact_tree_sha256": tree_digest,
        "records": [record],
        "primary_mirror_hashes_exact": True,
        "write_once_snapshot": True,
    }
    preterminal = _write(preterminal_root / "manifest.json", preterminal_payload)
    mirrored.chmod(0o444)
    artifacts.chmod(0o555)
    preterminal_root.chmod(0o555)
    assert (
        verify_evidence_snapshot(preterminal, preterminal_payload)[
            "artifact_tree_sha256"
        ]
        == tree_digest
    )
    custody = _write(
        tmp_path / "custody.json",
        {
            "schema": CUSTODY_SCHEMA,
            "status": "complete",
            "run_id": "run",
            "custody_verified": True,
            "evidence_mirror_tree_sha256": tree_digest,
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
