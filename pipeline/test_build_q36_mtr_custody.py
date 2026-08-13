from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from build_q36_mtr_custody import (
    PRECOMPUTE_SCHEMA,
    Q36MTRCustodyError,
    _manifest_tree,
    build_authorization,
    sha256_file,
)
from q36_mtr_contract import graph_payload
from score_q36_mtr import AUTHORIZATION_SCHEMA


def test_q36_manifest_tree_accepts_only_exact_hash_bound_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "member").write_text("value")
    digest = hashlib.sha256(b"value").hexdigest()
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{digest}  ./member\n")
    receipt = _manifest_tree(root, manifest)
    assert receipt["exact_membership"] is True
    (root / "extra").write_text("extra")
    with pytest.raises(Q36MTRCustodyError):
        _manifest_tree(root, manifest)


def test_q36_authorization_binds_exact_score_inputs_without_board_open(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload("a" * 40)) + "\n")
    names = {
        "application_report",
        "assessor_receipt",
        "commit_training_report",
        "data_report",
        "development_data",
        "draft_hidden_candidates",
        "draft_hidden_evaluation_report",
        "environment_receipt",
        "revision_candidates",
        "revision_report",
        "selections",
        "self_refinement_candidates",
        "self_refinement_report",
        "unchanged_candidates",
        "unchanged_report",
    }
    artifacts = {}
    for name in names:
        path = tmp_path / f"{name}.json"
        path.write_text(name + "\n")
        artifacts[name] = path
    precompute = {
        "schema": PRECOMPUTE_SCHEMA,
        "status": "complete",
        "run_id": "run",
        "graph_contract_sha256": sha256_file(graph_path),
        "identity_order_sha256": hashlib.sha256(b"identities").hexdigest(),
        "assessor_board_sha256": hashlib.sha256(b"board").hexdigest(),
        "artifact_sha256s": {
            name: sha256_file(path) for name, path in artifacts.items()
        },
    }
    precompute_path = tmp_path / "precompute.json"
    precompute_path.write_text(json.dumps(precompute) + "\n")
    output = tmp_path / "authorization.json"
    result = build_authorization(
        argparse.Namespace(
            precompute_custody=precompute_path,
            graph_contract=graph_path,
            artifact=[f"{name}={path}" for name, path in sorted(artifacts.items())],
            score_output_root=tmp_path / "score",
            output=output,
        )
    )
    assert result["schema"] == AUTHORIZATION_SCHEMA
    assert result["one_shot"] is True
    assert result["assessor_board_access_count_before"] == 0
    assert "assessor_board" not in result["input_hashes"]
