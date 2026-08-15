from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import score_upward_moe_temporal_gate as module
from hf_product_reasoning_eval import GENERATED_ONLY_SEQUENCE_CONTRACT
from hf_upward_moe_train_temporal_gate import host_spec


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(module, "ROWS", 2)
    monkeypatch.setattr(module, "SHARDS", 2)
    monkeypatch.setattr(module, "TASKS", ("math500",))
    spec = host_spec("nemotron-super")
    identities = ["1" * 64, "2" * 64]
    assessors = _jsonl(
        tmp_path / "assessors.jsonl",
        [
            {
                "schema": module.ASSESSOR_SCHEMA,
                "identity_sha256": identity,
                "split": "confirmation",
                "task": "math500",
                "assessor": {
                    "identity_sha256": identity,
                    "task": "math500",
                },
            }
            for identity in identities
        ],
    )
    patterns = {
        "unchanged": ("correct", "wrong"),
        "self_refinement": ("correct", "correct"),
        "owner": ("correct", "wrong"),
        "aligned_revision": ("correct", "wrong"),
        "temporal_gate": ("correct", "correct"),
    }
    candidates = []
    reports = []
    for arm in module.ARMS:
        for index, identity in enumerate(identities):
            candidate = _jsonl(
                tmp_path / f"{arm}-{index}.jsonl",
                [
                    {
                        "schema": module.CANDIDATE_SCHEMA,
                        "host": spec.host,
                        "arm": arm,
                        "identity_sha256": identity,
                        "task": "math500",
                        "completion": patterns[arm][index],
                        "generated_tokens": 1,
                        "max_token_exhausted": False,
                    }
                ],
            )
            report = _json(
                tmp_path / f"{arm}-{index}.report.json",
                {
                    "schema": module.EVALUATION_REPORT_SCHEMA,
                    "status": "complete",
                    "host": spec.host,
                    "host_contract": spec.receipt(),
                    "model_receipt": {"manifest": "a" * 64},
                    "role_lineage": {"warm_start_exact": True},
                    "arm": arm,
                    "split": "development",
                    "data_sha256": "b" * 64,
                    "mechanics_report_sha256": "c" * 64,
                    "generation_mode": "greedy",
                    "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
                    "max_new_tokens": 768,
                    "seed": 2026080816,
                    "batch_size": 1,
                    "shard_index": index,
                    "shard_count": 2,
                    "row_start": index,
                    "row_end": index + 1,
                    "full_row_count": 2,
                    "candidates_output": str(candidate.resolve()),
                    "candidates_sha256": _sha(candidate),
                    "counters": {"rows": 1},
                    "assessor_access_count": 0,
                    "development_labels_read": 0,
                    "sealed_access": {"holdout": 0, "product": 0, "public": 0},
                },
            )
            candidates.append([arm, str(candidate)])
            reports.append([arm, str(report)])
    monkeypatch.setattr(
        module, "qualify_allocation", lambda: {"probe_sha256": "d" * 64}
    )
    monkeypatch.setattr(module, "sandbox_atomic_json", lambda *_args: "e" * 64)
    monkeypatch.setattr(module, "qualify_mbpp_assessor_setups", lambda _rows: [])
    monkeypatch.setattr(
        module,
        "score_completion",
        lambda _assessor, completion: {"correct": completion == "correct"},
    )
    return SimpleNamespace(
        host="nemotron-super",
        assessors=assessors,
        candidates=candidates,
        reports=reports,
        sandbox_receipt=tmp_path / "sandbox.json",
        output=tmp_path / "score.json",
    )


def test_single_score_process_reports_paired_capability_and_retention(
    tmp_path: Path, monkeypatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    report = module.run(args)
    assert report["semantic_assessor_opens"] == 1
    assert report["arms"]["temporal_gate"]["correct"] == 2
    assert report["arms"]["unchanged"]["correct"] == 1
    assert report["paired_temporal_vs_controls"]["unchanged"]["net_correct"] == 1
    assert report["paired_temporal_vs_controls"]["aligned_revision"]["net_correct"] == 1
    assert report["unchanged_correct_retention"] == 1.0
    assert report["domain_correct_deltas_temporal_vs_controls"]["unchanged"] == {
        "math500": 1
    }
    assert report["arm_custody"]["temporal_gate"]["exact_identity_coverage"] is True
    assert args.output.is_file()


def test_cross_arm_candidate_tamper_fails_closed(tmp_path: Path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    arm, path = args.candidates[0]
    row = json.loads(Path(path).read_text())
    row["arm"] = "temporal_gate"
    _jsonl(Path(path), [row])
    report_path = Path(args.reports[0][1])
    report = json.loads(report_path.read_text())
    report["candidates_sha256"] = _sha(Path(path))
    _json(report_path, report)
    with pytest.raises(module.UpwardMoETemporalScoreError):
        module.run(args)


def test_exact_mcnemar_is_symmetric_and_bounded() -> None:
    assert module._mcnemar_exact(0, 0) == 1.0
    assert module._mcnemar_exact(7, 1) == module._mcnemar_exact(1, 7)
    assert 0.0 <= module._mcnemar_exact(7, 1) <= 1.0
