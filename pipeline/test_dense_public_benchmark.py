from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

import analyze_dense_public_screen as analyze
import build_dense_public_benchmark_data as data
import restore_dense_public_model as restore
import reclaim_dense_public_model as reclaim
import score_dense_public_benchmark as score


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ranked_screen_is_deterministic_and_stratified() -> None:
    rows = [
        {
            "identity_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "kind": "a" if index < 75 else "b",
        }
        for index in range(100)
    ]
    selected = data.ranked_screen(rows, 20, "kind")
    assert selected == data.ranked_screen(list(reversed(rows)), 20, "kind")
    assert sum(row["kind"] == "a" for row in selected) == 15
    assert sum(row["kind"] == "b" for row in selected) == 5


def test_mmlu_and_musr_parsers_follow_official_terminal_answer() -> None:
    assert score.mmlu_answer("work A then the answer is (C)") == "C"
    assert score.mmlu_answer("Answer: B") == "B"
    assert score.musr_answer("ANSWER: (3)", 4, 1) == 3
    assert score.musr_answer("unparseable", 4, 1) == score.musr_answer(
        "unparseable", 4, 1
    )


def test_model_restoration_binds_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "config.json").write_text('{"model_type":"fixture"}\n')
    (upstream / "weights.bin").write_bytes(b"weights")

    def snapshot_download(**kwargs: object) -> str:
        target = Path(str(kwargs["local_dir"]))
        for source in upstream.iterdir():
            (target / source.name).write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    model = tmp_path / "model"
    receipt = tmp_path / "receipt.json"
    payload = restore.run(
        argparse.Namespace(
            repository="fixture/model",
            revision="a" * 40,
            config_sha256=_digest(upstream / "config.json"),
            model_root=model,
            receipt=receipt,
        )
    )
    assert payload["files"] == 2
    assert payload["manifest_verified"] is True
    assert json.loads(receipt.read_text())["tree_sha256"] == payload["tree_sha256"]
    assert not (model / ".cache").exists()

    reclaimed = reclaim.run(
        argparse.Namespace(
            model_root=model,
            model_receipt=receipt,
            output=tmp_path / "reclaim.json",
        )
    )
    assert reclaimed["tree_sha256"] == payload["tree_sha256"]
    assert reclaimed["locally_recoverable"] is False
    assert not model.exists()


def _score_payload(benchmark: str, delta: int = 6) -> dict[str, object]:
    unchanged = 100
    revision = unchanged + delta
    return {
        "schema": score.REPORT_SCHEMA,
        "status": "complete",
        "label": "prospective_256_row_screen_not_full_benchmark",
        "host": "qwen9",
        "benchmark": benchmark,
        "rows": 256,
        "model_revision": "r",
        "draft_checkpoint_sha256": "d" * 64,
        "revision_checkpoint_sha256": "e" * 64,
        "metrics": {
            "unchanged_correct": unchanged,
            "trained_revision_correct": revision,
            "paired_delta_count": delta,
            "paired_delta_points": 100 * delta / 256,
            "wins": max(delta, 0),
            "losses": max(-delta, 0),
            "baseline_correct_retention": 1.0,
        },
    }


def test_analysis_promotes_only_the_frozen_three_benchmark_gate(tmp_path: Path) -> None:
    paths = []
    for benchmark in analyze.BENCHMARKS:
        path = tmp_path / f"{benchmark}.json"
        path.write_text(json.dumps(_score_payload(benchmark)))
        paths.append(path)
    payload = analyze.run(paths, tmp_path / "analysis.json")
    assert payload["combined"]["delta_count"] == 18
    assert payload["promote_to_full_confirmations"] is True


def test_analysis_stops_on_material_benchmark_regression(tmp_path: Path) -> None:
    paths = []
    for benchmark, delta in zip(analyze.BENCHMARKS, (10, 10, -4), strict=True):
        path = tmp_path / f"{benchmark}.json"
        path.write_text(json.dumps(_score_payload(benchmark, delta)))
        paths.append(path)
    payload = analyze.run(paths, tmp_path / "analysis.json")
    assert payload["promote_to_full_confirmations"] is False
    assert payload["stop_this_host_after_screen"] is True
