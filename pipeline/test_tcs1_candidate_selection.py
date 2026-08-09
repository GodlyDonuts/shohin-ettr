from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from build_tcs1_candidate_sets import build
from tcs1_shape_selector import FEATURE_NAMES, feature_vector, run


def _candidate(completion: str, correct: bool) -> dict:
    return {
        "completion": completion,
        "correct": correct,
        "generated_tokens": len(completion.split()),
        "max_token_exhausted": False,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_builder_preserves_disjoint_three_candidate_geometry(tmp_path: Path) -> None:
    train_ids = [f"{index:064x}" for index in range(5824)]
    dev_ids = [f"{index + 10000:064x}" for index in range(1289)]
    pairs = []
    train_depth = []
    for identity in train_ids:
        pairs.append(
            {
                "identity_sha256": identity,
                "split": "train",
                "task": "math500",
                "question": "What is 1+1?",
                "candidates": [_candidate("2", True), _candidate("3", False)],
            }
        )
        train_depth.append(
            {"identity_sha256": identity, "task": "math500", **_candidate("2", True)}
        )
    dev_sources = {name: [] for name in ("depth1", "depth2", "direct")}
    for identity in dev_ids:
        pairs.append(
            {
                "identity_sha256": identity,
                "split": "development",
                "task": "bbh_logic",
                "question": "Choose A or B.",
                "candidates": [_candidate("A", True), _candidate("B", False)],
            }
        )
        for index, name in enumerate(dev_sources):
            dev_sources[name].append(
                {
                    "identity_sha256": identity,
                    "task": "bbh_logic",
                    **_candidate(str(index), index == 0),
                }
            )
    pair_path = tmp_path / "pairs.jsonl"
    _write(pair_path, pairs)
    train_path = tmp_path / "train_depth.jsonl"
    _write(train_path, train_depth)
    dev_paths = {}
    for name, rows in dev_sources.items():
        dev_paths[name] = tmp_path / f"{name}.jsonl"
        _write(dev_paths[name], rows)
    output = tmp_path / "out"
    report = build(
        SimpleNamespace(
            pairs=pair_path,
            train_depth_one=train_path,
            development_depth_one=dev_paths["depth1"],
            development_depth_two=dev_paths["depth2"],
            development_direct=dev_paths["direct"],
            output=output,
        )
    )
    assert report["source_disjoint"] is True
    assert report["outputs"]["train"]["candidates"] == 17472
    assert report["outputs"]["development"]["candidates"] == 3867


def test_shape_features_ignore_forbidden_metadata() -> None:
    row = {
        "completion": "Therefore the final answer is \\boxed{4}.",
        "question": "What is 2+2?",
        "generated_tokens": 10,
        "max_token_exhausted": False,
        "lineage": "depth1",
        "task": "math500",
        "gold": "4",
        "correct": True,
    }
    changed = {
        **row,
        "lineage": "direct",
        "task": "mbpp",
        "gold": "wrong",
        "correct": False,
    }
    assert len(feature_vector(row)) == len(FEATURE_NAMES)
    assert feature_vector(row) == feature_vector(changed)


def test_shape_selector_runs_on_disjoint_groups(tmp_path: Path) -> None:
    def rows(split: str, start: int, count: int) -> list[dict]:
        result = []
        for value in range(start, start + count):
            identity = f"{value:064x}"
            for index, correct in enumerate((True, False, False)):
                result.append(
                    {
                        "schema": "shohin-tcs1-candidate-v1",
                        "split": split,
                        "identity_sha256": identity,
                        "task": "math500",
                        "question": "What is 2+2?",
                        "sample_index": index,
                        "lineage": ("depth1", "depth2", "direct")[index],
                        "completion": "4" if correct else f"wrong {index}",
                        "correct": correct,
                        "generated_tokens": 1 + index,
                        "max_token_exhausted": False,
                    }
                )
        return result

    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    _write(train, rows("train", 20000, 60))
    _write(development, rows("development", 30000, 12))
    report = run(
        SimpleNamespace(
            train=train,
            development=development,
            output=tmp_path / "selection",
            seed=2026080902,
            width=16,
            epochs=5,
            patience=3,
        )
    )
    assert report["metrics"]["buckets"]["overall"]["total"] == 12
    assert report["shuffled_label_control"]["buckets"]["overall"]["total"] == 12
