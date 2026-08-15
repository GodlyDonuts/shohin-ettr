import hashlib
import json

import numpy as np
import pytest

import select_q36_mtr_nested_control_forest as module


def _write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n")


def test_answer_features_are_finite_and_bounded():
    assert module._answer_features(None) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert module._answer_features("choice") == (1.0, 0.06, 0.0, 0.0, 0.0)
    values = module._answer_features("-1e10000")
    assert values[:4] == (1.0, 0.08, 1.0, 1.0)
    assert values[4] == 5.0


def test_threshold_retains_conservatively_on_tie():
    predictions = np.asarray([0.0, 0.1, 0.2])
    control = np.asarray([1, 0, 1])
    forest = np.asarray([0, 1, 0])
    threshold, correct = module._choose_threshold(predictions, control, forest)
    assert correct == 2
    assert threshold == 0.195


def test_control_score_is_bound_to_exact_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CONTROL_SHARDS", 1)
    monkeypatch.setattr(module, "ROWS", 1)
    identity = "a" * 64
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-candidate-v1",
                "arm": "unchanged",
                "identity_sha256": identity,
                "task": "math500",
                "completion": "The answer is 2.",
                "generated_tokens": 8,
                "max_token_exhausted": False,
            }
        )
        + "\n"
    )
    score = tmp_path / "score.json"
    _write_json(
        score,
        {
            "schema": "shohin-q36-mtr-draft-preview-v1",
            "status": "complete",
            "evaluation_arm": "unchanged",
            "split": "development",
            "candidates_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "outcomes": [
                {
                    "identity_sha256": identity,
                    "task": "math500",
                    "correct": True,
                    "explicit_final_answer": True,
                    "max_token_exhausted": False,
                }
            ],
        },
    )
    candidates, outcomes = module._load_control([candidate], [score])
    assert candidates[identity]["completion"] == "The answer is 2."
    assert outcomes[identity]["correct"] is True

    payload = json.loads(score.read_text())
    payload["outcomes"][0]["max_token_exhausted"] = True
    _write_json(score, payload)
    with pytest.raises(module.Q36MTRNestedControlForestError):
        module._load_control([candidate], [score])
