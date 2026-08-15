from __future__ import annotations

import hashlib

import score_q36_mtr_external_forest as module


def _identity() -> str:
    return hashlib.sha256(b"external-forest").hexdigest()


def _candidates(answers):
    identity = _identity()
    return {
        arm: {
            identity: {
                "schema": "shohin-q36-mtr-candidate-v1",
                "arm": arm,
                "identity_sha256": identity,
                "task": "math500",
                "completion": rf"\boxed{{{answer}}}",
                "generated_tokens": 10 + index,
                "max_token_exhausted": False,
            }
        }
        for index, (arm, answer) in enumerate(zip(module.ARMS, answers, strict=True))
    }


def test_feature_vector_is_finite_and_tracks_arm_agreement():
    candidates = _candidates(("1", "2", "2", "3", "2"))
    identity = _identity()
    revision = module.feature_vector(identity, "revision", candidates)
    unchanged = module.feature_vector(identity, "unchanged", candidates)
    assert len(revision) == 41
    assert len(unchanged) == 41
    assert revision != unchanged
    assert all(value == value for value in revision)


def test_retention_threshold_keeps_interpolation_without_clear_advantage():
    candidates = _candidates(("1", "2", "2", "3", "2"))
    identity = _identity()
    predictions = {(identity, arm): 0.5 for arm in module.ARMS}
    predictions[(identity, "revision")] = 0.53
    assert module._choose(identity, predictions, candidates) == "interpolation"
    predictions[(identity, "revision")] = 0.54
    assert module._choose(identity, predictions, candidates) == "revision"


def test_development_interpolation_accepts_exact_legacy_draft_schema(
    tmp_path, monkeypatch
):
    row = _candidates(("1", "2", "2", "3", "2"))["interpolation"][_identity()]
    row["schema"] = "shohin-q36-mtr-model-draft-v1"
    row.pop("arm")
    path = tmp_path / "candidates.jsonl"
    import json

    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(module, "TASKS", ("math500",))
    loaded = module.load_candidate_group(
        "interpolation", [path], 1, 1, development=True
    )
    assert tuple(loaded) == (_identity(),)
