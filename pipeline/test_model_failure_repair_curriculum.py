import hashlib

from build_model_failure_repair_curriculum import build, preference_pairs
from materialize_verified_mbpp_anchor_board import materialize


def _raw() -> dict:
    return {
        "task_id": 1,
        "text": "Increment an integer.",
        "code": "def inc(x):\n    return x + 1",
        "test_setup_code": "",
        "test_list": ["assert inc(2) == 3"],
    }


def _anchor() -> dict:
    return {
        "question": "Increment an integer.",
        "response": "def inc(x):\n    return x + 1",
        "source": "mbpp_train",
    }


def test_materializes_exact_executed_anchor() -> None:
    rows = materialize([_raw()], [_anchor()], timeout_seconds=2)
    assert rows[0]["task"] == "mbpp"
    assert rows[0]["test_list"] == ["assert inc(2) == 3"]
    assert rows[0]["source"] == "mbpp_train"


def test_builds_only_actual_model_failures() -> None:
    board = materialize([_raw()], [_anchor()], timeout_seconds=2)
    identity = hashlib.sha256(b"mbpp\0Increment an integer.").hexdigest()
    evaluation = {
        "status": "complete",
        "task": "mbpp",
        "total": 1,
        "correct": 0,
        "results": [
            {
                "identity_sha256": identity,
                "correct": False,
                "completion": "def inc(x):\n    return x",
                "execution": {"returncode": 1, "stderr": "AssertionError"},
            }
        ],
    }
    rows = build(board, evaluation)
    assert len(rows) == 1
    assert "Previous solution:\ndef inc(x)" in rows[0]["question"]
    assert "AssertionError" in rows[0]["question"]
    assert rows[0]["response"] == _raw()["code"]
    assert rows[0]["rejected_response"] == "def inc(x):\n    return x"

    pairs = preference_pairs(rows)
    assert pairs == [
        {
            "schema": "shohin-product-verifier-preference-pairs-v1",
            "question": rows[0]["question"],
            "chosen": _raw()["code"],
            "rejected": "def inc(x):\n    return x",
            "source_identity_sha256": identity,
            "verification": "model_failure_vs_execution_verified_gold_v1",
        }
    ]
