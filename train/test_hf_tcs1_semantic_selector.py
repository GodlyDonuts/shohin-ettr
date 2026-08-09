from collections import OrderedDict

from hf_tcs1_semantic_selector import selection_metrics


def test_semantic_selection_metrics_preserve_whole_candidates() -> None:
    groups = {
        "a": [
            {"task": "math500", "lineage": "depth1", "correct": False},
            {"task": "math500", "lineage": "depth2", "correct": True},
            {"task": "math500", "lineage": "direct", "correct": False},
        ],
        "b": [
            {"task": "mbpp", "lineage": "depth1", "correct": True},
            {"task": "mbpp", "lineage": "depth2", "correct": False},
            {"task": "mbpp", "lineage": "direct", "correct": False},
        ],
    }
    scores = [[0.0, 2.0, 1.0], [3.0, 2.0, 1.0]]
    metrics = selection_metrics(OrderedDict(groups), scores)
    assert metrics["buckets"]["overall"]["selected_correct"] == 2
    assert metrics["buckets"]["overall"]["repaired"] == 1
    permuted = selection_metrics(OrderedDict(groups), scores, rotate_contents=True)
    assert permuted["buckets"]["overall"]["selected_correct"] == 0
