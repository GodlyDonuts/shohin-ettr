import json
from pathlib import Path

from merge_dseo1_evaluation_shards import merge


def _report(index: int, identity: str, pair: str) -> dict:
    member = "clean" if index == 0 else "fault"
    return {
        "schema": "shohin-dseo1-paired-evaluation-v1",
        "status": "complete",
        "arm": "aligned",
        "model_root": "/model",
        "model_revision": "rev",
        "adapter_checkpoint_sha256": "checkpoint",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "shard_index": index,
        "shard_count": 2,
        "max_new_tokens": 768,
        "elapsed_seconds": 2.0,
        "generated_tokens": 3,
        "max_token_exhausted": 0,
        "peak_gpu_memory_bytes": 10,
        "results": [
            {
                "identity_sha256": identity,
                "pair_identity_sha256": pair,
                "pair_member": member,
                "corruption_family": "numeric_final",
                "predicted_action": "<KEEP>" if member == "clean" else "<FIX_FINAL>",
                "action_correct": True,
                "answer_correct": True,
            }
        ],
    }


def test_merge_dseo1_shards_preserves_pair_metrics(tmp_path: Path) -> None:
    paths = []
    for index, identity in enumerate(("a", "b")):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(_report(index, identity, "pair")))
        paths.append(path)
    report = merge(paths, tmp_path / "merged.json")
    assert report["pair_count"] == 1
    assert report["action_accuracy"] == 1.0
    assert report["counterfactual_consistency"] == 1.0
