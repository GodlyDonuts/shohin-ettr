import json

from merge_dset1_evaluation_shards import merge


def _report(index, member, identity):
    return {
        "schema": "shohin-dset1-span-edit-evaluation-v1",
        "status": "complete",
        "arm": "aligned",
        "model_root": "/model",
        "model_revision": "rev",
        "adapter_checkpoint_sha256": "checkpoint",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "shard_count": 2,
        "shard_index": index,
        "max_new_tokens": 32,
        "elapsed_seconds": 1.0,
        "generated_tokens": 1,
        "max_token_exhausted": 0,
        "peak_gpu_memory_bytes": 1,
        "results": [
            {
                "identity_sha256": identity,
                "pair_identity_sha256": "pair",
                "pair_member": member,
                "corruption_family": "numeric_final",
                "action_correct": True,
                "script_exact": True,
                "execution_correct": True,
                "trajectory_exact": True,
                "execution_error": None,
            }
        ],
    }


def test_merge_complete_pair(tmp_path) -> None:
    paths = []
    for index, member in enumerate(("clean", "fault")):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(_report(index, member, member)))
        paths.append(path)
    output = tmp_path / "merged.json"
    report = merge(paths, output)
    assert report["row_count"] == 2
    assert report["counterfactual_consistency"] == 1.0
    assert report["family_metrics"]["numeric_final"]["execution_correct_accuracy"] == 1.0
