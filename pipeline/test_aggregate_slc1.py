import json

import pytest

from aggregate_slc1 import SLC1AggregateError, aggregate


def report(control, identities, *, terminal=1):
    rows = len(identities)
    counts = {
        "rows": rows,
        "syntax_valid": rows,
        "canonical_exact": terminal,
        "record_count_exact": rows,
        "operation_sequence_exact": rows,
        "all_records_exact": terminal,
        "terminal_exact": terminal,
    }
    values = {**counts}
    return {
        "status": "complete",
        "control": control,
        "holdout_used": False,
        "checkpoint_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "counts": counts,
        "by_family": {"basic_arithmetic": values},
        "by_depth": {"5": values},
        "generated_tokens": rows,
        "exhausted": 0,
        "details": [{"identity_sha256": identity} for identity in identities],
    }


def write_shards(tmp_path, control, terminal):
    paths = []
    for shard in range(2):
        identities = [f"{index:064x}" for index in range(shard, 3917, 2)]
        path = tmp_path / f"{control}_{shard}.json"
        path.write_text(json.dumps(report(control, identities, terminal=len(identities) if terminal else 0)))
        paths.append(path)
    return paths


def test_aggregate_passes_complete_causal_fixture(tmp_path):
    normal = write_shards(tmp_path, "normal", True)
    shuffled = write_shards(tmp_path, "source_shuffled", False)
    result = aggregate(normal, shuffled)
    assert result["overall_pass"] is True
    assert result["terminal_causal_margin"] == 1.0


def test_aggregate_rejects_missing_identity(tmp_path):
    normal = write_shards(tmp_path, "normal", True)
    shuffled = write_shards(tmp_path, "source_shuffled", False)
    payload = json.loads(shuffled[0].read_text())
    payload["details"].pop()
    shuffled[0].write_text(json.dumps(payload))
    with pytest.raises(SLC1AggregateError, match="3,917"):
        aggregate(normal, shuffled)
