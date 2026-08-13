from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from q36_mtr_contract import (
    ARMS,
    EXCLUDED_NODES,
    MODEL_REVISION,
    PROHIBITED_RETRIES,
    Q36MTRContractError,
    SOURCE_SHA256,
    STAGES,
    graph_payload,
    main,
    validate_graph,
)

COMMIT = "1" * 40


def test_exact_no_submit_graph_is_acyclic_and_resource_complete() -> None:
    payload = graph_payload(COMMIT)
    assert payload["model"]["revision"] == MODEL_REVISION
    assert payload["h100_requests"] == 61
    assert payload["expected_h100_hours"] == 58.9
    assert payload["maximum_concurrent_single_h100_requests"] == 32
    assert payload["arms"] == list(ARMS)
    assert payload["excluded_nodes"] == list(EXCLUDED_NODES)
    assert payload["prohibited_retries"] == list(PROHIBITED_RETRIES)
    assert payload["data"]["source_sha256"] == SOURCE_SHA256
    assert payload["scientific_submit_authorized"] is False
    assert payload["model_acquisition_authorized"] is False
    assert payload["data_materialization_authorized"] is False
    assert payload["automatic_retry"] is False
    assert payload["automatic_confirmation"] is False
    assert payload["automatic_successor"] is False
    assert payload["requeue"] is False
    seen: set[str] = set()
    ordered: list[str] = []
    for stage in payload["stages"]:
        assert set(stage["dependencies"]) <= seen
        seen.add(stage["name"])
        ordered.append(stage["name"])
        assert stage["h100_per_task"] in {0, 1}
    assert tuple(ordered) == tuple(stage.name for stage in STAGES)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("scientific_submit_authorized", True),
        lambda value: value.__setitem__("requeue", True),
        lambda value: value.__setitem__("h100_requests", 62),
        lambda value: value.__setitem__("expected_h100_hours", 58.91),
        lambda value: value.__setitem__("maximum_concurrent_single_h100_requests", 16),
        lambda value: value["model"].__setitem__("revision", "2" * 40),
        lambda value: value["excluded_nodes"].pop(),
        lambda value: value["prohibited_retries"].pop(),
        lambda value: value["arms"].remove("draft_hidden"),
        lambda value: value["stages"][3].__setitem__("dependencies", ["final_compare"]),
        lambda value: value["data"]["source_sha256"].__setitem__("code", "0" * 64),
        lambda value: value["data"].__setitem__("split_seed", 2026080812),
        lambda value: value["training"].__setitem__("revision_updates", 257),
        lambda value: value["minimum_storage"].__setitem__("free_inodes", 149999),
        lambda value: value.__setitem__("one_output_per_identity", False),
        lambda value: value.__setitem__("partition", "preemptable"),
    ],
)
def test_graph_mutations_fail_closed(mutation) -> None:
    payload = copy.deepcopy(graph_payload(COMMIT))
    mutation(payload)
    with pytest.raises(Q36MTRContractError):
        validate_graph(payload)


@pytest.mark.parametrize("commit", ["", "a" * 39, "A" * 40, "g" * 40])
def test_source_commit_is_exact_lower_hex(commit: str) -> None:
    with pytest.raises(Q36MTRContractError):
        graph_payload(commit)


def test_cli_publishes_once_and_revalidates(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "graph.json"
    monkeypatch.setattr(
        "sys.argv",
        ["q36_mtr_contract.py", "--source-commit", COMMIT, "--output", str(output)],
    )
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_graph(payload)
    with pytest.raises(Q36MTRContractError):
        main()
    monkeypatch.setattr(
        "sys.argv",
        ["q36_mtr_contract.py", "--source-commit", COMMIT, "--check", str(output)],
    )
    assert main() == 0
