import hashlib
import json

import pytest

from materialize_capability_floor_replay import (
    CapabilityFloorReplayPublicationError,
    build_replay_publication,
    publication_sha256,
    validate_replay_publication,
    write_no_replace,
)


def _cohort(tmp_path):
    candidates = ("candidate-a", "candidate-b")
    rows = []
    for split in ("train", "development"):
        rectangles = []
        for index in range(16):
            rectangles.append(
                {
                    "charged_positions": {
                        "candidate-a": 100 + index,
                        "candidate-b": 200 + index,
                    },
                    "rectangle_id": f"{split}-{index:03d}",
                    "strata": [
                        "COMMAND",
                        "COMMAND-factor",
                        "LINK",
                        "NONE",
                        "WORLD",
                        "WORLD-factor",
                        "WRITE",
                        "effect-family",
                    ],
                }
            )
        rows.append(
            {
                "accepted": True,
                "assessor_fields_in_model_input": False,
                "index_schema": "shohin-ettr-capability-floor-core-index-v2",
                "rectangles": rectangles,
                "split": split,
            }
        )
    content = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path = tmp_path / "cohort-index.jsonl"
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest(), candidates


def _interface():
    return {
        "optimizer": {
            "component_strata": {
                "oracle-program-executor": ["NONE", "WRITE", "LINK"],
                "oracle-state-query-reader": ["WORLD", "COMMAND"],
                "world-compiler-effect-binding": [
                    "WORLD-factor",
                    "effect-family",
                ],
                "autonomous-composition": [
                    "WORLD-factor",
                    "COMMAND-factor",
                ],
            },
            "component_updates_per_seed": 1,
            "composition_updates_per_seed": 2,
            "seed_pairs": [[31, 11], [32, 12]],
        }
    }


def test_publication_freezes_all_components_seeds_and_arm_identity(tmp_path) -> None:
    index, digest, candidates = _cohort(tmp_path)
    payload = build_replay_publication(
        index_path=index,
        index_sha256=digest,
        candidates=candidates,
        interface_contract=_interface(),
    )
    validate_replay_publication(payload)
    assert len(payload["matrices"]) == 8
    assert {entry["updates"] for entry in payload["matrices"]} == {1, 2}
    for entry in payload["matrices"]:
        for hashes in entry["matrix"]["arm_schedule_sha256"].values():
            assert hashes["ettr"] == hashes["dense"]
    assert len(publication_sha256(payload)) == 64


def test_publication_rejects_tamper_and_atomic_writer_rejects_overwrite(
    tmp_path,
) -> None:
    index, digest, candidates = _cohort(tmp_path)
    payload = build_replay_publication(
        index_path=index,
        index_sha256=digest,
        candidates=candidates,
        interface_contract=_interface(),
    )
    payload["matrices"][0]["matrix_sha256"] = "0" * 64
    with pytest.raises(CapabilityFloorReplayPublicationError, match="digest"):
        validate_replay_publication(payload)

    output = tmp_path / "publication.json"
    write_no_replace(output, b"first\n")
    assert output.read_bytes() == b"first\n"
    with pytest.raises(CapabilityFloorReplayPublicationError, match="exists"):
        write_no_replace(output, b"second\n")
    assert output.read_bytes() == b"first\n"
