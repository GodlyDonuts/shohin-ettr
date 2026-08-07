#!/usr/bin/env python3
"""Focused source-disjoint confirmation tests for DIVERGE-EAL2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from diverge_eal2_data import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_EPISODES,
    build_development_episode,
    build_evaluation_episode,
    validate_episode,
)


DEVELOPMENT_PUBLIC_SHA256 = (
    "9303b2a0a543491f8c829799d5476ea3387f9b58fd876729771658f058cb24a1"
)
DEVELOPMENT_ASSESSOR_SHA256 = (
    "bdb997cbac845fdbc096a9558d5e3d0843e79b207bd6ea4d691179f7d2c59d22"
)


def _serialized_sha256(index: int) -> str:
    digest = hashlib.sha256()
    for serial in range(DEVELOPMENT_EPISODES):
        row = build_development_episode(serial)[index]
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _fake_report(seed: int) -> dict[str, object]:
    score = {
        "complete_exact": 6144,
        "total": 6144,
        "complete_exact_rate": 1.0,
    }
    execution = {
        "state_exact": 4096,
        "programs": 4096,
        "query_exact": 8192,
        "queries": 8192,
    }
    return {
        "status": "pass",
        "source_commit": "confirmation-test",
        "checkpoint_sha256": "1" * 64,
        "reader_state_sha256": "2" * 64,
        "data": {"public": f"seed_{seed}_public.jsonl"},
        "reader": {
            "normal": score,
            "counterfactual": score,
            "temporal_scrub": {
                "complete_exact": 1500,
                "total": 6144,
                "complete_exact_rate": 1500 / 6144,
            },
        },
        "execution": {
            "learned": execution,
            "shuffled_episode_evidence": {
                **execution,
                "state_exact": 0,
                "query_exact": 64,
            },
            "unrelated_law_transplant": {
                **execution,
                "state_exact": 0,
                "query_exact": 64,
            },
        },
        "gate": {"conditions": {"all": True}, "passed": True},
    }


def main() -> None:
    assert _serialized_sha256(0) == DEVELOPMENT_PUBLIC_SHA256
    assert _serialized_sha256(1) == DEVELOPMENT_ASSESSOR_SHA256
    occupied_sources: set[str] = set()
    occupied_names: set[str] = set()
    occupied_identities: set[str] = set()
    for seed in CONFIRMATION_SEEDS:
        pairs = [build_evaluation_episode(serial, seed=seed) for serial in range(4)]
        public = [pair[0] for pair in pairs]
        assessor = [pair[1] for pair in pairs]
        for visible, hidden in zip(public, assessor, strict=True):
            validate_episode(visible, hidden)
        sources = {
            item["source_sha256"] for episode in public for item in episode["evidence"]
        }
        names = {
            value
            for episode in public
            for value in (*episode["aliases"], *episode["registers"])
        }
        identities = {episode["identity_sha256"] for episode in public}
        assert not (sources & occupied_sources)
        assert not (names & occupied_names)
        assert not (identities & occupied_identities)
        occupied_sources |= sources
        occupied_names |= names
        occupied_identities |= identities

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        arguments = []
        for seed in CONFIRMATION_SEEDS:
            path = root / f"seed_{seed}.json"
            path.write_text(json.dumps(_fake_report(seed), sort_keys=True) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            arguments.extend(("--result", str(seed), str(path), digest))
        output = root / "aggregate.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("aggregate_diverge_eal2_confirmation.py")),
                *arguments,
                "--output",
                str(output),
                "--source-commit",
                "confirmation-test",
            ],
            check=True,
        )
        aggregate = json.loads(output.read_text())
        assert aggregate["status"] == "pass"
        assert aggregate["aggregate"]["normal"]["rate"] == 1.0
        assert aggregate["aggregate"]["learned_execution"]["query_rate"] == 1.0
    print("diverge EAL2 confirmation tests passed")


if __name__ == "__main__":
    main()
