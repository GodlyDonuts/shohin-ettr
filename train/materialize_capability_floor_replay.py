#!/usr/bin/env python3
"""Seal split-safe replay matrices for the frozen capability-floor campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Mapping, Sequence

from capability_floor_interface import (
    build_interface_contract,
    validate_interface_contract,
)
from capability_floor_replay import (
    ReplayScheduleConfig,
    build_candidate_replay_matrix,
    candidate_replay_matrix_sha256,
    load_candidate_replay_rectangles,
)


PUBLICATION_SCHEMA = "shohin-ettr-capability-floor-replay-publication-v1"


class CapabilityFloorReplayPublicationError(RuntimeError):
    """The replay publication is incomplete or would overwrite evidence."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_replay_publication(
    *,
    index_path: Path,
    index_sha256: str,
    candidates: Sequence[str],
    interface_contract: Mapping[str, object],
) -> dict[str, object]:
    optimizer = interface_contract.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise CapabilityFloorReplayPublicationError("optimizer contract differs")
    component_strata = optimizer.get("component_strata")
    seed_pairs = optimizer.get("seed_pairs")
    if not isinstance(component_strata, Mapping) or not isinstance(seed_pairs, list):
        raise CapabilityFloorReplayPublicationError("replay contract differs")
    component_updates = optimizer.get("component_updates_per_seed")
    composition_updates = optimizer.get("composition_updates_per_seed")
    if (
        not isinstance(component_updates, int)
        or isinstance(component_updates, bool)
        or component_updates <= 0
        or not isinstance(composition_updates, int)
        or isinstance(composition_updates, bool)
        or composition_updates <= 0
    ):
        raise CapabilityFloorReplayPublicationError("replay update budget differs")

    inventories = load_candidate_replay_rectangles(
        index_path,
        expected_sha256=index_sha256,
        candidates=candidates,
        split="train",
    )
    matrices = []
    for seed_pair in seed_pairs:
        if (
            not isinstance(seed_pair, list)
            or len(seed_pair) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in seed_pair
            )
        ):
            raise CapabilityFloorReplayPublicationError("replay seed pair differs")
        architecture_seed, data_seed = seed_pair
        for component, strata in component_strata.items():
            if (
                not isinstance(component, str)
                or not isinstance(strata, list)
                or any(not isinstance(value, str) for value in strata)
            ):
                raise CapabilityFloorReplayPublicationError("component strata differ")
            updates = (
                composition_updates
                if component == "autonomous-composition"
                else component_updates
            )
            config = ReplayScheduleConfig(
                component=component,
                required_strata=tuple(strata),
                updates=updates,
                seed=data_seed,
                dataset_sha256=index_sha256,
            )
            matrix = build_candidate_replay_matrix(inventories, config)
            matrices.append(
                {
                    "architecture_seed": architecture_seed,
                    "component": component,
                    "data_seed": data_seed,
                    "matrix": matrix,
                    "matrix_sha256": candidate_replay_matrix_sha256(matrix),
                    "updates": updates,
                }
            )

    payload: dict[str, object] = {
        "candidate_order": list(candidates),
        "cohort_index_sha256": index_sha256,
        "ettr_dense_schedule_identity": "byte-identical-within-candidate",
        "interface_contract_sha256": _sha256_bytes(
            _canonical_bytes(interface_contract)
        ),
        "matrices": matrices,
        "schema": PUBLICATION_SCHEMA,
        "split": "train",
        "status": "replay-frozen",
    }
    validate_replay_publication(payload)
    return payload


def validate_replay_publication(payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema") != PUBLICATION_SCHEMA
        or payload.get("status") != "replay-frozen"
        or payload.get("split") != "train"
        or payload.get("ettr_dense_schedule_identity")
        != "byte-identical-within-candidate"
    ):
        raise CapabilityFloorReplayPublicationError("replay publication differs")
    candidates = payload.get("candidate_order")
    matrices = payload.get("matrices")
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(set(candidates)) != len(candidates)
        or not isinstance(matrices, list)
        or not matrices
    ):
        raise CapabilityFloorReplayPublicationError("replay publication is empty")
    keys = set()
    for entry in matrices:
        if not isinstance(entry, Mapping):
            raise CapabilityFloorReplayPublicationError("replay matrix entry differs")
        matrix = entry.get("matrix")
        key = (
            entry.get("architecture_seed"),
            entry.get("data_seed"),
            entry.get("component"),
        )
        if key in keys or not isinstance(matrix, Mapping):
            raise CapabilityFloorReplayPublicationError("replay matrix identity differs")
        keys.add(key)
        if entry.get("matrix_sha256") != candidate_replay_matrix_sha256(matrix):
            raise CapabilityFloorReplayPublicationError("replay matrix digest differs")
        arm_hashes = matrix.get("arm_schedule_sha256")
        if not isinstance(arm_hashes, Mapping) or set(arm_hashes) != set(candidates):
            raise CapabilityFloorReplayPublicationError("replay arm inventory differs")
        for candidate in candidates:
            hashes = arm_hashes[candidate]
            if (
                not isinstance(hashes, Mapping)
                or hashes.get("ettr") != hashes.get("dense")
            ):
                raise CapabilityFloorReplayPublicationError(
                    "ETTR and dense schedule hashes differ"
                )
    for digest_name in ("cohort_index_sha256", "interface_contract_sha256"):
        digest = payload.get(digest_name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise CapabilityFloorReplayPublicationError(
                f"{digest_name} differs"
            )


def publication_sha256(payload: Mapping[str, object]) -> str:
    validate_replay_publication(payload)
    return _sha256_bytes(_canonical_bytes(payload))


def write_no_replace(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.partial")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as sink:
            descriptor = None
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise CapabilityFloorReplayPublicationError(
            "replay publication already exists"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    interface = build_interface_contract()
    validate_interface_contract(interface)
    payload = build_replay_publication(
        index_path=args.index,
        index_sha256=args.index_sha256,
        candidates=tuple(args.candidate),
        interface_contract=interface,
    )
    write_no_replace(args.output, _canonical_bytes(payload))
    print(publication_sha256(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
