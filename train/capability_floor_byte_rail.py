#!/usr/bin/env python3
"""Lossless source-visible byte-role rail for capability-floor diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from capability_floor_corpus import EncodedSource
from capability_floor_feature_sufficiency import (
    _fit_probe,
    _load_feature_bundle,
    _save_feature_bundle,
    _select_records,
    _sha256_file,
    _source_tasks,
    _stack_state,
)
from capability_floor_sufficiency import tensor_sha256
from capability_floor_trajectory import UnifiedTrajectoryConfig
from ettr_il_v2_token_native_surface import CODEWORD_BYTES
from ettr_v3_streaming import ETTRV3StreamingRelease
from token_native_syntax_router import TokenNativeOperationRouter


BYTE_RAIL_SCHEMA = "shohin-ettr-capability-floor-canonical-byte-role-rail-v1"
ASCII_CARDINALITY = 128
BYTE_ROLE_WIDTH = CODEWORD_BYTES * ASCII_CARDINALITY
BYTE_RAIL_CANDIDATE = "shared-canonical-byte-role-rail-v1"


class CapabilityFloorByteRailError(ValueError):
    """The lossless public byte rail drifted or publication was unsafe."""


class _AsciiByteAdapter:
    def encode(self, text: str) -> EncodedSource:
        if not isinstance(text, str) or not text.isascii():
            raise CapabilityFloorByteRailError("byte rail source is not strict ASCII")
        result = EncodedSource(
            token_ids=tuple(text.encode("ascii")),
            offsets=tuple((index, index + 1) for index in range(len(text))),
        )
        result.validate(text_length=len(text), context_limit=1 << 20)
        return result


def byte_role_features(
    source_bytes: Sequence[int],
    role_masks: Sequence[Sequence[bool]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every byte jointly with its exact within-codeword position."""

    if not source_bytes or len(source_bytes) % CODEWORD_BYTES:
        raise CapabilityFloorByteRailError("byte rail source alignment differs")
    if len(role_masks) != 4:
        raise CapabilityFloorByteRailError("byte rail role geometry differs")
    result = torch.zeros(4, BYTE_ROLE_WIDTH, dtype=torch.bfloat16)
    present = torch.zeros(4, dtype=torch.bool)
    for role, mask in enumerate(role_masks):
        if len(mask) != len(source_bytes):
            raise CapabilityFloorByteRailError("byte rail role/source geometry differs")
        positions = [index for index, enabled in enumerate(mask) if enabled]
        if not positions:
            continue
        if len(positions) != CODEWORD_BYTES:
            raise CapabilityFloorByteRailError("byte rail role is not one public atom")
        offsets = sorted(position % CODEWORD_BYTES for position in positions)
        if offsets != list(range(CODEWORD_BYTES)):
            raise CapabilityFloorByteRailError("byte rail atom position differs")
        for position in positions:
            byte = int(source_bytes[position])
            if not 0 <= byte < ASCII_CARDINALITY:
                raise CapabilityFloorByteRailError("byte rail leaves ASCII")
            result[role, (position % CODEWORD_BYTES) * ASCII_CARDINALITY + byte] = 1
        present[role] = True
    if not present[0]:
        raise CapabilityFloorByteRailError("byte rail operation root is absent")
    return result, present


def _materialize_split(
    tasks: Sequence[object],
) -> dict[str, object]:
    features = []
    present_rows = []
    labels = []
    states = []
    orbit_ids = []
    sample_ids = []
    source_hashes = []
    for task in tasks:
        for operation, role_masks in enumerate(task.role_masks):
            role_features, present = byte_role_features(task.token_ids, role_masks)
            features.append(role_features)
            present_rows.append(present)
            labels.append(task.labels[operation])
            states.append(task.states[operation])
            orbit_ids.append(task.orbit_ids[operation])
            sample_ids.append(task.sample_ids[operation])
            source_hashes.append(task.source_sha256)
    if not features:
        raise CapabilityFloorByteRailError("byte rail split is empty")
    present = torch.stack(present_rows)
    role_masks = torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(
        len(features),
        -1,
        -1,
    )
    role_masks = role_masks & present.unsqueeze(-1)
    state = _stack_state(states)
    tensors = {
        "labels": torch.tensor(labels, dtype=torch.long),
        "role_masks": role_masks,
        "source_features": torch.stack(features),
        "source_mask": present,
        "state": {
            name: getattr(state, name)
            for name in (
                "value_probabilities",
                "type_probabilities",
                "relations",
                "active",
                "root",
                "committed",
            )
        },
    }
    identity = {
        "orbit_ids": orbit_ids,
        "sample_ids": sample_ids,
        "source_sha256": source_hashes,
    }
    split_sha256 = hashlib.sha256(
        json.dumps(
            {
                "identity": identity,
                "labels_sha256": tensor_sha256(tensors["labels"]),
                "role_masks_sha256": tensor_sha256(tensors["role_masks"]),
                "schema": BYTE_RAIL_SCHEMA,
                "source_features_sha256": tensor_sha256(
                    tensors["source_features"]
                ),
                "source_mask_sha256": tensor_sha256(tensors["source_mask"]),
                "state_sha256": {
                    name: tensor_sha256(value)
                    for name, value in tensors["state"].items()
                },
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "identity": identity,
        "split_sha256": split_sha256,
        "tensors": tensors,
    }


def materialize(args: argparse.Namespace) -> None:
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    stream.verify_source_shards()
    router = TokenNativeOperationRouter(
        stream.codec.codebook.token_ids,
        vocab_size=stream.codec.tokenizer.get_vocab_size(),
        maximum_positions=96,
        maximum_operations=6,
    )
    config = UnifiedTrajectoryConfig(input_width=BYTE_ROLE_WIDTH)
    splits = {}
    for split, count in (
        ("train", args.train_cores),
        ("development", args.development_cores),
    ):
        records = _select_records(stream, split, count=count, seed=args.selection_seed)
        tasks = _source_tasks(
            records,
            adapter=_AsciiByteAdapter(),
            codec=stream.codec,
            router=router,
            config=config,
        )
        splits[split] = _materialize_split(tasks)
    receipt = _save_feature_bundle(
        args.feature_bundle,
        candidate=BYTE_RAIL_CANDIDATE,
        checkpoint_sha256=hashlib.sha256(BYTE_RAIL_SCHEMA.encode("ascii")).hexdigest(),
        release_sha256=args.release_sha256,
        config=config,
        splits=splits,
    )
    print(json.dumps({"byte_rail_bundle": receipt}, sort_keys=True), flush=True)


def fit(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0).upper():
        raise CapabilityFloorByteRailError("byte rail fit requires one H100")
    if _sha256_file(args.feature_bundle) != args.feature_bundle_sha256:
        raise CapabilityFloorByteRailError("byte rail bundle SHA-256 differs")
    bundle = dict(_load_feature_bundle(args.feature_bundle, args.feature_bundle_sha256))
    if bundle.get("candidate") != BYTE_RAIL_CANDIDATE:
        raise CapabilityFloorByteRailError("byte rail bundle candidate differs")
    bundle["feature_bundle_sha256"] = args.feature_bundle_sha256
    _fit_probe(
        bundle,
        output=args.output,
        updates=args.updates,
        batch_size=args.probe_batch,
        seed=args.architecture_seed,
        learning_rate=args.learning_rate,
        device=torch.device("cuda", 0),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("materialize", "fit"), required=True)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--feature-bundle-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-cores", type=int, default=128)
    parser.add_argument("--development-cores", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=11)
    parser.add_argument("--architecture-seed", type=int, default=31)
    parser.add_argument("--probe-batch", type=int, default=16)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "materialize":
        required = (
            args.release_root,
            args.data_root,
            args.release_sha256,
            args.tokenizer,
        )
        if (
            any(value is None for value in required)
            or args.feature_bundle_sha256
            or args.output is not None
        ):
            raise CapabilityFloorByteRailError("byte rail materialization arguments differ")
        materialize(args)
    else:
        if not args.feature_bundle_sha256 or args.output is None:
            raise CapabilityFloorByteRailError("byte rail fit arguments differ")
        fit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
