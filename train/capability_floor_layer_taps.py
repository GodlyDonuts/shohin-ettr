#!/usr/bin/env python3
"""Locate source syntax across frozen Shohin depth without vocabulary logits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from capability_floor_feature_sufficiency import (
    PROTECTED_CANDIDATE,
    REPORT_SCHEMA,
    _RawTokenizerAdapter,
    _fit_probe,
    _family_logits,
    _index_state,
    _load_protected_model,
    _renderer_metrics,
    _select_records,
    _sha256_file,
    _source_tasks,
    _stack_state,
    _state_from_payload,
    _torch_save_no_replace,
    _write_no_replace,
)
from capability_floor_corpus import TokenizerSpec
from capability_floor_sufficiency import tensor_sha256
from capability_floor_sufficiency import OperationFamilyTensorProbe
from capability_floor_trajectory import UnifiedTrajectoryConfig
from ettr_v3_streaming import ETTRV3StreamingRelease
from token_native_syntax_router import TokenNativeOperationRouter


TAP_SCHEMA = "shohin-ettr-capability-floor-layer-taps-v1"
TAP_BLOCKS = (0, 4, 9, 14, 19, 24, 29)
TAP_NAMES = ("embedding",) + tuple(f"block-{index:02d}" for index in TAP_BLOCKS)


class CapabilityFloorLayerTapError(ValueError):
    """The frozen layer-tap audit drifted or attempted an invalid publication."""


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


def _encode_taps(model: torch.nn.Module, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    if tokens.ndim != 2 or tokens.dtype != torch.long:
        raise CapabilityFloorLayerTapError("tap token tensor differs")
    if model.cfg.n_loop != 1 or model.cfg.n_layer <= max(TAP_BLOCKS):
        raise CapabilityFloorLayerTapError("tap backbone depth differs")
    length = tokens.shape[1]
    hidden = model.tok(tokens)
    result = {"embedding": hidden}
    cos = model.cos[:length].to(hidden.device)
    sin = model.sin[:length].to(hidden.device)
    for index, block in enumerate(model.blocks):
        hidden, _ = block(hidden, cos, sin)
        if index in TAP_BLOCKS:
            result[f"block-{index:02d}"] = (
                model.norm(hidden) if index == model.cfg.n_layer - 1 else hidden
            )
    if tuple(result) != TAP_NAMES:
        raise CapabilityFloorLayerTapError("tap inventory differs")
    return result


def pool_task_taps(
    taps: Mapping[str, torch.Tensor],
    *,
    row: int,
    source_length: int,
    role_masks: Sequence[Sequence[Sequence[bool]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool every declared public role at every frozen depth tap."""

    if tuple(taps) != TAP_NAMES or source_length <= 0:
        raise CapabilityFloorLayerTapError("tap pooling inventory differs")
    pooled_operations = []
    present_operations = []
    for operation_roles in role_masks:
        if len(operation_roles) != 4:
            raise CapabilityFloorLayerTapError("tap role geometry differs")
        role_tensor = torch.tensor(operation_roles, dtype=torch.bool)
        if role_tensor.shape != (4, source_length):
            raise CapabilityFloorLayerTapError("tap role/source geometry differs")
        present = role_tensor.any(-1)
        weights = role_tensor.to(torch.float32)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
        by_tap = []
        for name in TAP_NAMES:
            hidden = taps[name]
            if hidden.ndim != 3 or not 0 <= row < hidden.shape[0]:
                raise CapabilityFloorLayerTapError("tap hidden geometry differs")
            source = hidden[row, :source_length].float()
            if source.shape[0] != source_length:
                raise CapabilityFloorLayerTapError("tap source length differs")
            by_tap.append(torch.einsum("rt,tw->rw", weights, source))
        pooled_operations.append(torch.stack(by_tap).to(torch.bfloat16))
        present_operations.append(present)
    if not pooled_operations:
        raise CapabilityFloorLayerTapError("tap operation set is empty")
    return torch.stack(pooled_operations), torch.stack(present_operations)


def _materialize_tap_split(
    *,
    model: torch.nn.Module,
    tasks: Sequence[object],
    device: torch.device,
    inference_batch: int,
) -> dict[str, object]:
    if not tasks or inference_batch <= 0:
        raise CapabilityFloorLayerTapError("tap split geometry differs")
    features = []
    role_present = []
    labels = []
    states = []
    orbit_ids = []
    sample_ids = []
    source_hashes = []
    with torch.inference_mode():
        for start in range(0, len(tasks), inference_batch):
            chunk = tasks[start : start + inference_batch]
            length = max(len(task.token_ids) for task in chunk)
            tokens = torch.zeros(len(chunk), length, dtype=torch.long, device=device)
            for row, task in enumerate(chunk):
                tokens[row, : len(task.token_ids)] = torch.tensor(
                    task.token_ids,
                    dtype=torch.long,
                    device=device,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                taps = {
                    name: value.to(torch.bfloat16).cpu()
                    for name, value in _encode_taps(model, tokens).items()
                }
            for row, task in enumerate(chunk):
                pooled, present = pool_task_taps(
                    taps,
                    row=row,
                    source_length=len(task.token_ids),
                    role_masks=task.role_masks,
                )
                for operation in range(len(task.labels)):
                    features.append(pooled[operation])
                    role_present.append(present[operation])
                    labels.append(task.labels[operation])
                    states.append(task.states[operation])
                    orbit_ids.append(task.orbit_ids[operation])
                    sample_ids.append(task.sample_ids[operation])
                    source_hashes.append(task.source_sha256)
    state = _stack_state(states)
    tensors = {
        "role_features": torch.stack(features),
        "role_present": torch.stack(role_present),
        "labels": torch.tensor(labels, dtype=torch.long),
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
        _canonical_bytes(
            {
                "identity": identity,
                "labels_sha256": tensor_sha256(tensors["labels"]),
                "role_features_sha256": tensor_sha256(tensors["role_features"]),
                "role_present_sha256": tensor_sha256(tensors["role_present"]),
                "schema": TAP_SCHEMA,
                "state_sha256": {
                    name: tensor_sha256(value)
                    for name, value in tensors["state"].items()
                },
                "tap_names": list(TAP_NAMES),
            }
        )
    ).hexdigest()
    return {
        "identity": identity,
        "split_sha256": split_sha256,
        "tensors": tensors,
    }


def _save_tap_bundle(
    path: Path,
    *,
    checkpoint_sha256: str,
    release_sha256: str,
    config: UnifiedTrajectoryConfig,
    splits: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    payload = {
        "candidate": PROTECTED_CANDIDATE,
        "checkpoint_sha256": checkpoint_sha256,
        "config": asdict(config),
        "release_sha256": release_sha256,
        "schema": TAP_SCHEMA,
        "splits": splits,
        "tap_names": list(TAP_NAMES),
    }
    _torch_save_no_replace(path, payload)
    receipt = {
        "bytes": path.stat().st_size,
        "candidate": PROTECTED_CANDIDATE,
        "checkpoint_sha256": checkpoint_sha256,
        "path": path.name,
        "release_sha256": release_sha256,
        "schema": TAP_SCHEMA,
        "sha256": _sha256_file(path),
        "splits": {
            split: {
                "examples": int(value["tensors"]["labels"].numel()),
                "split_sha256": value["split_sha256"],
            }
            for split, value in splits.items()
        },
        "tap_names": list(TAP_NAMES),
    }
    _write_no_replace(
        path.with_suffix(path.suffix + ".receipt.json"),
        _canonical_bytes(receipt),
    )
    return receipt


def _load_tap_bundle(path: Path, expected_sha256: str) -> Mapping[str, object]:
    if _sha256_file(path) != expected_sha256:
        raise CapabilityFloorLayerTapError("tap bundle SHA-256 differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TAP_SCHEMA
        or payload.get("tap_names") != list(TAP_NAMES)
    ):
        raise CapabilityFloorLayerTapError("tap bundle contract differs")
    return payload


def virtual_feature_bundle(
    bundle: Mapping[str, object],
    *,
    tap_name: str,
    bundle_sha256: str,
) -> dict[str, object]:
    if tap_name not in TAP_NAMES:
        raise CapabilityFloorLayerTapError("tap selection differs")
    tap_index = TAP_NAMES.index(tap_name)
    splits = {}
    for split, value in bundle["splits"].items():
        tensors = value["tensors"]
        role_features = tensors["role_features"][:, tap_index]
        present = tensors["role_present"]
        roles = torch.eye(4, dtype=torch.bool).unsqueeze(0).expand(
            role_features.shape[0], -1, -1
        )
        roles = roles & present.unsqueeze(-1)
        if not present[:, 0].all():
            raise CapabilityFloorLayerTapError("tap root role is absent")
        splits[split] = {
            "identity": value["identity"],
            "split_sha256": hashlib.sha256(
                _canonical_bytes(
                    {
                        "parent_split_sha256": value["split_sha256"],
                        "schema": TAP_SCHEMA,
                        "tap": tap_name,
                    }
                )
            ).hexdigest(),
            "tensors": {
                "source_features": role_features,
                "source_mask": present,
                "role_masks": roles,
                "labels": tensors["labels"],
                "state": tensors["state"],
            },
        }
    return {
        "candidate": f"{bundle['candidate']}:{tap_name}",
        "config": bundle["config"],
        "feature_bundle_sha256": bundle_sha256,
        "splits": splits,
    }


def source_matched_world_swap_indices(
    split: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair byte-identical COMMAND rows whose WORLD changes the family."""

    identity = split["identity"]
    labels = split["tensors"]["labels"]
    sample_ids = identity["sample_ids"]
    source_hashes = identity["source_sha256"]
    if labels.ndim != 1 or len(sample_ids) != labels.numel():
        raise CapabilityFloorLayerTapError("binding-pair identity geometry differs")
    groups: dict[tuple[str, int, int, int], dict[int, int]] = {}
    for index, sample_id in enumerate(sample_ids):
        try:
            core_id, corner_text, operation_text, view_text = sample_id.rsplit(":", 3)
            corner = int(corner_text)
            operation = int(operation_text)
            if not view_text.startswith("view="):
                raise ValueError
            view = int(view_text.removeprefix("view="))
        except (AttributeError, TypeError, ValueError) as error:
            raise CapabilityFloorLayerTapError(
                "binding-pair sample identity differs"
            ) from error
        if not 0 <= corner < 4 or not 0 <= view < 4 or operation < 0:
            raise CapabilityFloorLayerTapError("binding-pair coordinate differs")
        world = corner // 2
        command = corner % 2
        key = (core_id, command, operation, view)
        by_world = groups.setdefault(key, {})
        if world in by_world:
            raise CapabilityFloorLayerTapError("binding-pair WORLD duplicate")
        by_world[world] = index
    source_indices = []
    swapped_indices = []
    for key in sorted(groups):
        by_world = groups[key]
        if set(by_world) != {0, 1}:
            raise CapabilityFloorLayerTapError("binding-pair WORLD incomplete")
        left = by_world[0]
        right = by_world[1]
        if source_hashes[left] != source_hashes[right]:
            raise CapabilityFloorLayerTapError("binding-pair COMMAND source differs")
        if int(labels[left]) == int(labels[right]):
            continue
        source_indices.extend((left, right))
        swapped_indices.extend((right, left))
    if not source_indices:
        raise CapabilityFloorLayerTapError("binding-pair causal subset is empty")
    source = torch.tensor(source_indices, dtype=torch.long)
    swapped = torch.tensor(swapped_indices, dtype=torch.long)
    if not torch.equal(source.sort().values, swapped.sort().values):
        raise CapabilityFloorLayerTapError("binding-pair swap is not bijective")
    return source, swapped


def _score_indices(
    probe: OperationFamilyTensorProbe,
    split: Mapping[str, object],
    source_indices: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tensors = split["tensors"]
    source_state = _state_from_payload(tensors["state"])
    predictions = []
    for start in range(0, source_indices.numel(), batch_size):
        selected = source_indices[start : start + batch_size]
        selected_state = state_indices[start : start + batch_size]
        state = _index_state(
            source_state,
            selected_state,
            device=device,
            dtype=torch.float32,
        )
        features = tensors["source_features"].index_select(0, selected).to(device)
        source_mask = tensors["source_mask"].index_select(0, selected).to(device)
        role_masks = tensors["role_masks"].index_select(0, selected).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = _family_logits(probe, features, source_mask, role_masks, state)
        predictions.append(logits.argmax(-1).cpu())
    return torch.cat(predictions)


def score_binding(args: argparse.Namespace) -> None:
    bundle = _load_tap_bundle(args.feature_bundle, args.feature_bundle_sha256)
    virtual = virtual_feature_bundle(
        bundle,
        tap_name=args.tap,
        bundle_sha256=args.feature_bundle_sha256,
    )
    if _sha256_file(args.probe_model) != args.probe_model_sha256:
        raise CapabilityFloorLayerTapError("binding probe model SHA-256 differs")
    try:
        model_payload = torch.load(
            args.probe_model,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        model_payload = torch.load(args.probe_model, map_location="cpu")
    if (
        not isinstance(model_payload, Mapping)
        or model_payload.get("schema") != REPORT_SCHEMA
        or model_payload.get("candidate") != virtual["candidate"]
    ):
        raise CapabilityFloorLayerTapError("binding probe model contract differs")
    config = UnifiedTrajectoryConfig(**dict(virtual["config"]))
    probe = OperationFamilyTensorProbe(config, max_roles=4)
    incompatible = probe.load_state_dict(model_payload["probe"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise CapabilityFloorLayerTapError("binding probe model state differs")
    device = torch.device("cuda", 0)
    probe.to(device).eval().requires_grad_(False)
    development = virtual["splits"]["development"]
    source_indices, swapped_indices = source_matched_world_swap_indices(development)
    clean = _score_indices(
        probe,
        development,
        source_indices,
        source_indices,
        device=device,
        batch_size=args.probe_batch,
    )
    swapped = _score_indices(
        probe,
        development,
        source_indices,
        swapped_indices,
        device=device,
        batch_size=args.probe_batch,
    )
    labels = development["tensors"]["labels"]
    clean_targets = labels.index_select(0, source_indices)
    swapped_targets = labels.index_select(0, swapped_indices)
    source_orbits = [
        development["identity"]["orbit_ids"][index]
        for index in source_indices.tolist()
    ]
    swapped_orbit_accuracy, swapped_orbit_agreement = _renderer_metrics(
        swapped,
        swapped_targets,
        source_orbits,
    )
    report = {
        "candidate": virtual["candidate"],
        "causal_rows": int(source_indices.numel()),
        "clean_accuracy": float(clean.eq(clean_targets).float().mean()),
        "feature_bundle_sha256": args.feature_bundle_sha256,
        "probe_model_sha256": args.probe_model_sha256,
        "schema": "shohin-ettr-capability-floor-source-matched-binding-v1",
        "source_indices_sha256": tensor_sha256(source_indices),
        "swapped_indices_sha256": tensor_sha256(swapped_indices),
        "swapped_original_accuracy": float(
            swapped.eq(clean_targets).float().mean()
        ),
        "swapped_renderer_orbit_accuracy": swapped_orbit_accuracy,
        "swapped_renderer_orbit_agreement": swapped_orbit_agreement,
        "swapped_target_accuracy": float(
            swapped.eq(swapped_targets).float().mean()
        ),
        "tap": args.tap,
    }
    _write_no_replace(args.output, _canonical_bytes(report))
    print(json.dumps(report, sort_keys=True), flush=True)


def extract_taps(args: argparse.Namespace) -> None:
    device = torch.device("cuda", 0)
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    stream.verify_source_shards()
    model, checkpoint = _load_protected_model(
        args.checkpoint,
        expected_sha256=args.checkpoint_sha256,
        device=device,
    )
    config = UnifiedTrajectoryConfig(input_width=int(model.cfg.d_model))
    adapter = _RawTokenizerAdapter(
        TokenizerSpec(
            candidate=PROTECTED_CANDIDATE,
            path=args.tokenizer,
            source_revision=args.checkpoint_sha256,
            context_limit=int(model.cfg.seq_len),
        )
    )
    router = TokenNativeOperationRouter(
        stream.codec.codebook.token_ids,
        vocab_size=stream.codec.tokenizer.get_vocab_size(),
        maximum_positions=96,
        maximum_operations=6,
    )
    splits = {}
    for split, count in (
        ("train", args.train_cores),
        ("development", args.development_cores),
    ):
        records = _select_records(
            stream,
            split,
            count=count,
            seed=args.selection_seed,
        )
        tasks = _source_tasks(
            records,
            adapter=adapter,
            codec=stream.codec,
            router=router,
            config=config,
        )
        splits[split] = _materialize_tap_split(
            model=model,
            tasks=tasks,
            device=device,
            inference_batch=args.inference_batch,
        )
    receipt = _save_tap_bundle(
        args.feature_bundle,
        checkpoint_sha256=args.checkpoint_sha256,
        release_sha256=args.release_sha256,
        config=config,
        splits=splits,
    )
    del model, checkpoint
    print(json.dumps({"tap_bundle": receipt}, sort_keys=True), flush=True)


def fit_tap(args: argparse.Namespace) -> None:
    bundle = _load_tap_bundle(args.feature_bundle, args.feature_bundle_sha256)
    virtual = virtual_feature_bundle(
        bundle,
        tap_name=args.tap,
        bundle_sha256=args.feature_bundle_sha256,
    )
    _fit_probe(
        virtual,
        output=args.output,
        updates=args.updates,
        batch_size=args.probe_batch,
        seed=args.architecture_seed,
        learning_rate=args.learning_rate,
        device=torch.device("cuda", 0),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("extract", "fit", "binding"), required=True)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--feature-bundle-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-model", type=Path)
    parser.add_argument("--probe-model-sha256")
    parser.add_argument("--tap", choices=TAP_NAMES)
    parser.add_argument("--train-cores", type=int, default=128)
    parser.add_argument("--development-cores", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=11)
    parser.add_argument("--architecture-seed", type=int, default=31)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--probe-batch", type=int, default=16)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0).upper():
        raise CapabilityFloorLayerTapError("layer-tap sufficiency requires one H100")
    if args.mode == "extract":
        required = (
            args.release_root,
            args.data_root,
            args.release_sha256,
            args.tokenizer,
            args.checkpoint,
            args.checkpoint_sha256,
        )
        if any(value is None for value in required) or args.feature_bundle_sha256:
            raise CapabilityFloorLayerTapError("tap extraction arguments differ")
        extract_taps(args)
    elif args.mode == "fit":
        if (
            not args.feature_bundle_sha256
            or args.output is None
            or args.tap is None
        ):
            raise CapabilityFloorLayerTapError("tap fit arguments differ")
        fit_tap(args)
    else:
        if (
            not args.feature_bundle_sha256
            or args.output is None
            or args.tap is None
            or args.probe_model is None
            or not args.probe_model_sha256
        ):
            raise CapabilityFloorLayerTapError("tap binding arguments differ")
        score_binding(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
