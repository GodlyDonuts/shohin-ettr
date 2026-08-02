#!/usr/bin/env python3
"""Measure whether exact frozen-backbone tensors preserve ETTR effect families.

This is an interface diagnostic, not an ETTR training result.  It reads an
immutable ETTR-v3 release, extracts the candidate's final post-norm COMMAND
residuals, and pairs each public operation with the exact typed state that
precedes that operation.  Assessor state and family labels are available only
to this offline probe.  They are never candidate inputs in autonomous ETTR.

The development evaluation is balanced by renderer orbit.  In addition to the
clean tensor score, the runner measures renderer agreement, a label-opposed
source/state derangement, a global value-code permutation, and state reset.
The strict receipt remains fail-closed through ``capability_floor_sufficiency``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
for _path in (_ROOT / "train", _PIPELINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from audit_ettr_operation_family_state_conditioning import _operation_contexts  # noqa: E402
from audit_ettr_public_operation_state_delta import runtime_state_value  # noqa: E402
from capability_floor_corpus import (  # noqa: E402
    TokenizerSpec,
    _RawTokenizerAdapter,
    _iter_release_records,
    _operation_role_atom_positions,
    _role_spans,
    token_mask_for_spans,
)
from capability_floor_sufficiency import (  # noqa: E402
    OperationFamilyTensorProbe,
    SufficiencyScores,
    build_sufficiency_receipt,
    tensor_sha256,
    validate_sufficiency_receipt,
)
from capability_floor_trajectory import (  # noqa: E402
    UnifiedTrajectoryConfig,
    UnifiedTypedState,
    empty_unified_state,
)
from ettr_il_v3_protocol import canonical_json_bytes  # noqa: E402
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec  # noqa: E402
from ettr_v3_streaming import ETTRV3StreamingRelease  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from token_native_syntax_router import TokenNativeOperationRouter  # noqa: E402


FEATURE_SCHEMA = "shohin-ettr-capability-floor-feature-bundle-v1"
REPORT_SCHEMA = "shohin-ettr-capability-floor-real-tensor-sufficiency-v1"
FAMILY_TO_INDEX = {"NONE": 0, "WRITE": 1, "LINK": 2}
PROTECTED_CANDIDATE = "protected-shohin-125m-step300k"


class FeatureSufficiencyError(ValueError):
    """A protected tensor, split, or diagnostic contract differs."""


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()


def _torch_save_no_replace(path: Path, payload: object) -> None:
    """Publish one torch artifact atomically without permitting replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FeatureSufficiencyError("torch artifact already exists")
    try:
        torch.save(payload, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_rank(seed: int, split: str, core_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\x1f{split}\x1f{core_id}".encode("ascii")).digest(),
        "big",
    )


def _select_records(
    stream: ETTRV3StreamingRelease,
    split: str,
    *,
    count: int,
    seed: int,
) -> tuple[tuple[str, int, bytes, object], ...]:
    """Select a whole-release hash sample without retaining the population."""

    if count <= 0:
        raise FeatureSufficiencyError("selected core count must be positive")
    heap: list[tuple[int, str, str, int, bytes, object]] = []
    seen = 0
    for path, row_index, payload, record in _iter_release_records(stream, split):
        core_id = str(record.identity.core_id)
        rank = _hash_rank(seed, split, core_id)
        item = (-rank, core_id, path, row_index, payload, record)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
        seen += 1
    if seen < count:
        raise FeatureSufficiencyError("selected core count exceeds split")
    selected = sorted(heap, key=lambda value: (-value[0], value[1]))
    return tuple(
        (path, row_index, payload, record)
        for _rank, _core_id, path, row_index, payload, record in selected
    )


def _runtime_state_payload(
    state: object,
    config: UnifiedTrajectoryConfig,
) -> dict[str, torch.Tensor]:
    value = runtime_state_value(state)
    values = torch.zeros(config.num_slots, config.num_value_codes, dtype=torch.bool)
    types = torch.zeros(config.num_slots, config.num_types, dtype=torch.bool)
    relations = torch.zeros(
        config.num_relations,
        config.num_slots,
        config.num_slots,
        dtype=torch.bool,
    )
    active = torch.zeros(config.num_slots, dtype=torch.bool)
    root = torch.zeros(config.num_slots, dtype=torch.bool)
    for row in value["nodes"]:
        if len(row) != 5:
            raise FeatureSufficiencyError("runtime node geometry differs")
        slot, is_active, type_index, value_code, is_root = row
        slot = int(slot)
        type_index = int(type_index)
        value_code = int(value_code)
        if (
            not bool(is_active)
            or not 0 <= slot < config.num_slots
            or not 0 <= type_index < config.num_types
            or not 0 <= value_code < config.num_value_codes
        ):
            raise FeatureSufficiencyError("runtime node leaves unified state")
        active[slot] = True
        root[slot] = bool(is_root)
        types[slot, type_index] = True
        values[slot, value_code] = True
    for relation, source, target in value["edges"]:
        relation = int(relation)
        source = int(source)
        target = int(target)
        if (
            not 0 <= relation < config.num_relations
            or not 0 <= source < config.num_slots
            or not 0 <= target < config.num_slots
            or not active[source]
            or not active[target]
        ):
            raise FeatureSufficiencyError("runtime edge leaves unified state")
        relations[relation, source, target] = True
    status = list(value["status"])
    if len(status) != 2:
        raise FeatureSufficiencyError("runtime status differs")
    return {
        "value_probabilities": values,
        "type_probabilities": types,
        "relations": relations,
        "active": active,
        "root": root,
        "committed": torch.tensor(bool(status[0])),
    }


def _stack_state(
    rows: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> UnifiedTypedState:
    if not rows:
        raise FeatureSufficiencyError("typed-state rows are empty")
    values = {
        name: torch.stack([row[name] for row in rows]).to(device=device, dtype=dtype)
        for name in (
            "value_probabilities",
            "type_probabilities",
            "relations",
            "active",
            "root",
            "committed",
        )
    }
    return UnifiedTypedState(**values, step=0)


def _index_state(
    state: UnifiedTypedState,
    indices: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> UnifiedTypedState:
    return UnifiedTypedState(
        **{
            name: getattr(state, name).index_select(0, indices.cpu()).to(
                device=device,
                dtype=dtype,
            )
            for name in (
                "value_probabilities",
                "type_probabilities",
                "relations",
                "active",
                "root",
                "committed",
            )
        },
        step=0,
    )


def _encode_hidden(model: GPT, tokens: torch.Tensor) -> torch.Tensor:
    """Exact final post-norm hidden state without materializing vocabulary logits."""

    if tokens.ndim != 2 or tokens.dtype != torch.long:
        raise FeatureSufficiencyError("backbone token tensor differs")
    _batch, length = tokens.shape
    if length > model.cfg.seq_len:
        raise FeatureSufficiencyError("backbone source exceeds context")
    hidden = model.tok(tokens)
    cos = model.cos[:length].to(hidden.device)
    sin = model.sin[:length].to(hidden.device)
    for _loop in range(model.cfg.n_loop):
        for block in model.blocks:
            hidden, _ = block(hidden, cos, sin)
    return model.norm(hidden)


@dataclass(frozen=True, slots=True)
class _SourceTask:
    token_ids: tuple[int, ...]
    role_masks: tuple[tuple[tuple[bool, ...], ...], ...]
    labels: tuple[int, ...]
    states: tuple[Mapping[str, torch.Tensor], ...]
    orbit_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    source_sha256: str


def _contexts_by_corner(record: object) -> tuple[tuple[tuple[str, object], ...], ...]:
    contexts = iter(_operation_contexts(record))
    commands = tuple(record.assessor_only.semantic_factors.commands)
    if len(commands) != 2:
        raise FeatureSufficiencyError("semantic command factors differ")
    lengths = tuple(len(value["operations"]) for value in commands)
    result = []
    for _world in range(2):
        for command in range(2):
            rows = []
            for _operation in range(lengths[command]):
                try:
                    rows.append(next(contexts))
                except StopIteration as error:
                    raise FeatureSufficiencyError("operation contexts underflow") from error
            result.append(tuple(rows))
    try:
        next(contexts)
    except StopIteration:
        return tuple(result)
    raise FeatureSufficiencyError("operation contexts overflow")


def _source_tasks(
    records: Sequence[tuple[str, int, bytes, object]],
    *,
    adapter: _RawTokenizerAdapter,
    codec: TokenNativeSurfaceCodec,
    router: TokenNativeOperationRouter,
    config: UnifiedTrajectoryConfig,
) -> tuple[_SourceTask, ...]:
    tasks = []
    for _path, _row_index, _payload, record in records:
        core_id = str(record.identity.core_id)
        contexts = _contexts_by_corner(record)
        views = tuple(record.source_visible.views)
        if len(views) != 4:
            raise FeatureSufficiencyError("renderer orbit differs")
        for view_index, view in enumerate(views):
            for corner, source in enumerate(view.command_sources):
                encoded = adapter.encode(source)
                operation_roles = _operation_role_atom_positions(
                    codec,
                    router,
                    source,
                )
                if len(operation_roles) != len(contexts[corner]):
                    raise FeatureSufficiencyError("operation/context geometry differs")
                masks = []
                labels = []
                states = []
                orbit_ids = []
                sample_ids = []
                for operation, (roles, (family, state)) in enumerate(
                    zip(operation_roles, contexts[corner], strict=True)
                ):
                    label = FAMILY_TO_INDEX.get(str(family).upper())
                    if label is None:
                        raise FeatureSufficiencyError("operation family differs")
                    role_rows = [
                        token_mask_for_spans(encoded.offsets, spans)
                        for spans in _role_spans(roles)
                    ]
                    while len(role_rows) < 4:
                        role_rows.append(tuple(False for _ in encoded.token_ids))
                    if len(role_rows) != 4:
                        raise FeatureSufficiencyError("operation role count differs")
                    orbit = f"{core_id}:{corner}:{operation}"
                    masks.append(tuple(role_rows))
                    labels.append(label)
                    states.append(_runtime_state_payload(state, config))
                    orbit_ids.append(orbit)
                    sample_ids.append(f"{orbit}:view={view_index}")
                tasks.append(
                    _SourceTask(
                        token_ids=encoded.token_ids,
                        role_masks=tuple(masks),
                        labels=tuple(labels),
                        states=tuple(states),
                        orbit_ids=tuple(orbit_ids),
                        sample_ids=tuple(sample_ids),
                        source_sha256=hashlib.sha256(source.encode("ascii")).hexdigest(),
                    )
                )
    return tuple(tasks)


def _load_protected_model(
    checkpoint: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[GPT, Mapping[str, object]]:
    observed = _sha256_file(checkpoint)
    if observed != expected_sha256:
        raise FeatureSufficiencyError("protected checkpoint SHA-256 differs")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("cfg"), Mapping)
        or not isinstance(payload.get("model"), Mapping)
        or payload.get("step") != 300000
    ):
        raise FeatureSufficiencyError("protected checkpoint contract differs")
    model = GPT(GPTConfig(**dict(payload["cfg"])))
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FeatureSufficiencyError("protected checkpoint strict load differs")
    model.to(device).eval().requires_grad_(False)
    return model, payload


def _materialize_split(
    *,
    model: GPT,
    tasks: Sequence[_SourceTask],
    config: UnifiedTrajectoryConfig,
    device: torch.device,
    inference_batch: int,
) -> dict[str, object]:
    features = []
    masks = []
    roles = []
    labels = []
    states = []
    orbit_ids = []
    sample_ids = []
    source_hashes = []
    maximum = max(len(task.token_ids) for task in tasks)
    if maximum > model.cfg.seq_len:
        raise FeatureSufficiencyError("selected source exceeds model context")
    with torch.inference_mode():
        for start in range(0, len(tasks), inference_batch):
            chunk = tasks[start : start + inference_batch]
            length = max(len(task.token_ids) for task in chunk)
            tokens = torch.zeros(len(chunk), length, dtype=torch.long, device=device)
            for row, task in enumerate(chunk):
                tokens[row, : len(task.token_ids)] = torch.tensor(
                    task.token_ids,
                    device=device,
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = _encode_hidden(model, tokens).to(torch.bfloat16).cpu()
            for row, task in enumerate(chunk):
                source_length = len(task.token_ids)
                padded_hidden = torch.zeros(maximum, model.cfg.d_model, dtype=torch.bfloat16)
                padded_hidden[:source_length] = hidden[row, :source_length]
                source_mask = torch.zeros(maximum, dtype=torch.bool)
                source_mask[:source_length] = True
                for operation in range(len(task.labels)):
                    role_mask = torch.zeros(4, maximum, dtype=torch.bool)
                    raw_roles = torch.tensor(task.role_masks[operation], dtype=torch.bool)
                    role_mask[:, :source_length] = raw_roles
                    features.append(padded_hidden.clone())
                    masks.append(source_mask.clone())
                    roles.append(role_mask)
                    labels.append(task.labels[operation])
                    states.append(task.states[operation])
                    orbit_ids.append(task.orbit_ids[operation])
                    sample_ids.append(task.sample_ids[operation])
                    source_hashes.append(task.source_sha256)
    state = _stack_state(states)
    tensors = {
        "source_features": torch.stack(features),
        "source_mask": torch.stack(masks),
        "role_masks": torch.stack(roles),
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
                "role_masks_sha256": tensor_sha256(tensors["role_masks"]),
                "schema": FEATURE_SCHEMA,
                "source_features_sha256": tensor_sha256(
                    tensors["source_features"]
                ),
                "source_mask_sha256": tensor_sha256(tensors["source_mask"]),
                "state_sha256": {
                    name: tensor_sha256(value)
                    for name, value in tensors["state"].items()
                },
            }
        )
    ).hexdigest()
    return {
        "identity": identity,
        "split_sha256": split_sha256,
        "tensors": tensors,
    }


def _save_feature_bundle(
    path: Path,
    *,
    candidate: str,
    checkpoint_sha256: str,
    release_sha256: str,
    config: UnifiedTrajectoryConfig,
    splits: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if path.exists():
        raise FeatureSufficiencyError("feature bundle already exists")
    payload = {
        "candidate": candidate,
        "checkpoint_sha256": checkpoint_sha256,
        "config": asdict(config),
        "release_sha256": release_sha256,
        "schema": FEATURE_SCHEMA,
        "splits": splits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _torch_save_no_replace(path, payload)
    receipt = {
        "bytes": path.stat().st_size,
        "candidate": candidate,
        "checkpoint_sha256": checkpoint_sha256,
        "path": path.name,
        "release_sha256": release_sha256,
        "schema": FEATURE_SCHEMA,
        "sha256": _sha256_file(path),
        "splits": {
            split: {
                "examples": int(value["tensors"]["labels"].numel()),
                "family_counts": dict(
                    sorted(Counter(int(item) for item in value["tensors"]["labels"].tolist()).items())
                ),
                "renderer_orbits": len(set(value["identity"]["orbit_ids"])),
                "split_sha256": value["split_sha256"],
            }
            for split, value in splits.items()
        },
    }
    _write_no_replace(path.with_suffix(path.suffix + ".receipt.json"), _canonical_bytes(receipt) + b"\n")
    return receipt


def _load_feature_bundle(path: Path, expected_sha256: str) -> Mapping[str, object]:
    if _sha256_file(path) != expected_sha256:
        raise FeatureSufficiencyError("feature bundle SHA-256 differs")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping) or value.get("schema") != FEATURE_SCHEMA:
        raise FeatureSufficiencyError("feature bundle contract differs")
    return value


def _balanced_orbit_indices(
    labels: torch.Tensor,
    orbit_ids: Sequence[str],
    *,
    seed: int,
) -> torch.Tensor:
    if labels.ndim != 1 or len(orbit_ids) != labels.numel():
        raise FeatureSufficiencyError("orbit balance inputs differ")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, orbit in enumerate(orbit_ids):
        groups[orbit].append(index)
    by_label: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for orbit, indices in groups.items():
        values = {int(labels[index]) for index in indices}
        if len(values) != 1 or len(indices) != 4:
            raise FeatureSufficiencyError("renderer orbit is incomplete")
        by_label[next(iter(values))].append((orbit, indices))
    if set(by_label) != set(FAMILY_TO_INDEX.values()):
        raise FeatureSufficiencyError("balanced split lacks an effect family")
    keep = min(len(values) for values in by_label.values())
    selected = []
    for label in sorted(by_label):
        values = sorted(
            by_label[label],
            key=lambda item: hashlib.sha256(
                f"{seed}\x1f{label}\x1f{item[0]}".encode("ascii")
            ).digest(),
        )[:keep]
        selected.extend(index for _orbit, indices in values for index in indices)
    return torch.tensor(sorted(selected), dtype=torch.long)


def _label_opposed_derangement(labels: torch.Tensor) -> torch.Tensor:
    """Map every row to a row from another family without replacement."""

    if labels.ndim != 1 or labels.numel() < 3:
        raise FeatureSufficiencyError("derangement labels differ")
    by_label = {
        label: torch.where(labels.eq(label))[0].tolist()
        for label in sorted(set(int(value) for value in labels.tolist()))
    }
    counts = {len(value) for value in by_label.values()}
    if set(by_label) != set(FAMILY_TO_INDEX.values()) or len(counts) != 1:
        raise FeatureSufficiencyError("derangement requires balanced families")
    order = sorted(by_label)
    mapping = torch.empty(labels.numel(), dtype=torch.long)
    for index, label in enumerate(order):
        source = by_label[label]
        target = by_label[order[(index + 1) % len(order)]]
        for left, right in zip(source, target, strict=True):
            mapping[left] = right
    if labels.index_select(0, mapping).eq(labels).any() or mapping.unique().numel() != labels.numel():
        raise FeatureSufficiencyError("label-opposed derangement failed")
    return mapping


def _family_logits(
    probe: OperationFamilyTensorProbe,
    source_features: torch.Tensor,
    source_mask: torch.Tensor,
    role_masks: torch.Tensor,
    state: UnifiedTypedState,
) -> torch.Tensor:
    role_logits = probe(source_features, source_mask, role_masks, state)
    valid = role_masks.any(-1).to(role_logits.dtype)
    return (role_logits * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1.0)


def _state_from_payload(payload: Mapping[str, torch.Tensor]) -> UnifiedTypedState:
    return UnifiedTypedState(**dict(payload), step=0)


def _evaluate(
    probe: OperationFamilyTensorProbe,
    split: Mapping[str, object],
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    state_permutation: torch.Tensor | None = None,
    value_code_permutation: torch.Tensor | None = None,
    reset_state: bool = False,
) -> tuple[float, torch.Tensor]:
    tensors = split["tensors"]
    state_payload = tensors["state"]
    source_state = _state_from_payload(state_payload)
    predictions = []
    correct = 0
    for start in range(0, indices.numel(), batch_size):
        selected = indices[start : start + batch_size]
        state_indices = selected
        if state_permutation is not None:
            state_indices = indices.index_select(0, state_permutation[start : start + selected.numel()])
        state = _index_state(source_state, state_indices, device=device, dtype=torch.float32)
        if reset_state:
            state = empty_unified_state(
                selected.numel(),
                probe.config,
                device=device,
                dtype=torch.float32,
            )
        elif value_code_permutation is not None:
            state = UnifiedTypedState(
                value_probabilities=state.value_probabilities.index_select(
                    -1,
                    value_code_permutation.to(device),
                ),
                type_probabilities=state.type_probabilities,
                relations=state.relations,
                active=state.active,
                root=state.root,
                committed=state.committed,
                step=0,
            )
        features = tensors["source_features"].index_select(0, selected).to(device)
        source_mask = tensors["source_mask"].index_select(0, selected).to(device)
        role_masks = tensors["role_masks"].index_select(0, selected).to(device)
        labels = tensors["labels"].index_select(0, selected).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = _family_logits(probe, features, source_mask, role_masks, state)
        predicted = logits.argmax(-1).cpu()
        predictions.append(predicted)
        correct += int(predicted.eq(labels.cpu()).sum())
    result = torch.cat(predictions)
    return correct / indices.numel(), result


def _renderer_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    orbit_ids: Sequence[str],
) -> tuple[float, float]:
    if predictions.shape != labels.shape or predictions.numel() != len(orbit_ids):
        raise FeatureSufficiencyError("renderer metrics geometry differs")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, orbit in enumerate(orbit_ids):
        groups[orbit].append(index)
    correct = 0
    agreement = 0
    for indices in groups.values():
        if len(indices) != 4:
            raise FeatureSufficiencyError("renderer orbit is incomplete")
        values = predictions[indices]
        target = labels[indices]
        agreement += int(values.eq(values[0]).all())
        correct += int(values.eq(target).all())
    return correct / len(groups), agreement / len(groups)


def _fit_probe(
    bundle: Mapping[str, object],
    *,
    output: Path,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    if output.exists():
        raise FeatureSufficiencyError("sufficiency report already exists")
    config = UnifiedTrajectoryConfig(**dict(bundle["config"]))
    train = bundle["splits"]["train"]
    development = bundle["splits"]["development"]
    train_indices = _balanced_orbit_indices(
        train["tensors"]["labels"],
        train["identity"]["orbit_ids"],
        seed=seed,
    )
    development_indices = _balanced_orbit_indices(
        development["tensors"]["labels"],
        development["identity"]["orbit_ids"],
        seed=seed,
    )
    torch.manual_seed(seed)
    probe = OperationFamilyTensorProbe(config, max_roles=4).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    generator = torch.Generator().manual_seed(seed)
    state_payload = _state_from_payload(train["tensors"]["state"])
    trace = []
    probe.train()
    for update in range(1, updates + 1):
        positions = torch.randint(
            train_indices.numel(),
            (batch_size,),
            generator=generator,
        )
        selected = train_indices.index_select(0, positions)
        state = _index_state(state_payload, selected, device=device, dtype=torch.float32)
        features = train["tensors"]["source_features"].index_select(0, selected).to(device)
        source_mask = train["tensors"]["source_mask"].index_select(0, selected).to(device)
        role_masks = train["tensors"]["role_masks"].index_select(0, selected).to(device)
        labels = train["tensors"]["labels"].index_select(0, selected).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = _family_logits(probe, features, source_mask, role_masks, state)
            loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise FeatureSufficiencyError("sufficiency loss is nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == updates:
            row = {
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "update": update,
            }
            trace.append(row)
            print(json.dumps({"sufficiency": row}, sort_keys=True), flush=True)
    probe.eval()
    clean_accuracy, predictions = _evaluate(
        probe,
        development,
        development_indices,
        device=device,
        batch_size=batch_size,
    )
    selected_labels = development["tensors"]["labels"].index_select(0, development_indices)
    selected_orbits = [
        development["identity"]["orbit_ids"][index]
        for index in development_indices.tolist()
    ]
    renderer_accuracy, renderer_agreement = _renderer_metrics(
        predictions,
        selected_labels,
        selected_orbits,
    )
    derangement = _label_opposed_derangement(selected_labels)
    binding_accuracy, _ = _evaluate(
        probe,
        development,
        development_indices,
        device=device,
        batch_size=batch_size,
        state_permutation=derangement,
    )
    value_permutation = torch.randperm(config.num_value_codes, generator=generator)
    value_accuracy, _ = _evaluate(
        probe,
        development,
        development_indices,
        device=device,
        batch_size=batch_size,
        value_code_permutation=value_permutation,
    )
    reset_accuracy, _ = _evaluate(
        probe,
        development,
        development_indices,
        device=device,
        batch_size=batch_size,
        reset_state=True,
    )
    scores = SufficiencyScores(
        symbolic_reference_accuracy=1.0,
        tensor_probe_accuracy=float(clean_accuracy),
        renderer_orbit_accuracy=float(renderer_accuracy),
        renderer_orbit_prediction_agreement=float(renderer_agreement),
        binding_deranged_accuracy=float(binding_accuracy),
        state_value_permuted_accuracy=float(value_accuracy),
        empirical_chance_accuracy=1.0 / 3.0,
    )
    dev_tensors = development["tensors"]
    receipt_features = dev_tensors["source_features"].index_select(
        0,
        development_indices,
    )
    receipt_source_mask = dev_tensors["source_mask"].index_select(
        0,
        development_indices,
    )
    receipt_role_masks = dev_tensors["role_masks"].index_select(
        0,
        development_indices,
    )
    receipt_labels = dev_tensors["labels"].index_select(0, development_indices)
    full_dev_state = _state_from_payload(dev_tensors["state"])
    receipt_state = _index_state(
        full_dev_state,
        development_indices,
        device=torch.device("cpu"),
        dtype=torch.bool,
    )
    balanced_split_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "full_split_sha256": development["split_sha256"],
                "indices": development_indices.tolist(),
                "sample_ids": [
                    development["identity"]["sample_ids"][index]
                    for index in development_indices.tolist()
                ],
            }
        )
    ).hexdigest()
    receipt = build_sufficiency_receipt(
        candidate=str(bundle["candidate"]),
        component="operation-family",
        split_sha256=balanced_split_sha256,
        source_features=receipt_features,
        source_mask=receipt_source_mask,
        role_masks=receipt_role_masks,
        state=receipt_state,
        labels=receipt_labels,
        scores=scores,
    )
    validate_sufficiency_receipt(receipt)
    model_path = output.with_suffix(".pt")
    _torch_save_no_replace(
        model_path,
        {
            "candidate": bundle["candidate"],
            "config": asdict(config),
            "probe": probe.state_dict(),
            "schema": REPORT_SCHEMA,
            "seed": seed,
        },
    )
    report = {
        "assessor_features_available_at_inference": False,
        "balanced_development_examples": int(development_indices.numel()),
        "balanced_development_split_sha256": balanced_split_sha256,
        "balanced_train_examples": int(train_indices.numel()),
        "candidate": bundle["candidate"],
        "feature_bundle_sha256": bundle["feature_bundle_sha256"],
        "learning_rate": learning_rate,
        "model": {
            "bytes": model_path.stat().st_size,
            "path": model_path.name,
            "sha256": _sha256_file(model_path),
        },
        "parameter_count": sum(value.numel() for value in probe.parameters()),
        "control_permutations": {
            "binding_derangement_sha256": tensor_sha256(derangement),
            "value_code_permutation_sha256": tensor_sha256(value_permutation),
        },
        "receipt": receipt,
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "state_reset_accuracy": reset_accuracy,
        "symbolic_reference_boundary": (
            "The exact interpreter defines and recomputes each offline family from the "
            "public operation plus its preceding typed state; it is never a model input."
        ),
        "trace": trace,
        "updates": updates,
    }
    _write_no_replace(output, _canonical_bytes(report) + b"\n")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")), flush=True)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--feature-bundle", type=Path, required=True)
    parser.add_argument("--feature-bundle-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-cores", type=int, default=128)
    parser.add_argument("--development-cores", type=int, default=128)
    parser.add_argument("--selection-seed", type=int, default=11)
    parser.add_argument("--architecture-seed", type=int, default=31)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--probe-batch", type=int, default=16)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0).upper():
        raise FeatureSufficiencyError("real-tensor sufficiency requires one H100")
    device = torch.device("cuda", 0)
    if args.feature_bundle.exists():
        if not args.feature_bundle_sha256:
            raise FeatureSufficiencyError("existing feature bundle requires exact SHA-256")
        bundle = dict(_load_feature_bundle(args.feature_bundle, args.feature_bundle_sha256))
        bundle["feature_bundle_sha256"] = args.feature_bundle_sha256
    else:
        if args.feature_bundle_sha256:
            raise FeatureSufficiencyError("new feature bundle cannot predeclare SHA-256")
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
        spec = TokenizerSpec(
            candidate=PROTECTED_CANDIDATE,
            path=args.tokenizer,
            source_revision=args.checkpoint_sha256,
            context_limit=int(model.cfg.seq_len),
        )
        adapter = _RawTokenizerAdapter(spec)
        router = TokenNativeOperationRouter(
            stream.codec.codebook.token_ids,
            vocab_size=stream.codec.tokenizer.get_vocab_size(),
            maximum_positions=96,
            maximum_operations=6,
        )
        split_counts = {
            "train": args.train_cores,
            "development": args.development_cores,
        }
        splits = {}
        for split, count in split_counts.items():
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
            splits[split] = _materialize_split(
                model=model,
                tasks=tasks,
                config=config,
                device=device,
                inference_batch=args.inference_batch,
            )
        receipt = _save_feature_bundle(
            args.feature_bundle,
            candidate=PROTECTED_CANDIDATE,
            checkpoint_sha256=args.checkpoint_sha256,
            release_sha256=args.release_sha256,
            config=config,
            splits=splits,
        )
        del model, checkpoint
        torch.cuda.empty_cache()
        bundle = dict(_load_feature_bundle(args.feature_bundle, receipt["sha256"]))
        bundle["feature_bundle_sha256"] = receipt["sha256"]
        print(json.dumps({"feature_bundle": receipt}, sort_keys=True), flush=True)
    if args.extract_only:
        return 0
    if args.output is None:
        raise FeatureSufficiencyError("probe output is required unless extract-only")
    _fit_probe(
        bundle,
        output=args.output,
        updates=args.updates,
        batch_size=args.probe_batch,
        seed=args.architecture_seed,
        learning_rate=args.learning_rate,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
