#!/usr/bin/env python3
"""Deterministic natural-command data for DIVERGE-NCP1."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

from diverge_eal1_data import _word as eal_word
from diverge_eal1_data import canonical_sha256


TRAIN_SCHEMA = "shohin-diverge-ncp1-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-ncp1-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-ncp1-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-ncp1-data-report-v1"
TRAIN_SEED = 2026080791
DEVELOPMENT_SEED = 2026080798
CONFIRMATION_SEEDS = (
    2026080799,
    2026080800,
    2026080801,
    2026080802,
    2026080803,
)
TRAIN_ROWS = 100_000
OPERATIONS = 8
TRAIN_DEPTHS = tuple(range(4, 21))

_ACTIONS = (
    "apply {alias}",
    "execute {alias}",
    "run {alias}",
    "use {alias}",
)
_CONNECTORS = (
    " Then {clause}.",
    " Next, {clause}.",
    " After that, {clause}.",
    " Proceed to {clause}.",
)


class NCP1DataError(RuntimeError):
    """A natural command record violates its frozen contract."""


def _word(namespace: str, seed: int, serial: int, index: int) -> str:
    if (
        namespace
        not in (
            "ncp1-train-alias",
            "ncp1-renamed-alias",
            "ncp1-decoy-alias",
        )
        or not 0 <= seed - TRAIN_SEED < 26
    ):
        raise NCP1DataError("NCP1 opaque-name namespace or seed differs")
    return eal_word(namespace, seed, serial, index)


def command_renderer_pairs(split: str) -> tuple[tuple[int, int], ...]:
    if split not in ("train", "development"):
        raise NCP1DataError("NCP1 renderer split differs")
    bucket = 0 if split == "train" else 1
    return tuple(
        (action, connector)
        for action in range(len(_ACTIONS))
        for connector in range(len(_CONNECTORS))
        if (action + connector) % 2 == bucket
    )


def render_command(
    aliases: Sequence[str],
    targets: Sequence[int],
    *,
    split: str,
    seed: int,
    serial: int,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    table = tuple(str(value) for value in aliases)
    target_ids = tuple(int(value) for value in targets)
    if (
        len(table) != OPERATIONS
        or len(set(table)) != OPERATIONS
        or any(not value.isalpha() or not value.islower() for value in table)
        or not target_ids
        or any(value < 0 or value >= OPERATIONS for value in target_ids)
    ):
        raise NCP1DataError("NCP1 command carrier differs")
    pairs = command_renderer_pairs(split)
    selected = tuple(
        pairs[(serial + step) % len(pairs)] for step in range(len(target_ids))
    )
    clauses = [
        _ACTIONS[action].format(alias=table[target])
        for target, (action, _) in zip(target_ids, selected, strict=True)
    ]
    text = clauses[0].capitalize() + "."
    for clause, (_, connector) in zip(clauses[1:], selected[1:], strict=True):
        text += _CONNECTORS[connector].format(clause=clause)
    if len(text.encode("ascii")) > 1_536:
        raise NCP1DataError("NCP1 command exceeds frozen byte width")
    return text, selected


def alignment_spans(
    text: str, aliases: Sequence[str], targets: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Return supervisor-only byte spans for each emitted alias occurrence."""

    cursor = 0
    spans = []
    for target in targets:
        alias = str(aliases[int(target)])
        start = text.find(alias, cursor)
        if start < 0:
            raise NCP1DataError("NCP1 command omits a target alias")
        end = start + len(alias)
        spans.append((start + 1, end + 1))  # Runtime byte tensors prepend CLS.
        cursor = end
    return tuple(spans)


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["ncp1-training", TRAIN_SEED, serial]))
    alias_values = [
        _word("ncp1-train-alias", TRAIN_SEED, serial, index)
        for index in range(OPERATIONS)
    ]
    rng.shuffle(alias_values)
    aliases = tuple(alias_values)
    depth = TRAIN_DEPTHS[rng.randrange(len(TRAIN_DEPTHS))]
    targets = tuple(rng.randrange(OPERATIONS) for _ in range(depth))
    text, renderer = render_command(
        aliases,
        targets,
        split="train",
        seed=TRAIN_SEED,
        serial=serial,
    )
    record = {
        "schema": TRAIN_SCHEMA,
        "seed": TRAIN_SEED,
        "serial": serial,
        "aliases": list(aliases),
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "renderer": [list(value) for value in renderer],
        "targets": list(targets),
        "alignment_spans": [
            list(value) for value in alignment_spans(text, aliases, targets)
        ],
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_training_record(record)
    return record


def validate_training_record(record: Mapping[str, Any]) -> None:
    if (
        record.get("schema") != TRAIN_SCHEMA
        or int(record.get("seed", -1)) != TRAIN_SEED
    ):
        raise NCP1DataError("NCP1 training schema or seed differs")
    aliases = tuple(str(value) for value in record["aliases"])
    targets = tuple(int(value) for value in record["targets"])
    text = str(record["source_text"])
    spans = tuple(
        tuple(int(value) for value in span) for span in record["alignment_spans"]
    )
    if (
        len(aliases) != OPERATIONS
        or len(set(aliases)) != OPERATIONS
        or len(targets) not in TRAIN_DEPTHS
        or len(spans) != len(targets)
        or spans != alignment_spans(text, aliases, targets)
        or any(value < 0 or value >= OPERATIONS for value in targets)
        or hashlib.sha256(text.encode()).hexdigest() != record["source_sha256"]
    ):
        raise NCP1DataError("NCP1 training geometry differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise NCP1DataError("NCP1 training identity differs")


def augment_evaluation_episode(
    public: Mapping[str, Any],
    assessor: Mapping[str, Any],
    *,
    seed: int,
    serial: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = tuple(str(value) for value in public["aliases"])
    renamed_values = [
        _word("ncp1-renamed-alias", seed, serial, index) for index in range(OPERATIONS)
    ]
    decoy_values = [
        _word("ncp1-decoy-alias", seed, serial, index) for index in range(OPERATIONS)
    ]
    rename_rng = random.Random(canonical_sha256(["ncp1-rename-order", seed, serial]))
    decoy_rng = random.Random(canonical_sha256(["ncp1-decoy-order", seed, serial]))
    rename_rng.shuffle(renamed_values)
    decoy_rng.shuffle(decoy_values)
    renamed = tuple(renamed_values)
    decoys = tuple(decoy_values)
    transfers = []
    command_targets = []
    for program_index, transfer in enumerate(public["transfer"]):
        targets = tuple(aliases.index(str(symbol)) for symbol in transfer["symbols"])
        command_serial = serial * 100 + program_index
        source, renderer = render_command(
            aliases,
            targets,
            split="development",
            seed=seed,
            serial=command_serial,
        )
        reverse_source, _ = render_command(
            aliases,
            tuple(reversed(targets)),
            split="development",
            seed=seed,
            serial=command_serial,
        )
        renamed_source, _ = render_command(
            renamed,
            targets,
            split="development",
            seed=seed,
            serial=command_serial,
        )
        scrubbed_source, _ = render_command(
            decoys,
            targets,
            split="development",
            seed=seed,
            serial=command_serial,
        )
        visible = {key: value for key, value in transfer.items() if key != "symbols"}
        visible.update(
            {
                "command_text": source,
                "command_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "reverse_command_text": reverse_source,
                "reverse_command_sha256": hashlib.sha256(
                    reverse_source.encode()
                ).hexdigest(),
                "renamed_command_text": renamed_source,
                "renamed_command_sha256": hashlib.sha256(
                    renamed_source.encode()
                ).hexdigest(),
                "scrubbed_command_text": scrubbed_source,
                "scrubbed_command_sha256": hashlib.sha256(
                    scrubbed_source.encode()
                ).hexdigest(),
                "command_renderer": [list(value) for value in renderer],
            }
        )
        transfers.append(visible)
        command_targets.append(
            {
                "program_id": transfer["program_id"],
                "targets": list(targets),
                "reverse_targets": list(reversed(targets)),
            }
        )
    visible_episode = {
        **public,
        "schema": PUBLIC_SCHEMA,
        "seed": seed,
        "renamed_aliases": list(renamed),
        "transfer": transfers,
    }
    hidden_episode = {
        **assessor,
        "schema": ASSESSOR_SCHEMA,
        "seed": seed,
        "command_targets": command_targets,
    }
    visible_episode["identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in visible_episode.items()
            if key != "identity_sha256"
        }
    )
    hidden_episode["identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in hidden_episode.items()
            if key != "identity_sha256"
        }
    )
    validate_evaluation_episode(visible_episode, hidden_episode)
    return visible_episode, hidden_episode


def validate_evaluation_episode(
    public: Mapping[str, Any], assessor: Mapping[str, Any]
) -> None:
    if (
        public.get("schema") != PUBLIC_SCHEMA
        or assessor.get("schema") != ASSESSOR_SCHEMA
    ):
        raise NCP1DataError("NCP1 evaluation schema differs")
    aliases = tuple(str(value) for value in public["aliases"])
    renamed = tuple(str(value) for value in public["renamed_aliases"])
    targets = tuple(assessor["command_targets"])
    transfers = tuple(public["transfer"])
    if (
        len(aliases) != OPERATIONS
        or len(renamed) != OPERATIONS
        or len(set(aliases)) != OPERATIONS
        or len(set(renamed)) != OPERATIONS
        or set(aliases) & set(renamed)
        or len(transfers) != len(targets)
    ):
        raise NCP1DataError("NCP1 evaluation geometry differs")
    for transfer, target in zip(transfers, targets, strict=True):
        depth = int(transfer["depth"])
        sequence = tuple(int(value) for value in target["targets"])
        reverse = tuple(int(value) for value in target["reverse_targets"])
        if (
            transfer["program_id"] != target["program_id"]
            or len(sequence) != depth
            or reverse != tuple(reversed(sequence))
            or any(value < 0 or value >= OPERATIONS for value in sequence)
            or "symbols" in transfer
        ):
            raise NCP1DataError("NCP1 command target differs")
        for key, hash_key in (
            ("command_text", "command_sha256"),
            ("reverse_command_text", "reverse_command_sha256"),
            ("renamed_command_text", "renamed_command_sha256"),
            ("scrubbed_command_text", "scrubbed_command_sha256"),
        ):
            if (
                hashlib.sha256(str(transfer[key]).encode()).hexdigest()
                != transfer[hash_key]
            ):
                raise NCP1DataError("NCP1 command commitment differs")
    for episode in (public, assessor):
        payload = dict(episode)
        identity = str(payload.pop("identity_sha256"))
        if canonical_sha256(payload) != identity:
            raise NCP1DataError("NCP1 episode identity differs")


__all__ = [
    "ASSESSOR_SCHEMA",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_SEED",
    "NCP1DataError",
    "OPERATIONS",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "TRAIN_DEPTHS",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_evaluation_episode",
    "alignment_spans",
    "build_training_record",
    "command_renderer_pairs",
    "render_command",
    "validate_evaluation_episode",
    "validate_training_record",
]
