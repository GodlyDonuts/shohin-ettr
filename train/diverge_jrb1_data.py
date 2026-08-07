#!/usr/bin/env python3
"""Deterministic joint-register data for DIVERGE-JRB1."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Mapping, Sequence

from diverge_eal1_data import (
    DEVELOPMENT_EPISODES,
    OPERATIONS,
    TRAIN_MATRICES,
    _word as eal_word,
    apply_matrix,
    canonical_sha256,
    scan_integer_spans,
)
from diverge_eal2_data import (
    _transition_record,
    build_evaluation_episode,
    validate_episode,
)
from diverge_ncp1_data import render_command


TRAIN_SCHEMA = "shohin-diverge-jrb1-training-v1"
PUBLIC_SCHEMA = "shohin-diverge-jrb1-public-v1"
ASSESSOR_SCHEMA = "shohin-diverge-jrb1-assessor-v1"
REPORT_SCHEMA = "shohin-diverge-jrb1-data-report-v1"
TRAIN_SEED = 2026080811
DEVELOPMENT_SEED = 2026080812
CONFIRMATION_SEEDS = (
    2026080813,
    2026080814,
    2026080815,
    2026080816,
    2026080817,
)
TRAIN_ROWS = 100_000
REGISTERS = 2

_INITIAL_PREFIXES = (
    "Initially",
    "At the outset",
    "Before the sequence",
    "At program start",
)
_INITIAL_FRAMES = (
    "{register} was {value}",
    "{register} held {value}",
    "{register} contained {value}",
    "the value in {register} equaled {value}",
)
_QUERY_PREFIXES = (
    "After every operation",
    "At the end",
    "Once the sequence finishes",
    "Following all updates",
)
_QUERY_FRAMES = (
    "what value does {register} hold?",
    "report the value in {register}.",
    "what number is stored in {register}?",
    "give the final contents of {register}.",
)


class JRB1DataError(RuntimeError):
    """A joint-register record violates its frozen contract."""


def _word(namespace: str, seed: int, serial: int, index: int) -> str:
    if not namespace.startswith("jrb1-"):
        raise JRB1DataError("JRB1 opaque-name namespace differs")
    return eal_word(namespace, seed, serial, index)


def _pairs(size_left: int, size_right: int, split: str) -> tuple[tuple[int, int], ...]:
    if split not in ("train", "development"):
        raise JRB1DataError("JRB1 renderer split differs")
    bucket = 0 if split == "train" else 1
    return tuple(
        (left, right)
        for left in range(size_left)
        for right in range(size_right)
        if (left + right) % 2 == bucket
    )


def render_initial_state(
    registers: Sequence[str],
    state: Sequence[int],
    *,
    split: str,
    serial: int,
    order: Sequence[int],
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    table = tuple(str(value) for value in registers)
    values = tuple(int(value) for value in state)
    order_tuple = tuple(int(value) for value in order)
    if (
        len(table) != REGISTERS
        or len(set(table)) != REGISTERS
        or len(values) != REGISTERS
        or any(value < 0 or value >= 97 for value in values)
        or sorted(order_tuple) != list(range(REGISTERS))
    ):
        raise JRB1DataError("JRB1 initial-state carrier differs")
    pairs = _pairs(len(_INITIAL_PREFIXES), len(_INITIAL_FRAMES), split)
    first = pairs[serial % len(pairs)]
    second = pairs[(serial + 3) % len(pairs)]
    clauses = []
    for target, (_, frame) in zip(order_tuple, (first, second), strict=True):
        clauses.append(
            _INITIAL_FRAMES[frame].format(
                register=table[target], value=values[target]
            )
        )
    text = f"{_INITIAL_PREFIXES[first[0]]}, {clauses[0]}; meanwhile, {clauses[1]}."
    if len(scan_integer_spans(text)) != REGISTERS:
        raise JRB1DataError("JRB1 initial state does not expose two integers")
    return text, order_tuple, (first, second)


def render_query(
    registers: Sequence[str],
    target: int,
    *,
    split: str,
    serial: int,
) -> tuple[str, tuple[int, int]]:
    table = tuple(str(value) for value in registers)
    target = int(target)
    if len(table) != REGISTERS or len(set(table)) != REGISTERS or target not in (0, 1):
        raise JRB1DataError("JRB1 query carrier differs")
    pairs = _pairs(len(_QUERY_PREFIXES), len(_QUERY_FRAMES), split)
    pair = pairs[serial % len(pairs)]
    text = f"{_QUERY_PREFIXES[pair[0]]}, " + _QUERY_FRAMES[pair[1]].format(
        register=table[target]
    )
    return text, pair


def _replace_names(text: str, source: Sequence[str], target: Sequence[str]) -> str:
    output = str(text)
    if len(source) != len(target) or len(set(source)) != len(source):
        raise JRB1DataError("JRB1 replacement table differs")
    placeholders = [f"zzjrbplaceholder{index}zz" for index in range(len(source))]
    replacements = 0
    for old, placeholder in zip(source, placeholders, strict=True):
        output, count = re.subn(
            rf"(?<![a-z]){re.escape(str(old))}(?![a-z])", placeholder, output
        )
        replacements += count
    if replacements < 1:
        raise JRB1DataError("JRB1 source omits the register table")
    for placeholder, new in zip(placeholders, target, strict=True):
        output = output.replace(placeholder, str(new))
    return output


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_training_record(serial: int) -> dict[str, Any]:
    rng = random.Random(canonical_sha256(["jrb1-training", TRAIN_SEED, serial]))
    operation = _word("jrb1-train-operation", TRAIN_SEED, serial, 0)
    registers = tuple(
        _word("jrb1-train-register", TRAIN_SEED, serial, index)
        for index in range(REGISTERS)
    )
    matrix = TRAIN_MATRICES[rng.randrange(len(TRAIN_MATRICES))]
    before = (rng.randrange(97), rng.randrange(97))
    transition = _transition_record(
        split="train",
        seed=TRAIN_SEED,
        serial=serial,
        operation_index=0,
        operation=operation,
        registers=registers,
        before=before,
        after=apply_matrix(matrix, before),
        rng=rng,
    )
    initial = (rng.randrange(97), rng.randrange(97))
    initial_order = [0, 1]
    rng.shuffle(initial_order)
    initial_text, initial_targets, initial_renderer = render_initial_state(
        registers,
        initial,
        split="train",
        serial=serial,
        order=initial_order,
    )
    query_target = rng.randrange(REGISTERS)
    query_text, query_renderer = render_query(
        registers, query_target, split="train", serial=serial
    )
    record = {
        "schema": TRAIN_SCHEMA,
        "seed": TRAIN_SEED,
        "serial": serial,
        "operation": operation,
        "registers": list(registers),
        "evidence_text": transition["source_text"],
        "evidence_sha256": transition["source_sha256"],
        "evidence_register_targets": [
            int(value) % REGISTERS for value in transition["numeric_role_ids"]
        ],
        "initial_text": initial_text,
        "initial_sha256": _hash(initial_text),
        "initial_register_targets": list(initial_targets),
        "initial_renderer": [list(value) for value in initial_renderer],
        "query_text": query_text,
        "query_sha256": _hash(query_text),
        "query_register_target": query_target,
        "query_renderer": list(query_renderer),
    }
    record["identity_sha256"] = canonical_sha256(record)
    validate_training_record(record)
    return record


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != TRAIN_SCHEMA or int(record.get("seed", -1)) != TRAIN_SEED:
        raise JRB1DataError("JRB1 training schema or seed differs")
    registers = tuple(str(value) for value in record["registers"])
    operation = str(record["operation"])
    evidence = str(record["evidence_text"])
    initial = str(record["initial_text"])
    query = str(record["query_text"])
    evidence_targets = tuple(int(value) for value in record["evidence_register_targets"])
    initial_targets = tuple(int(value) for value in record["initial_register_targets"])
    if (
        len(registers) != REGISTERS
        or len(set(registers)) != REGISTERS
        or not operation.isalpha()
        or not operation.islower()
        or len(scan_integer_spans(evidence)) != 4
        or len(scan_integer_spans(initial)) != 2
        or len(evidence_targets) != 4
        or sorted(evidence_targets) != [0, 0, 1, 1]
        or sorted(initial_targets) != [0, 1]
        or any(value not in (0, 1) for value in evidence_targets)
        or int(record["query_register_target"]) not in (0, 1)
        or _hash(evidence) != record["evidence_sha256"]
        or _hash(initial) != record["initial_sha256"]
        or _hash(query) != record["query_sha256"]
    ):
        raise JRB1DataError("JRB1 training geometry differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise JRB1DataError("JRB1 training identity differs")


def augment_evaluation_episode(
    serial: int, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_public, base_assessor = build_evaluation_episode(serial, seed=seed)
    validate_episode(base_public, base_assessor)
    aliases = tuple(str(value) for value in base_public["aliases"])
    registers = tuple(str(value) for value in base_public["registers"])
    renamed_aliases = tuple(
        _word("jrb1-renamed-operation", seed, serial, index)
        for index in range(OPERATIONS)
    )
    decoy_aliases = tuple(
        _word("jrb1-decoy-operation", seed, serial, index)
        for index in range(OPERATIONS)
    )
    renamed_registers = tuple(
        _word("jrb1-renamed-register", seed, serial, index)
        for index in range(REGISTERS)
    )
    decoy_registers = tuple(
        _word("jrb1-decoy-register", seed, serial, index)
        for index in range(REGISTERS)
    )
    evidence = []
    for item in base_public["evidence"]:
        renamed = _replace_names(item["source_text"], registers, renamed_registers)
        scrubbed = _replace_names(item["source_text"], registers, decoy_registers)
        evidence.append(
            {
                **item,
                "renamed_source_text": renamed,
                "renamed_source_sha256": _hash(renamed),
                "register_scrubbed_text": scrubbed,
                "register_scrubbed_sha256": _hash(scrubbed),
            }
        )

    transfer = []
    command_targets = []
    initial_targets = []
    for index, item in enumerate(base_public["transfer"]):
        targets = tuple(aliases.index(str(symbol)) for symbol in item["symbols"])
        command_serial = serial * 100 + index
        command, renderer = render_command(
            aliases, targets, split="development", seed=seed, serial=command_serial
        )
        renamed_command, _ = render_command(
            renamed_aliases,
            targets,
            split="development",
            seed=seed,
            serial=command_serial,
        )
        command_scrub, _ = render_command(
            decoy_aliases,
            targets,
            split="development",
            seed=seed,
            serial=command_serial,
        )
        order = (index + serial) % 2, (index + serial + 1) % 2
        initial_text, mention_targets, initial_renderer = render_initial_state(
            registers,
            item["initial_state"],
            split="development",
            serial=command_serial,
            order=order,
        )
        renamed_initial = _replace_names(initial_text, registers, renamed_registers)
        scrubbed_initial = _replace_names(initial_text, registers, decoy_registers)
        transfer.append(
            {
                "program_id": item["program_id"],
                "depth": item["depth"],
                "command_text": command,
                "command_sha256": _hash(command),
                "renamed_command_text": renamed_command,
                "renamed_command_sha256": _hash(renamed_command),
                "scrubbed_command_text": command_scrub,
                "scrubbed_command_sha256": _hash(command_scrub),
                "command_renderer": [list(value) for value in renderer],
                "initial_text": initial_text,
                "initial_sha256": _hash(initial_text),
                "renamed_initial_text": renamed_initial,
                "renamed_initial_sha256": _hash(renamed_initial),
                "register_scrubbed_initial_text": scrubbed_initial,
                "register_scrubbed_initial_sha256": _hash(scrubbed_initial),
                "initial_renderer": [list(value) for value in initial_renderer],
            }
        )
        command_targets.append(
            {"program_id": item["program_id"], "targets": list(targets)}
        )
        initial_targets.append(
            {
                "program_id": item["program_id"],
                "state": list(item["initial_state"]),
                "mention_register_targets": list(mention_targets),
            }
        )

    queries = []
    query_targets = []
    for index, item in enumerate(base_public["queries"]):
        target = int(item["register_index"])
        query, renderer = render_query(
            registers,
            target,
            split="development",
            serial=serial * 1000 + index,
        )
        renamed = _replace_names(query, registers, renamed_registers)
        scrubbed = _replace_names(query, registers, decoy_registers)
        queries.append(
            {
                "program_id": item["program_id"],
                "query_text": query,
                "query_sha256": _hash(query),
                "renamed_query_text": renamed,
                "renamed_query_sha256": _hash(renamed),
                "register_scrubbed_query_text": scrubbed,
                "register_scrubbed_query_sha256": _hash(scrubbed),
                "query_renderer": list(renderer),
            }
        )
        query_targets.append(
            {"program_id": item["program_id"], "register_index": target}
        )

    public = {
        "schema": PUBLIC_SCHEMA,
        "seed": seed,
        "serial": serial,
        "episode_id": base_public["episode_id"],
        "aliases": list(aliases),
        "renamed_aliases": list(renamed_aliases),
        "registers": list(registers),
        "renamed_registers": list(renamed_registers),
        "evidence": evidence,
        "transfer": transfer,
        "queries": queries,
    }
    public["identity_sha256"] = canonical_sha256(public)
    assessor = {
        **base_assessor,
        "schema": ASSESSOR_SCHEMA,
        "seed": seed,
        "serial": serial,
        "episode_id": base_public["episode_id"],
        "public_identity_sha256": public["identity_sha256"],
        "command_targets": command_targets,
        "initial_targets": initial_targets,
        "query_targets": query_targets,
    }
    assessor["identity_sha256"] = canonical_sha256(
        {key: value for key, value in assessor.items() if key != "identity_sha256"}
    )
    validate_evaluation_episode(public, assessor)
    return public, assessor


def validate_evaluation_episode(
    public: Mapping[str, Any], assessor: Mapping[str, Any]
) -> None:
    if public.get("schema") != PUBLIC_SCHEMA or assessor.get("schema") != ASSESSOR_SCHEMA:
        raise JRB1DataError("JRB1 evaluation schema differs")
    if (
        int(public.get("seed", -1)) != int(assessor.get("seed", -2))
        or int(public.get("serial", -1)) != int(assessor.get("serial", -2))
        or public.get("episode_id") != assessor.get("episode_id")
        or assessor.get("public_identity_sha256") != public.get("identity_sha256")
        or len(public["registers"]) != REGISTERS
        or len(set(public["registers"])) != REGISTERS
        or len(public["renamed_registers"]) != REGISTERS
        or len(set(public["renamed_registers"])) != REGISTERS
        or len(public["evidence"]) != OPERATIONS * 3
        or len(public["transfer"]) != len(assessor["transfer"])
        or len(public["transfer"]) != len(assessor["command_targets"])
        or len(public["transfer"]) != len(assessor["initial_targets"])
        or len(public["queries"]) != len(assessor["query_targets"])
        or "initial_state" in public["transfer"][0]
        or "register_index" in public["queries"][0]
    ):
        raise JRB1DataError("JRB1 evaluation geometry differs")
    for episode in (public, assessor):
        payload = dict(episode)
        identity = str(payload.pop("identity_sha256"))
        if canonical_sha256(payload) != identity:
            raise JRB1DataError("JRB1 evaluation identity differs")
    for item in public["evidence"]:
        if len(scan_integer_spans(str(item["source_text"]))) != 4:
            raise JRB1DataError("JRB1 evidence mention count differs")
        for key, digest_key in (
            ("source_text", "source_sha256"),
            ("renamed_source_text", "renamed_source_sha256"),
            ("register_scrubbed_text", "register_scrubbed_sha256"),
        ):
            if _hash(str(item[key])) != item[digest_key]:
                raise JRB1DataError("JRB1 evidence commitment differs")
    for item in public["transfer"]:
        if (
            len(scan_integer_spans(str(item["initial_text"]))) != 2
            or "symbols" in item
            or "initial_state" in item
        ):
            raise JRB1DataError("JRB1 public transfer leaks a typed carrier")
        for key, digest_key in (
            ("command_text", "command_sha256"),
            ("renamed_command_text", "renamed_command_sha256"),
            ("scrubbed_command_text", "scrubbed_command_sha256"),
            ("initial_text", "initial_sha256"),
            ("renamed_initial_text", "renamed_initial_sha256"),
            ("register_scrubbed_initial_text", "register_scrubbed_initial_sha256"),
        ):
            if _hash(str(item[key])) != item[digest_key]:
                raise JRB1DataError("JRB1 transfer commitment differs")
    for item in public["queries"]:
        if "register_index" in item:
            raise JRB1DataError("JRB1 public query leaks a register target")
        for key, digest_key in (
            ("query_text", "query_sha256"),
            ("renamed_query_text", "renamed_query_sha256"),
            ("register_scrubbed_query_text", "register_scrubbed_query_sha256"),
        ):
            if _hash(str(item[key])) != item[digest_key]:
                raise JRB1DataError("JRB1 query commitment differs")


def build_development_episode(serial: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return augment_evaluation_episode(serial, seed=DEVELOPMENT_SEED)


__all__ = [
    "ASSESSOR_SCHEMA",
    "CONFIRMATION_SEEDS",
    "DEVELOPMENT_EPISODES",
    "DEVELOPMENT_SEED",
    "JRB1DataError",
    "PUBLIC_SCHEMA",
    "REGISTERS",
    "REPORT_SCHEMA",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_evaluation_episode",
    "build_development_episode",
    "build_training_record",
    "render_initial_state",
    "render_query",
    "validate_evaluation_episode",
    "validate_training_record",
]
