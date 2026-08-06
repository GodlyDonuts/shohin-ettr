"""Deterministic natural query and confirmation data for DIVERGE-IEM1."""

from __future__ import annotations

import hashlib
import json
import random
import string
from typing import Any, Mapping, Sequence

from diverge_nve1_data import (
    MAX_EVIDENCE_BYTES,
    NUMERIC_ROLES,
    SYMBOL_ROLES,
    scan_rational_spans,
    symbol_occurrence_groups,
)
from diverge_tfs1_data import (
    FAULT_LINES,
    REGISTER_COUNT,
    TFS1_NAMES,
    validate_row as validate_tfs1_row,
)


QUERY_TRAIN_SCHEMA = "shohin-diverge-iem1-query-training-v1"
BOARD_SCHEMA = "shohin-diverge-iem1-board-v1"
TRAIN_SEED = 2026080614
BOARD_SEED = 2026080615
QUERY_TRAIN_ROWS = 50_000
BOARD_ROWS = 256
MAX_QUERY_BYTES = MAX_EVIDENCE_BYTES


class IEM1DataError(RuntimeError):
    """An IEM1 data record violates the frozen contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _nonce(rng: random.Random, width: int = 10) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(width))


def _query_training_text(
    renderer: int,
    *,
    nonce: str,
    target: str,
    distractor: str,
) -> str:
    templates = (
        "For audit {nonce}, report register {target}, not decoy register {distractor}.",
        "Regarding {nonce}, ignore {distractor}; return the value in {target}.",
        "Case {nonce} requests {target}; {distractor} is only a decoy.",
        "Do not read {distractor} for {nonce}. The requested register is {target}.",
        "Use register {target} in audit {nonce}, while rejecting {distractor}.",
        "Audit {nonce}: reject {distractor} and answer from register {target}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise IEM1DataError("IEM1 query training renderer differs") from error
    return template.format(nonce=nonce, target=target, distractor=distractor)


def _query_confirmation_text(
    renderer: int,
    *,
    target: str,
    distractor: str,
) -> str:
    templates = (
        "Do not answer from {distractor}; the requested final register is {target}.",
        "Between {target} and decoy {distractor}, return the value belonging to "
        "the former.",
        "Treat {distractor} only as an audit decoy and use {target} for the answer.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise IEM1DataError("IEM1 query confirmation renderer differs") from error
    return template.format(target=target, distractor=distractor)


def _evidence_confirmation_text(
    renderer: int,
    *,
    step: int,
    value: str,
    target: str,
    distractor: str,
) -> str:
    # These are the protected NVE1 confirmation layouts. They are absent from
    # NVE1/IEM1 training and preserve the immutable separate-model ceiling.
    templates = (
        "Value {value} is certified for verified register {target}; reject decoy "
        "register {distractor} after instruction {step}.",
        "Once instruction {step} ends, decoy register {distractor} is rejected and "
        "verified register {target} reads value {value}.",
        "Reject decoy register {distractor}; following instruction {step}, value "
        "{value} belongs to verified register {target}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise IEM1DataError("IEM1 evidence confirmation renderer differs") from error
    return template.format(
        step=step,
        value=value,
        target=target,
        distractor=distractor,
    )


def _symbol_role_ids(
    text: str,
    symbols: Sequence[str],
    *,
    target: str,
    distractor: str,
) -> list[int]:
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2 or {value[0] for value in groups} != {target, distractor}:
        raise IEM1DataError("IEM1 source does not expose the two required symbols")
    role = {
        target: SYMBOL_ROLES.index("TARGET"),
        distractor: SYMBOL_ROLES.index("DISTRACTOR"),
    }
    return [role[symbol] for symbol, _ in groups]


def _numeric_role_ids(text: str, *, renderer: int) -> list[int]:
    spans = scan_rational_spans(text)
    if len(spans) != 2:
        raise IEM1DataError("IEM1 evidence does not expose two numeric mentions")
    orders = (
        (NUMERIC_ROLES.index("VALUE"), NUMERIC_ROLES.index("STEP")),
        (NUMERIC_ROLES.index("STEP"), NUMERIC_ROLES.index("VALUE")),
        (NUMERIC_ROLES.index("STEP"), NUMERIC_ROLES.index("VALUE")),
    )
    return list(orders[renderer])


def generate_query_training_records(
    count: int = QUERY_TRAIN_ROWS,
    seed: int = TRAIN_SEED,
) -> list[dict[str, Any]]:
    if count != QUERY_TRAIN_ROWS or seed != TRAIN_SEED:
        raise IEM1DataError("IEM1 query training geometry differs")
    rng = random.Random(seed ^ 0x49454D31)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(output) < count:
        index = len(output)
        renderer = index % 6
        symbols = tuple(rng.sample(TFS1_NAMES, REGISTER_COUNT))
        target, distractor = rng.sample(symbols, 2)
        text = _query_training_text(
            renderer,
            nonce=_nonce(rng),
            target=target,
            distractor=distractor,
        )
        if text in seen:
            continue
        seen.add(text)
        record: dict[str, Any] = {
            "schema": QUERY_TRAIN_SCHEMA,
            "split": "train",
            "renderer": renderer,
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
            "symbols": list(symbols),
            "target": target,
            "distractor": distractor,
            "symbol_role_ids": _symbol_role_ids(
                text,
                symbols,
                target=target,
                distractor=distractor,
            ),
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_query_training_record(record)
        output.append(record)
    return output


def validate_query_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != QUERY_TRAIN_SCHEMA or record.get("split") != "train":
        raise IEM1DataError("IEM1 query training schema differs")
    text = str(record["source_text"])
    if not text or len(text.encode("ascii")) + 1 > MAX_QUERY_BYTES:
        raise IEM1DataError("IEM1 query source width differs")
    if hashlib.sha256(text.encode("ascii")).hexdigest() != record["source_sha256"]:
        raise IEM1DataError("IEM1 query source commitment differs")
    symbols = tuple(str(value) for value in record["symbols"])
    if len(symbols) != REGISTER_COUNT or len(set(symbols)) != REGISTER_COUNT:
        raise IEM1DataError("IEM1 query symbol table differs")
    target = str(record["target"])
    distractor = str(record["distractor"])
    if target == distractor or target not in symbols or distractor not in symbols:
        raise IEM1DataError("IEM1 query role symbols differ")
    expected = _symbol_role_ids(
        text,
        symbols,
        target=target,
        distractor=distractor,
    )
    if list(record["symbol_role_ids"]) != expected:
        raise IEM1DataError("IEM1 query role assignment differs")
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise IEM1DataError("IEM1 query identity differs")


def augment_confirmation_board(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = BOARD_SEED,
) -> list[dict[str, Any]]:
    if len(rows) != BOARD_ROWS or seed != BOARD_SEED:
        raise IEM1DataError("IEM1 confirmation geometry differs")
    rng = random.Random(seed ^ 0x49454D31)
    output: list[dict[str, Any]] = []
    for row_index, tfs1 in enumerate(rows):
        symbols = tuple(str(value) for value in tfs1["symbols"])  # type: ignore[arg-type]
        evidence_items: list[dict[str, Any]] = []
        for evidence_index, typed in enumerate(tfs1["evidence"]):  # type: ignore[union-attr]
            target = str(typed["register"])
            distractor = rng.choice(
                tuple(value for value in symbols if value != target)
            )
            step_ordinal = int(typed["step_index"]) + 1
            value = str(typed["value"])
            renderer = (row_index * FAULT_LINES + evidence_index) % 3
            text = _evidence_confirmation_text(
                renderer,
                step=step_ordinal,
                value=value,
                target=target,
                distractor=distractor,
            )
            item: dict[str, Any] = {
                "index": evidence_index,
                "renderer": renderer,
                "source_text": text,
                "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                "step_ordinal": step_ordinal,
                "step_index": int(typed["step_index"]),
                "value": value,
                "target": target,
                "distractor": distractor,
                "numeric_role_ids": _numeric_role_ids(text, renderer=renderer),
                "symbol_role_ids": _symbol_role_ids(
                    text,
                    symbols,
                    target=target,
                    distractor=distractor,
                ),
            }
            item["identity_sha256"] = canonical_sha256(item)
            evidence_items.append(item)

        queries: dict[str, dict[str, Any]] = {}
        for query_offset, query_name in enumerate(
            ("sensitive", "invariant", "underdetermined")
        ):
            target = str(tfs1["query_registers"][query_name])  # type: ignore[index]
            distractor = rng.choice(
                tuple(value for value in symbols if value != target)
            )
            renderer = (row_index * 3 + query_offset) % 3
            text = _query_confirmation_text(
                renderer,
                target=target,
                distractor=distractor,
            )
            item = {
                "name": query_name,
                "renderer": renderer,
                "source_text": text,
                "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
                "target": target,
                "distractor": distractor,
                "symbol_role_ids": _symbol_role_ids(
                    text,
                    symbols,
                    target=target,
                    distractor=distractor,
                ),
            }
            item["identity_sha256"] = canonical_sha256(item)
            queries[query_name] = item

        record: dict[str, Any] = {
            "schema": BOARD_SCHEMA,
            "split": "confirmation",
            "tfs1": tfs1,
            "natural_evidence": evidence_items,
            "natural_queries": queries,
            "selection": {
                "model_score_used": False,
                "fresh_tfs1_seed": seed,
                "held_renderers_only": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_board_row(record)
        output.append(record)
    return output


def validate_board_row(record: Mapping[str, Any]) -> None:
    if record.get("schema") != BOARD_SCHEMA or record.get("split") != "confirmation":
        raise IEM1DataError("IEM1 board schema differs")
    tfs1 = record["tfs1"]
    if not isinstance(tfs1, Mapping):
        raise IEM1DataError("IEM1 typed board is absent")
    validate_tfs1_row(tfs1)
    symbols = tuple(str(value) for value in tfs1["symbols"])
    evidence = record["natural_evidence"]
    if not isinstance(evidence, Sequence) or len(evidence) != FAULT_LINES:
        raise IEM1DataError("IEM1 evidence count differs")
    for index, (item, typed) in enumerate(zip(evidence, tfs1["evidence"], strict=True)):
        if int(item["index"]) != index or int(item["renderer"]) not in range(3):
            raise IEM1DataError("IEM1 evidence index or renderer differs")
        if (
            int(item["step_index"]) != int(typed["step_index"])
            or int(item["step_ordinal"]) != int(typed["step_index"]) + 1
            or str(item["target"]) != str(typed["register"])
            or str(item["value"]) != str(typed["value"])
        ):
            raise IEM1DataError("IEM1 evidence supervisor differs")
        text = str(item["source_text"])
        if len(text.encode("ascii")) + 1 > MAX_EVIDENCE_BYTES:
            raise IEM1DataError("IEM1 evidence source width differs")
        if hashlib.sha256(text.encode("ascii")).hexdigest() != item["source_sha256"]:
            raise IEM1DataError("IEM1 evidence source commitment differs")
        expected_numeric = _numeric_role_ids(text, renderer=int(item["renderer"]))
        expected_symbols = _symbol_role_ids(
            text,
            symbols,
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        if list(item["numeric_role_ids"]) != expected_numeric:
            raise IEM1DataError("IEM1 evidence numeric roles differ")
        if list(item["symbol_role_ids"]) != expected_symbols:
            raise IEM1DataError("IEM1 evidence symbol roles differ")

    queries = record["natural_queries"]
    if not isinstance(queries, Mapping) or set(queries) != {
        "sensitive",
        "invariant",
        "underdetermined",
    }:
        raise IEM1DataError("IEM1 query set differs")
    for name, item in queries.items():
        target = str(tfs1["query_registers"][name])
        if str(item["target"]) != target or int(item["renderer"]) not in range(3):
            raise IEM1DataError("IEM1 query supervisor differs")
        text = str(item["source_text"])
        if len(text.encode("ascii")) + 1 > MAX_QUERY_BYTES:
            raise IEM1DataError("IEM1 query source width differs")
        if hashlib.sha256(text.encode("ascii")).hexdigest() != item["source_sha256"]:
            raise IEM1DataError("IEM1 query source commitment differs")
        expected = _symbol_role_ids(
            text,
            symbols,
            target=target,
            distractor=str(item["distractor"]),
        )
        if list(item["symbol_role_ids"]) != expected:
            raise IEM1DataError("IEM1 query role assignment differs")

    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise IEM1DataError("IEM1 board identity differs")


__all__ = [
    "BOARD_ROWS",
    "BOARD_SCHEMA",
    "BOARD_SEED",
    "IEM1DataError",
    "MAX_QUERY_BYTES",
    "QUERY_TRAIN_ROWS",
    "QUERY_TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_confirmation_board",
    "canonical_sha256",
    "generate_query_training_records",
    "validate_board_row",
    "validate_query_training_record",
]
