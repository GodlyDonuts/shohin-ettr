"""Deterministic natural-variable evidence data for DIVERGE-NVE1."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import random
import re
from typing import Any, Mapping, Sequence

from diverge_tfs1_data import (
    FAULT_LINES,
    REGISTER_COUNT,
    TFS1_NAMES,
    validate_row as validate_tfs1_row,
)
from diverge_tol1_ir import format_fraction, parse_fraction


TRAIN_SCHEMA = "shohin-diverge-nve1-evidence-training-v1"
BOARD_SCHEMA = "shohin-diverge-nve1-board-v1"
TRAIN_SEED = 2026080610
BOARD_SEED = 2026080611
TRAIN_ROWS = 50_000
BOARD_ROWS = 256
NUMERIC_ROLES = ("STEP", "VALUE")
SYMBOL_ROLES = ("TARGET", "DISTRACTOR")
MAX_EVIDENCE_BYTES = 192

_RATIONAL = re.compile(r"(?<![A-Za-z0-9_/])[+-]?\d+(?:/\d+)?(?![A-Za-z0-9_/])")


class NVE1DataError(RuntimeError):
    """A natural evidence record violates the frozen NVE1 contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def scan_rational_spans(text: str) -> tuple[tuple[int, int], ...]:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise NVE1DataError("NVE1 evidence is not ASCII") from error
    return tuple(match.span() for match in _RATIONAL.finditer(text))


def symbol_occurrence_groups(
    text: str,
    symbols: Sequence[str],
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    """Return source-owned symbol groups ordered by first occurrence."""

    groups: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for symbol in symbols:
        if not symbol.isalpha() or not symbol.islower():
            raise NVE1DataError("NVE1 source symbol is not a lowercase word")
        pattern = re.compile(rf"(?<![a-z]){re.escape(symbol)}(?![a-z])")
        spans = tuple(match.span() for match in pattern.finditer(text))
        if spans:
            groups.append((symbol, spans))
    groups.sort(key=lambda item: (item[1][0][0], item[0]))
    return tuple(groups)


def _render_training(
    renderer: int,
    *,
    step: int,
    value: str,
    target: str,
    distractor: str,
) -> str:
    templates = (
        "After instruction {step}, verified register {target}, not decoy register "
        "{distractor}, contains value {value}.",
        "At step {step}, reject register {distractor}; value {value} belongs to "
        "verified register {target}.",
        "Verified register {target} holds value {value}, not decoy register "
        "{distractor}, once instruction {step} completes.",
        "Decoy register {distractor} is irrelevant: value {value} belongs to "
        "verified register {target} after step {step}.",
        "Instruction {step} leaves verified register {target} at value {value}; "
        "keep {target} and reject register {distractor}.",
        "Reject register {distractor}; value {value} is not there. Following step "
        "{step}, ignore {distractor} and use verified register {target}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise NVE1DataError("NVE1 training renderer differs") from error
    return template.format(
        step=step,
        value=value,
        target=target,
        distractor=distractor,
    )


def _render_confirmation(
    renderer: int,
    *,
    step: int,
    value: str,
    target: str,
    distractor: str,
) -> str:
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
        raise NVE1DataError("NVE1 confirmation renderer differs") from error
    return template.format(
        step=step,
        value=value,
        target=target,
        distractor=distractor,
    )


_TRAIN_NUMERIC_ROLE_ORDER = (
    (0, 1),
    (0, 1),
    (1, 0),
    (1, 0),
    (0, 1),
    (1, 0),
)
_TRAIN_SYMBOL_ROLE_ORDER = (
    (0, 1),
    (1, 0),
    (0, 1),
    (1, 0),
    (0, 1),
    (1, 0),
)
_CONFIRM_NUMERIC_ROLE_ORDER = ((1, 0), (0, 1), (0, 1))
_CONFIRM_SYMBOL_ROLE_ORDER = ((0, 1), (1, 0), (1, 0))


def _random_value(rng: random.Random) -> str:
    numerator = rng.randint(-32, 32)
    denominator = rng.randint(1, 7)
    return format_fraction(Fraction(numerator, denominator))


def generate_training_records(
    count: int = TRAIN_ROWS,
    seed: int = TRAIN_SEED,
) -> list[dict[str, Any]]:
    if count != TRAIN_ROWS or seed != TRAIN_SEED:
        raise NVE1DataError("NVE1 training geometry differs")
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(output) < count:
        index = len(output)
        renderer = index % 6
        symbols = tuple(rng.sample(TFS1_NAMES, REGISTER_COUNT))
        target, distractor = rng.sample(symbols, 2)
        step = rng.randint(1, 40)
        value = _random_value(rng)
        text = _render_training(
            renderer,
            step=step,
            value=value,
            target=target,
            distractor=distractor,
        )
        if text in seen:
            continue
        record: dict[str, Any] = {
            "schema": TRAIN_SCHEMA,
            "renderer": renderer,
            "source_text": text,
            "source_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
            "symbols": list(symbols),
            "step_ordinal": step,
            "value": value,
            "target": target,
            "distractor": distractor,
            "numeric_role_ids": list(_TRAIN_NUMERIC_ROLE_ORDER[renderer]),
            "symbol_role_ids": list(_TRAIN_SYMBOL_ROLE_ORDER[renderer]),
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_training_record(record)
        output.append(record)
        seen.add(text)
    return output


def _validate_role_bindings(
    record: Mapping[str, Any],
    *,
    numeric_roles: Sequence[int],
    symbol_roles: Sequence[int],
) -> None:
    text = str(record["source_text"])
    numeric_spans = scan_rational_spans(text)
    if len(numeric_spans) != 2 or tuple(int(value) for value in numeric_roles) not in (
        (0, 1),
        (1, 0),
    ):
        raise NVE1DataError("NVE1 numeric role geometry differs")
    numeric_by_role = {
        NUMERIC_ROLES[int(role)]: text[start:end]
        for (start, end), role in zip(numeric_spans, numeric_roles, strict=True)
    }
    try:
        step = int(numeric_by_role["STEP"])
        value = format_fraction(parse_fraction(numeric_by_role["VALUE"]))
    except (ValueError, TypeError) as error:
        raise NVE1DataError("NVE1 numeric role parse differs") from error
    if step != int(record["step_ordinal"]) or value != str(record["value"]):
        raise NVE1DataError("NVE1 numeric roles disagree with supervision")

    symbols = tuple(str(value) for value in record["symbols"])
    groups = symbol_occurrence_groups(text, symbols)
    if len(groups) != 2 or tuple(int(value) for value in symbol_roles) not in (
        (0, 1),
        (1, 0),
    ):
        raise NVE1DataError("NVE1 symbol role geometry differs")
    symbol_by_role = {
        SYMBOL_ROLES[int(role)]: group[0]
        for group, role in zip(groups, symbol_roles, strict=True)
    }
    if symbol_by_role != {
        "TARGET": str(record["target"]),
        "DISTRACTOR": str(record["distractor"]),
    }:
        raise NVE1DataError("NVE1 symbol roles disagree with supervision")


def validate_training_record(record: Mapping[str, Any]) -> None:
    if record.get("schema") != TRAIN_SCHEMA:
        raise NVE1DataError("NVE1 training schema differs")
    text = str(record["source_text"])
    if not text or len(text.encode("ascii")) + 1 > MAX_EVIDENCE_BYTES:
        raise NVE1DataError("NVE1 training source width differs")
    if hashlib.sha256(text.encode("ascii")).hexdigest() != record["source_sha256"]:
        raise NVE1DataError("NVE1 training source commitment differs")
    renderer = int(record["renderer"])
    if text != _render_training(
        renderer,
        step=int(record["step_ordinal"]),
        value=str(record["value"]),
        target=str(record["target"]),
        distractor=str(record["distractor"]),
    ):
        raise NVE1DataError("NVE1 training renderer differs")
    symbols = tuple(str(value) for value in record["symbols"])
    if len(symbols) != REGISTER_COUNT or len(set(symbols)) != REGISTER_COUNT:
        raise NVE1DataError("NVE1 training symbol table differs")
    if any(symbol not in TFS1_NAMES for symbol in symbols):
        raise NVE1DataError("NVE1 training symbol is outside the source vocabulary")
    if str(record["target"]) == str(record["distractor"]):
        raise NVE1DataError("NVE1 training target equals distractor")
    _validate_role_bindings(
        record,
        numeric_roles=record["numeric_role_ids"],
        symbol_roles=record["symbol_role_ids"],
    )
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise NVE1DataError("NVE1 training identity differs")


def augment_confirmation_board(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = BOARD_SEED,
) -> list[dict[str, Any]]:
    if len(rows) != BOARD_ROWS or seed != BOARD_SEED:
        raise NVE1DataError("NVE1 confirmation geometry differs")
    rng = random.Random(seed ^ 0x4E564531)
    output: list[dict[str, Any]] = []
    for row_index, tfs1 in enumerate(rows):
        validate_tfs1_row(tfs1)
        symbols = tuple(str(value) for value in tfs1["symbols"])  # type: ignore[arg-type]
        natural: list[dict[str, Any]] = []
        for evidence_index, typed in enumerate(tfs1["evidence"]):  # type: ignore[union-attr]
            target = str(typed["register"])
            distractor = rng.choice(
                tuple(value for value in symbols if value != target)
            )
            step_ordinal = int(typed["step_index"]) + 1
            value = str(typed["value"])
            renderer = (row_index * FAULT_LINES + evidence_index) % 3
            text = _render_confirmation(
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
                "numeric_role_ids": list(_CONFIRM_NUMERIC_ROLE_ORDER[renderer]),
                "symbol_role_ids": list(_CONFIRM_SYMBOL_ROLE_ORDER[renderer]),
            }
            item["identity_sha256"] = canonical_sha256(item)
            natural.append(item)
        record: dict[str, Any] = {
            "schema": BOARD_SCHEMA,
            "split": "confirmation",
            "identity_sha256": canonical_sha256(
                {
                    "tfs1_identity": tfs1["identity_sha256"],
                    "natural_evidence": natural,
                }
            ),
            "tfs1": tfs1,
            "natural_evidence": natural,
            "selection": {
                "model_score_used": False,
                "fresh_tfs1_seed": seed,
                "confirmation_renderer_only": True,
            },
        }
        validate_board_row(record)
        output.append(record)
    return output


def validate_board_row(record: Mapping[str, Any]) -> None:
    if record.get("schema") != BOARD_SCHEMA or record.get("split") != "confirmation":
        raise NVE1DataError("NVE1 board schema differs")
    tfs1 = record["tfs1"]
    if not isinstance(tfs1, Mapping):
        raise NVE1DataError("NVE1 typed board is absent")
    validate_tfs1_row(tfs1)
    natural = record["natural_evidence"]
    if not isinstance(natural, Sequence) or len(natural) != FAULT_LINES:
        raise NVE1DataError("NVE1 natural evidence count differs")
    symbols = tuple(str(value) for value in tfs1["symbols"])
    typed = tfs1["evidence"]
    for index, (item, oracle) in enumerate(zip(natural, typed, strict=True)):
        if not isinstance(item, Mapping):
            raise NVE1DataError("NVE1 natural evidence item differs")
        renderer = int(item["renderer"])
        if int(item["index"]) != index or renderer not in range(3):
            raise NVE1DataError("NVE1 evidence index or renderer differs")
        if (
            int(item["step_index"]) != int(oracle["step_index"])
            or int(item["step_ordinal"]) != int(oracle["step_index"]) + 1
            or str(item["target"]) != str(oracle["register"])
            or str(item["value"]) != str(oracle["value"])
        ):
            raise NVE1DataError("NVE1 evidence supervisor differs")
        if str(item["distractor"]) not in symbols or str(item["distractor"]) == str(
            item["target"]
        ):
            raise NVE1DataError("NVE1 confirmation distractor differs")
        text = str(item["source_text"])
        if not text or len(text.encode("ascii")) + 1 > MAX_EVIDENCE_BYTES:
            raise NVE1DataError("NVE1 confirmation source width differs")
        if text != _render_confirmation(
            renderer,
            step=int(item["step_ordinal"]),
            value=str(item["value"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        ):
            raise NVE1DataError("NVE1 confirmation renderer differs")
        if hashlib.sha256(text.encode("ascii")).hexdigest() != item["source_sha256"]:
            raise NVE1DataError("NVE1 evidence source commitment differs")
        _validate_role_bindings(
            {**item, "symbols": symbols},
            numeric_roles=item["numeric_role_ids"],
            symbol_roles=item["symbol_role_ids"],
        )
        payload = dict(item)
        identity = str(payload.pop("identity_sha256"))
        if canonical_sha256(payload) != identity:
            raise NVE1DataError("NVE1 evidence identity differs")
    expected_identity = canonical_sha256(
        {
            "tfs1_identity": tfs1["identity_sha256"],
            "natural_evidence": list(natural),
        }
    )
    if record["identity_sha256"] != expected_identity:
        raise NVE1DataError("NVE1 board identity differs")


__all__ = [
    "BOARD_ROWS",
    "BOARD_SCHEMA",
    "BOARD_SEED",
    "MAX_EVIDENCE_BYTES",
    "NUMERIC_ROLES",
    "NVE1DataError",
    "SYMBOL_ROLES",
    "TRAIN_ROWS",
    "TRAIN_SCHEMA",
    "TRAIN_SEED",
    "augment_confirmation_board",
    "canonical_sha256",
    "generate_training_records",
    "scan_rational_spans",
    "symbol_occurrence_groups",
    "validate_board_row",
    "validate_training_record",
]
