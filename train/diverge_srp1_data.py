"""Fresh, mode-renderer-deconfounded confirmation data for DIVERGE-SRP1."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

from diverge_iem1_data import (
    BOARD_SCHEMA,
    _evidence_confirmation_text,
    _numeric_role_ids,
    _symbol_role_ids,
    canonical_sha256,
    validate_board_row,
)
from diverge_tfs1_data import FAULT_LINES


SRP1_BOARD_SEED = 2026080620
SRP1_BOARD_ROWS = 256
SRP1_NAMES = (
    "ambergris", "brooklet", "copperleaf", "daybreak", "estuary",
    "featherstone", "grovelet", "harborlight", "ivorywood", "juncture",
    "keystone", "meadowlark", "northwind", "opaline", "pebble",
    "quicklime", "rosewood", "sunbeam", "tidepool", "umbra", "vermilion",
    "windward", "xylitol", "yellowtail", "zephyr", "bellflower",
    "cloudbank", "driftwood", "fieldstone", "greenbriar", "hillcrest",
    "inkwell",
)


class SRP1DataError(RuntimeError):
    """An SRP1 confirmation row violates the frozen contract."""


def query_text(
    renderer: int,
    *,
    target: str,
    distractor: str,
) -> str:
    templates = (
        "Answer from register {target}; reject decoy register {distractor}.",
        "Ignore {distractor}; report the requested value in {target}.",
        "Use {target} for the final audit, not decoy register {distractor}.",
        "Do not answer from {distractor}; return the value in {target}.",
        "The requested register is {target}; {distractor} is only a decoy.",
        "Reject {distractor} and answer from register {target}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise SRP1DataError("SRP1 query renderer differs") from error
    return template.format(target=target, distractor=distractor)


def augment_board(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = SRP1_BOARD_SEED,
) -> list[dict[str, Any]]:
    if len(rows) != SRP1_BOARD_ROWS or seed != SRP1_BOARD_SEED:
        raise SRP1DataError("SRP1 confirmation geometry differs")
    rng = random.Random(seed ^ 0x53525031)
    output = []
    for row_index, tfs1 in enumerate(rows):
        symbols = tuple(str(value) for value in tfs1["symbols"])  # type: ignore[arg-type]
        evidence_items = []
        for evidence_index, typed in enumerate(tfs1["evidence"]):  # type: ignore[union-attr]
            target = str(typed["register"])
            distractor = rng.choice(tuple(value for value in symbols if value != target))
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
        for query_offset, mode in enumerate(
            ("sensitive", "invariant", "underdetermined")
        ):
            target = str(tfs1["query_registers"][mode])  # type: ignore[index]
            distractor = rng.choice(tuple(value for value in symbols if value != target))
            renderer = (row_index + 2 * query_offset) % 6
            text = query_text(renderer, target=target, distractor=distractor)
            item = {
                "name": mode,
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
            queries[mode] = item

        record: dict[str, Any] = {
            "schema": BOARD_SCHEMA,
            "split": "confirmation",
            "tfs1": tfs1,
            "natural_evidence": evidence_items,
            "natural_queries": queries,
            "selection": {
                "model_score_used": False,
                "fresh_tfs1_seed": seed,
                "srp1_semantic_referent_gate": True,
                "mode_renderer_deconfounded": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_srp1_board_row(record)
        output.append(record)
    if len({str(row["identity_sha256"]) for row in output}) != len(output):
        raise SRP1DataError("SRP1 confirmation identities are not unique")
    return output


def validate_srp1_board_row(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise SRP1DataError("SRP1 board identity differs")
    compatible = dict(record)
    compatible["natural_queries"] = {
        name: {**item, "renderer": int(item["renderer"]) % 3}
        for name, item in record["natural_queries"].items()
    }
    compatible_payload = dict(compatible)
    compatible_payload.pop("identity_sha256")
    compatible["identity_sha256"] = canonical_sha256(compatible_payload)
    validate_board_row(compatible)
    selection = record.get("selection")
    if not isinstance(selection, Mapping):
        raise SRP1DataError("SRP1 selection receipt is absent")
    if (
        selection.get("model_score_used") is not False
        or int(selection.get("fresh_tfs1_seed", -1)) != SRP1_BOARD_SEED
        or selection.get("srp1_semantic_referent_gate") is not True
        or selection.get("mode_renderer_deconfounded") is not True
    ):
        raise SRP1DataError("SRP1 selection receipt differs")
    for item in record["natural_queries"].values():
        if int(item["renderer"]) not in range(6):
            raise SRP1DataError("SRP1 query renderer differs")
        expected = query_text(
            int(item["renderer"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        if str(item["source_text"]) != expected:
            raise SRP1DataError("SRP1 query surface differs")


__all__ = [
    "SRP1_BOARD_ROWS",
    "SRP1_BOARD_SEED",
    "SRP1_NAMES",
    "augment_board",
    "query_text",
    "validate_srp1_board_row",
]
