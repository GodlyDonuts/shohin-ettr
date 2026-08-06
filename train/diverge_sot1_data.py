"""Fresh confirmation data for DIVERGE-SOT1 stage-owned transactions."""

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


SOT1_BOARD_SEED = 2026080617
SOT1_BOARD_ROWS = 256


class SOT1DataError(RuntimeError):
    """A SOT1 confirmation record violates the frozen contract."""


def query_confirmation_text(
    renderer: int,
    *,
    target: str,
    distractor: str,
) -> str:
    templates = (
        "At termination read {target}; exclude the alternate register {distractor}.",
        "The final answer belongs to {target}, whereas {distractor} is irrelevant.",
        "Select the terminal value from {target} and disregard {distractor}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise SOT1DataError("SOT1 query renderer differs") from error
    return template.format(target=target, distractor=distractor)


def augment_sot1_board(
    rows: Sequence[dict[str, object]],
    *,
    seed: int = SOT1_BOARD_SEED,
) -> list[dict[str, Any]]:
    if len(rows) != SOT1_BOARD_ROWS or seed != SOT1_BOARD_SEED:
        raise SOT1DataError("SOT1 confirmation geometry differs")
    rng = random.Random(seed ^ 0x534F5431)
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
            text = query_confirmation_text(
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
                "sot1_fresh_query_surfaces": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_sot1_board_row(record)
        output.append(record)
    identities = [str(row["identity_sha256"]) for row in output]
    if len(set(identities)) != len(identities):
        raise SOT1DataError("SOT1 confirmation identities are not unique")
    return output


def validate_sot1_board_row(record: Mapping[str, Any]) -> None:
    validate_board_row(record)
    selection = record.get("selection")
    if not isinstance(selection, Mapping):
        raise SOT1DataError("SOT1 selection receipt is absent")
    if (
        selection.get("model_score_used") is not False
        or int(selection.get("fresh_tfs1_seed", -1)) != SOT1_BOARD_SEED
        or selection.get("sot1_fresh_query_surfaces") is not True
    ):
        raise SOT1DataError("SOT1 selection receipt differs")
    for item in record["natural_queries"].values():
        expected = query_confirmation_text(
            int(item["renderer"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        if str(item["source_text"]) != expected:
            raise SOT1DataError("SOT1 query surface differs")
