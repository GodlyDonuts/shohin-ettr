"""Fresh source-disjoint confirmation board for DIVERGE-EIC1."""

from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Any, Mapping

from diverge_iem1_data import (
    _evidence_confirmation_text,
    _numeric_role_ids,
    _symbol_role_ids,
    canonical_sha256,
)
from diverge_tfs1_data import FAULT_LINES, generate_board, validate_row


SCHEMA = "shohin-diverge-eic1-confirmation-board-v1"
BOARD_SEED = 2026080713
BOARD_ROWS = 256
NAMES = tuple(
    f"eix{chr(ord('a') + index // 26)}{chr(ord('a') + index % 26)}q"
    for index in range(32)
)
MODES = ("sensitive", "invariant", "underdetermined")


class EIC1ConfirmationDataError(RuntimeError):
    """An EIC1 confirmation episode violates the frozen contract."""


def query_text(
    renderer: int,
    order: int,
    *,
    target: str,
    distractor: str,
) -> str:
    templates = (
        (
            "Report from {target}; treat {distractor} as irrelevant.",
            "Treat {distractor} as irrelevant; report from {target}.",
        ),
        (
            "Consult {target} for the result, whereas {distractor} is noise.",
            "Whereas {distractor} is noise, consult {target} for the result.",
        ),
        (
            "Only {target} supplies the requested value; omit {distractor}.",
            "Omit {distractor}; only {target} supplies the requested value.",
        ),
        (
            "Take the response from {target}, with {distractor} excluded.",
            "With {distractor} excluded, take the response from {target}.",
        ),
        (
            "The answer is sourced by {target}; {distractor} is not relevant.",
            "{distractor} is not relevant; the answer is sourced by {target}.",
        ),
        (
            "Choose {target} as the reporting register, never {distractor}.",
            "Never choose {distractor}; choose {target} as the reporting register.",
        ),
    )
    if renderer not in range(len(templates)) or order not in (0, 1):
        raise EIC1ConfirmationDataError("EIC1 confirmation renderer differs")
    return templates[renderer][order].format(
        target=target,
        distractor=distractor,
    )


def generate_confirmation_board(
    *,
    seed: int = BOARD_SEED,
    count: int = BOARD_ROWS,
) -> list[dict[str, Any]]:
    if seed != BOARD_SEED or count != BOARD_ROWS:
        raise EIC1ConfirmationDataError("EIC1 confirmation geometry differs")
    typed_rows = generate_board(count, seed, name_bank=NAMES)
    rng = random.Random(seed ^ 0x45494331)
    renderer_counts = Counter()
    output = []
    for row_index, typed in enumerate(typed_rows):
        symbols = tuple(str(value) for value in typed["symbols"])
        evidence_items = []
        for evidence_index, evidence in enumerate(typed["evidence"]):
            target = str(evidence["register"])
            distractor = rng.choice(tuple(value for value in symbols if value != target))
            step_ordinal = int(evidence["step_index"]) + 1
            value = str(evidence["value"])
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
                "step_index": int(evidence["step_index"]),
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
        for mode_offset, mode in enumerate(MODES):
            target = str(typed["query_registers"][mode])
            distractor = rng.choice(tuple(value for value in symbols if value != target))
            renderer = (row_index + 2 * mode_offset) % 6
            order = renderer_counts[renderer] % 2
            renderer_counts[renderer] += 1
            text = query_text(
                renderer,
                order,
                target=target,
                distractor=distractor,
            )
            item = {
                "name": mode,
                "renderer": renderer,
                "order": order,
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
            "schema": SCHEMA,
            "split": "confirmation",
            "tfs1": typed,
            "natural_evidence": evidence_items,
            "natural_queries": queries,
            "selection": {
                "seed": seed,
                "model_score_used": False,
                "generated_before_eic1_development_result": True,
                "renderer_role_order_balanced": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_confirmation_row(record)
        output.append(record)

    if renderer_counts != Counter({renderer: 128 for renderer in range(6)}):
        raise EIC1ConfirmationDataError("EIC1 renderer counts differ")
    order_counts = Counter(
        (int(query["renderer"]), int(query["order"]))
        for row in output
        for query in row["natural_queries"].values()
    )
    if order_counts != Counter(
        {(renderer, order): 64 for renderer in range(6) for order in (0, 1)}
    ):
        raise EIC1ConfirmationDataError("EIC1 renderer/order balance differs")
    if len({str(row["identity_sha256"]) for row in output}) != BOARD_ROWS:
        raise EIC1ConfirmationDataError("EIC1 confirmation identities collide")
    return output


def validate_confirmation_row(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity or record.get("schema") != SCHEMA:
        raise EIC1ConfirmationDataError("EIC1 confirmation identity differs")
    if record.get("selection") != {
        "seed": BOARD_SEED,
        "model_score_used": False,
        "generated_before_eic1_development_result": True,
        "renderer_role_order_balanced": True,
    }:
        raise EIC1ConfirmationDataError("EIC1 confirmation selection differs")
    validate_row(record["tfs1"])
    symbols = tuple(str(value) for value in record["tfs1"]["symbols"])
    if len(record["natural_evidence"]) != FAULT_LINES:
        raise EIC1ConfirmationDataError("EIC1 confirmation evidence count differs")
    for index, item in enumerate(record["natural_evidence"]):
        typed = record["tfs1"]["evidence"][index]
        expected = _evidence_confirmation_text(
            int(item["renderer"]),
            step=int(item["step_ordinal"]),
            value=str(item["value"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        item_payload = dict(item)
        item_identity = str(item_payload.pop("identity_sha256"))
        if (
            int(item["index"]) != index
            or int(item["step_index"]) != int(typed["step_index"])
            or int(item["step_ordinal"]) != int(typed["step_index"]) + 1
            or str(item["target"]) != str(typed["register"])
            or str(item["value"]) != str(typed["value"])
            or str(item["source_text"]) != expected
            or str(item["source_sha256"])
            != hashlib.sha256(expected.encode("ascii")).hexdigest()
            or item_identity != canonical_sha256(item_payload)
            or list(item["numeric_role_ids"])
            != _numeric_role_ids(expected, renderer=int(item["renderer"]))
            or list(item["symbol_role_ids"])
            != _symbol_role_ids(
                expected,
                symbols,
                target=str(item["target"]),
                distractor=str(item["distractor"]),
            )
        ):
            raise EIC1ConfirmationDataError("EIC1 confirmation evidence differs")
    for mode in MODES:
        item = record["natural_queries"][mode]
        expected = query_text(
            int(item["renderer"]),
            int(item["order"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        item_payload = dict(item)
        item_identity = str(item_payload.pop("identity_sha256"))
        if (
            str(item["source_text"]) != expected
            or str(item["source_sha256"])
            != hashlib.sha256(expected.encode("ascii")).hexdigest()
            or item_identity != canonical_sha256(item_payload)
            or list(item["symbol_role_ids"])
            != _symbol_role_ids(
                expected,
                symbols,
                target=str(item["target"]),
                distractor=str(item["distractor"]),
            )
        ):
            raise EIC1ConfirmationDataError("EIC1 confirmation query differs")


__all__ = [
    "BOARD_ROWS",
    "BOARD_SEED",
    "MODES",
    "NAMES",
    "SCHEMA",
    "generate_confirmation_board",
    "query_text",
    "validate_confirmation_row",
]
