"""Fresh label-balanced confirmation board for DIVERGE-CGL1."""

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


SCHEMA = "shohin-diverge-cgl1-confirmation-board-v1"
BOARD_SEED = 2026080703
BOARD_ROWS = 256
NAMES = (
    "amaranth", "bluefin", "cloudspire", "duskfall", "evergreen",
    "firecrest", "goldfinch", "hearthstone", "ironwood", "javelin",
    "kingfisher", "lumen", "moonstone", "nightjar", "opaline", "pinecone",
    "quicklime", "riverbend", "starling", "thunderhead", "ultraviolet",
    "verdant", "windward", "xenolith", "youngberry", "zinnia", "ashgrove",
    "bramble", "copperleaf", "driftwood", "elmshade", "frostline",
)
MODES = ("sensitive", "invariant", "underdetermined")


class CGL1ConfirmationDataError(RuntimeError):
    """A CGL1 confirmation episode violates the frozen contract."""


def query_text(
    renderer: int,
    order: int,
    *,
    target: str,
    distractor: str,
) -> str:
    templates = (
        (
            "For the final answer, select {target}; do not select {distractor}.",
            "Do not select {distractor}; select {target} for the final answer.",
        ),
        (
            "The requested source is {target}, while {distractor} is a decoy.",
            "While {distractor} is a decoy, the requested source is {target}.",
        ),
        (
            "Read the answer from {target}; exclude the unrelated {distractor}.",
            "Exclude the unrelated {distractor}; read the answer from {target}.",
        ),
        (
            "Use {target} rather than the rejected register {distractor}.",
            "Reject register {distractor}; use {target} instead.",
        ),
        (
            "The value to report belongs to {target}, not to {distractor}.",
            "The value does not belong to {distractor}; report it from {target}.",
        ),
        (
            "Return {target} as the answer source and ignore {distractor}.",
            "Ignore {distractor} and return {target} as the answer source.",
        ),
    )
    if renderer not in range(len(templates)) or order not in (0, 1):
        raise CGL1ConfirmationDataError("CGL1 confirmation renderer differs")
    return templates[renderer][order].format(
        target=target, distractor=distractor
    )


def generate_confirmation_board(
    *,
    seed: int = BOARD_SEED,
    count: int = BOARD_ROWS,
) -> list[dict[str, Any]]:
    if seed != BOARD_SEED or count != BOARD_ROWS:
        raise CGL1ConfirmationDataError("CGL1 confirmation geometry differs")
    typed_rows = generate_board(count, seed, name_bank=NAMES)
    rng = random.Random(seed ^ 0x43474C31)
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
                    text, symbols, target=target, distractor=distractor
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
                    text, symbols, target=target, distractor=distractor
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
                "generated_before_cgl1_development_result": True,
                "renderer_role_order_balanced": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_confirmation_row(record)
        output.append(record)

    if renderer_counts != Counter({renderer: 128 for renderer in range(6)}):
        raise CGL1ConfirmationDataError("CGL1 renderer counts differ")
    order_counts = Counter(
        (int(query["renderer"]), int(query["order"]))
        for row in output
        for query in row["natural_queries"].values()
    )
    if order_counts != Counter(
        {(renderer, order): 64 for renderer in range(6) for order in (0, 1)}
    ):
        raise CGL1ConfirmationDataError("CGL1 renderer/order balance differs")
    if len({str(row["identity_sha256"]) for row in output}) != BOARD_ROWS:
        raise CGL1ConfirmationDataError("CGL1 confirmation identities collide")
    return output


def validate_confirmation_row(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity or record.get("schema") != SCHEMA:
        raise CGL1ConfirmationDataError("CGL1 confirmation identity differs")
    selection = record.get("selection")
    if not isinstance(selection, Mapping) or selection != {
        "seed": BOARD_SEED,
        "model_score_used": False,
        "generated_before_cgl1_development_result": True,
        "renderer_role_order_balanced": True,
    }:
        raise CGL1ConfirmationDataError("CGL1 confirmation selection differs")
    validate_row(record["tfs1"])
    symbols = tuple(str(value) for value in record["tfs1"]["symbols"])
    typed_evidence = record["tfs1"]["evidence"]
    natural_evidence = record["natural_evidence"]
    if len(typed_evidence) != FAULT_LINES or len(natural_evidence) != FAULT_LINES:
        raise CGL1ConfirmationDataError("CGL1 confirmation evidence count differs")
    for index, item in enumerate(natural_evidence):
        typed = typed_evidence[index]
        target = str(item["target"])
        distractor = str(item["distractor"])
        renderer = int(item["renderer"])
        step_ordinal = int(item["step_ordinal"])
        value = str(item["value"])
        expected = _evidence_confirmation_text(
            renderer,
            step=step_ordinal,
            value=value,
            target=target,
            distractor=distractor,
        )
        payload = dict(item)
        item_identity = str(payload.pop("identity_sha256"))
        if (
            int(item["index"]) != index
            or int(item["step_index"]) != int(typed["step_index"])
            or step_ordinal != int(typed["step_index"]) + 1
            or target != str(typed["register"])
            or value != str(typed["value"])
            or str(item["source_text"]) != expected
            or str(item["source_sha256"])
            != hashlib.sha256(expected.encode("ascii")).hexdigest()
            or item_identity != canonical_sha256(payload)
        ):
            raise CGL1ConfirmationDataError("CGL1 confirmation evidence differs")
        if list(item["numeric_role_ids"]) != _numeric_role_ids(
            expected, renderer=renderer
        ) or list(item["symbol_role_ids"]) != _symbol_role_ids(
            expected, symbols, target=target, distractor=distractor
        ):
            raise CGL1ConfirmationDataError("CGL1 confirmation evidence roles differ")
    for mode in MODES:
        item = record["natural_queries"][mode]
        expected = query_text(
            int(item["renderer"]),
            int(item["order"]),
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        if str(item["source_text"]) != expected:
            raise CGL1ConfirmationDataError("CGL1 confirmation query differs")
        query_payload = dict(item)
        query_identity = str(query_payload.pop("identity_sha256"))
        if (
            query_identity != canonical_sha256(query_payload)
            or str(item["source_sha256"])
            != hashlib.sha256(expected.encode("ascii")).hexdigest()
        ):
            raise CGL1ConfirmationDataError("CGL1 confirmation query identity differs")
        if list(item["symbol_role_ids"]) != _symbol_role_ids(
            expected,
            symbols,
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        ):
            raise CGL1ConfirmationDataError("CGL1 confirmation roles differ")


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
