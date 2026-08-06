"""Fresh source-disjoint confirmation data for DIVERGE-CCR1."""

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


CCR1_BOARD_SEED = 2026080622
CCR1_BOARD_ROWS = 256
CCR1_NAMES = (
    "asteroid", "birchwood", "coralstone", "dawnmist", "evergreen",
    "firecrest", "goldfinch", "hazelwood", "ironbark", "juniper",
    "kingfisher", "larkspur", "moonstone", "nightjar", "oceanmist",
    "pinecone", "quartzite", "riverbend", "starling", "thunderhead",
    "upland", "violetwood", "willowmere", "xenolith", "yarrow", "zircon",
    "ashgrove", "bramble", "craneberry", "duskwater", "elmshade",
    "frostline",
)


class CCR1DataError(RuntimeError):
    """A CCR1 confirmation row violates the frozen contract."""


def query_text(renderer: int, *, target: str, distractor: str) -> str:
    templates = (
        "When answering, choose {target}; {distractor} is the decoy.",
        "Discard {distractor}; the answer comes from {target}.",
        "Read {target} for this request and exclude {distractor}.",
        "Instead of decoy {distractor}, return the value held by {target}.",
        "Only {target} supplies the requested result; not {distractor}.",
        "Treat {distractor} as irrelevant and report {target}.",
    )
    try:
        template = templates[renderer]
    except IndexError as error:
        raise CCR1DataError("CCR1 query renderer differs") from error
    return template.format(target=target, distractor=distractor)


def augment_board(
    rows: Sequence[dict[str, object]], *, seed: int = CCR1_BOARD_SEED
) -> list[dict[str, Any]]:
    if len(rows) != CCR1_BOARD_ROWS or seed != CCR1_BOARD_SEED:
        raise CCR1DataError("CCR1 confirmation geometry differs")
    rng = random.Random(seed ^ 0x43435231)
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
                    text, symbols, target=target, distractor=distractor
                ),
            }
            item["identity_sha256"] = canonical_sha256(item)
            evidence_items.append(item)

        queries: dict[str, dict[str, Any]] = {}
        for query_offset, mode in enumerate(("sensitive", "invariant", "underdetermined")):
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
                    text, symbols, target=target, distractor=distractor
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
                "ccr1_counterfactual_candidate_gate": True,
                "mode_renderer_deconfounded": True,
            },
        }
        record["identity_sha256"] = canonical_sha256(record)
        validate_ccr1_board_row(record)
        output.append(record)
    if len({str(row["identity_sha256"]) for row in output}) != len(output):
        raise CCR1DataError("CCR1 confirmation identities are not unique")
    return output


def validate_ccr1_board_row(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    identity = str(payload.pop("identity_sha256"))
    if canonical_sha256(payload) != identity:
        raise CCR1DataError("CCR1 board identity differs")
    selection = record.get("selection")
    if not isinstance(selection, Mapping):
        raise CCR1DataError("CCR1 selection receipt is absent")
    if (
        selection.get("model_score_used") is not False
        or int(selection.get("fresh_tfs1_seed", -1)) != CCR1_BOARD_SEED
        or selection.get("ccr1_counterfactual_candidate_gate") is not True
        or selection.get("mode_renderer_deconfounded") is not True
    ):
        raise CCR1DataError("CCR1 selection receipt differs")
    for item in record["natural_queries"].values():
        renderer = int(item["renderer"])
        if renderer not in range(6):
            raise CCR1DataError("CCR1 query renderer differs")
        expected = query_text(
            renderer,
            target=str(item["target"]),
            distractor=str(item["distractor"]),
        )
        if str(item["source_text"]) != expected:
            raise CCR1DataError("CCR1 query surface differs")
    compatible = dict(record)
    compatible["natural_queries"] = {
        name: {**item, "renderer": int(item["renderer"]) % 3}
        for name, item in record["natural_queries"].items()
    }
    compatible_payload = dict(compatible)
    compatible_payload.pop("identity_sha256")
    compatible["identity_sha256"] = canonical_sha256(compatible_payload)
    validate_board_row(compatible)


__all__ = [
    "CCR1_BOARD_ROWS",
    "CCR1_BOARD_SEED",
    "CCR1_NAMES",
    "augment_board",
    "query_text",
    "validate_ccr1_board_row",
]
