#!/usr/bin/env python3
"""Route a sealed effect-set result to one preregistered successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-eval-report-v2"
ROUTE_SCHEMA = "shohin-ettr-operation-effect-set-route-v1"
UNANCHORED_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v12"
ROLE_ANCHORED_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v13"
CARDINALITY_GATED_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v14"
WRITE_LINK_RAIL_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v15"
RAIL_LOCAL_EFFECT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v16"
POST_WRITE_LINK_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v17"
OPERATION_FAMILY_GATE_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v18"


class OperationEffectRouteError(RuntimeError):
    """The measured report cannot support an exact branch decision."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OperationEffectRouteError(f"{label} differs")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OperationEffectRouteError(f"{label} differs")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise OperationEffectRouteError(f"{label} differs")
    return result


def _phase_local(report: Mapping[str, object], phase: str) -> Mapping[str, object]:
    diagnostics = _mapping(
        report.get("operation_effect_diagnostics"),
        "operation effect diagnostics",
    )
    return _mapping(diagnostics.get(phase), f"{phase} operation effects")


def _exact_rate(local: Mapping[str, object], name: str) -> float:
    rates = _mapping(local.get("exact_rates"), "operation effect exact rates")
    return _number(rates.get(name), f"operation effect {name}")


def _optional_exact_rate(local: Mapping[str, object], name: str) -> float | None:
    rates = _mapping(local.get("exact_rates"), "operation effect exact rates")
    value = rates.get(name)
    if value is None:
        return None
    return _number(value, f"operation effect {name}")


def _positive_rate(local: Mapping[str, object], name: str) -> float | None:
    rates = _mapping(
        local.get("positive_exact_rates"),
        "operation effect positive exact rates",
    )
    value = rates.get(name)
    if value is None:
        return None
    return _number(value, f"operation effect positive {name}")


def _diagnostic_rate(local: Mapping[str, object], name: str) -> float | None:
    rates = local.get("diagnostic_rates")
    if rates is None:
        return None
    value = _mapping(rates, "operation effect diagnostic rates").get(name)
    if value is None:
        return None
    return _number(value, f"operation effect diagnostic {name}")


def _strict_rate(report: Mapping[str, object], phase: str, factor: str) -> float:
    evaluation = _mapping(report.get("evaluation"), "evaluation")
    phase_value = _mapping(evaluation.get(phase), f"{phase} evaluation")
    arms = _mapping(phase_value.get("arms"), "evaluation arms")
    autonomous = _mapping(
        arms.get("autonomous_program_autonomous_state"),
        "fully autonomous arm",
    )
    causal = _mapping(
        autonomous.get("source_deleted_causal"),
        "source-deleted causal evaluation",
    )
    summary = _mapping(causal.get(factor), f"{factor} causal evaluation")
    return _number(
        summary.get("paired_order_joint_rate"),
        f"{factor} strict paired rate",
    )


def _kind_shares(local: Mapping[str, object]) -> tuple[float, float]:
    histogram = _mapping(
        local.get("predicted_kind_histogram"),
        "predicted effect kind histogram",
    )
    values = []
    for name, value in histogram.items():
        if not isinstance(name, str) or not isinstance(value, int) or value < 0:
            raise OperationEffectRouteError("predicted effect kind histogram differs")
        values.append((name, value))
    total = sum(value for _name, value in values)
    if total <= 0:
        raise OperationEffectRouteError("predicted effect kind histogram is empty")
    noop = next((value for name, value in values if name == "0"), 0) / total
    dominant = max(value for _name, value in values) / total
    return noop, dominant


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_schema(
    report: Mapping[str, object],
    terminal_contract: Mapping[str, object] | None,
) -> str | None:
    if terminal_contract is None:
        return None
    receipt = _mapping(report.get("terminal_state_receipt"), "terminal state receipt")
    expected = receipt.get("contract_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise OperationEffectRouteError("terminal state contract receipt differs")
    schema = terminal_contract.get("schema")
    if schema not in {
        UNANCHORED_SCHEMA,
        ROLE_ANCHORED_SCHEMA,
        CARDINALITY_GATED_SCHEMA,
        WRITE_LINK_RAIL_SCHEMA,
        RAIL_LOCAL_EFFECT_SCHEMA,
        POST_WRITE_LINK_SCHEMA,
        OPERATION_FAMILY_GATE_SCHEMA,
    }:
        raise OperationEffectRouteError("terminal state effect contract differs")
    return str(schema)


def route_result(
    report: Mapping[str, object],
    terminal_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return exactly one mechanism-level successor from measured deltas."""

    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "pass":
        raise OperationEffectRouteError("operation effect evaluation report differs")
    before = _phase_local(report, "before")
    after = _phase_local(report, "after")
    before_set = _exact_rate(before, "complete_effect_set_exact")
    after_set = _exact_rate(after, "complete_effect_set_exact")
    before_dense = _exact_rate(before, "complete_dense_edit_exact")
    after_dense = _exact_rate(after, "complete_dense_edit_exact")
    before_terminal = _number(
        before.get("terminal_state_exact_rate"),
        "before terminal state exact rate",
    )
    after_terminal = _number(
        after.get("terminal_state_exact_rate"),
        "after terminal state exact rate",
    )
    before_world = _strict_rate(report, "before", "world")
    after_world = _strict_rate(report, "after", "world")
    before_command = _strict_rate(report, "before", "command")
    after_command = _strict_rate(report, "after", "command")
    noop_share, dominant_share = _kind_shares(after)
    entity = _positive_rate(after, "entity")
    relation_link = _positive_rate(after, "relation_link")
    operation_family = _optional_exact_rate(after, "operation_family_exact")
    family_conflict = _diagnostic_rate(
        after,
        "predicted_operation_family_conflict",
    )

    deltas = {
        "command_strict": after_command - before_command,
        "complete_dense_edit_exact": after_dense - before_dense,
        "complete_effect_set_exact": after_set - before_set,
        "terminal_state_exact": after_terminal - before_terminal,
        "world_strict": after_world - before_world,
    }
    measured = {
        "after_command_strict": after_command,
        "after_complete_dense_edit_exact": after_dense,
        "after_complete_effect_set_exact": after_set,
        "after_terminal_state_exact": after_terminal,
        "after_world_strict": after_world,
        "dominant_kind_share": dominant_share,
        "entity_positive_exact": entity,
        "noop_share": noop_share,
        "operation_family_exact": operation_family,
        "operation_family_conflict": family_conflict,
        "relation_link_positive_exact": relation_link,
    }

    local_exact_gain = after_set > before_set and after_set > 0.0
    terminal_gain = after_terminal > before_terminal and after_terminal > 0.0
    both_strict_gain = (
        after_world > before_world
        and after_world > 0.0
        and after_command > before_command
        and after_command > 0.0
    )
    contract_schema = _contract_schema(report, terminal_contract)
    if local_exact_gain and terminal_gain and both_strict_gain:
        route = "replicate_fresh_population"
        reason = "local effects, terminal state, WORLD, and COMMAND all improved"
    elif (
        contract_schema == WRITE_LINK_RAIL_SCHEMA
        and operation_family is not None
        and (
            operation_family < 0.9
            or (family_conflict is not None and family_conflict > 0.01)
        )
    ):
        route = "exclusive_operation_family_gate"
        reason = (
            "independent rails violate or miss the corpus-exact NONE/WRITE/LINK "
            "operation family; select one family before releasing rail payloads"
        )
    elif (
        contract_schema == OPERATION_FAMILY_GATE_SCHEMA
        and operation_family is not None
        and operation_family < 0.9
    ):
        route = "operation_family_island_curriculum"
        reason = (
            "the exclusive architecture is correct but its family controller is "
            "not yet exact; freeze rail payloads and isolate family acquisition"
        )
    elif noop_share >= 0.9 or dominant_share >= 0.9:
        if contract_schema == ROLE_ANCHORED_SCHEMA:
            route = "explicit_effect_cardinality_gate"
            reason = (
                "role-bound motors still collapsed; separate exact total cardinality, "
                "motor activity, and non-NOOP kind selection"
            )
        elif contract_schema == CARDINALITY_GATED_SCHEMA:
            route = "write_link_typed_rails"
            reason = (
                "global cardinality did not prevent kind collapse; remove the generic "
                "kind decision and separate WRITE/LINK counts, motors, and payloads"
            )
        elif contract_schema in {
            WRITE_LINK_RAIL_SCHEMA,
            RAIL_LOCAL_EFFECT_SCHEMA,
            POST_WRITE_LINK_SCHEMA,
            OPERATION_FAMILY_GATE_SCHEMA,
        }:
            route = "rail_local_pointer_payload_islands"
            reason = (
                "typed rails still selected zero or one dominant outcome; isolate count, "
                "pointer, and payload acquisition inside each rail"
            )
        else:
            route = "public_ast_role_anchored_effect_queries"
            reason = (
                "anonymous hard effect kinds collapsed to NOOP or one dominant class"
            )
    elif (
        entity is not None
        and relation_link is not None
        and entity >= 0.5
        and relation_link <= 0.25
        and entity - relation_link >= 0.25
    ):
        route = "two_phase_entity_then_relation_algebra"
        reason = "entity effects are accurate while relation additions remain wrong"
    elif (
        (local_exact_gain or after_dense > before_dense)
        and terminal_gain
        and after_world == 0.0
        and after_command == 0.0
    ):
        route = "crossed_state_sufficiency_isolation"
        reason = "local/state exactness improved without fully autonomous causal gain"
    else:
        route = "reject_unordered_effect_set"
        reason = "no preregistered exact local or causal advancement gate moved"
    return {
        "deltas": deltas,
        "measured": measured,
        "reason": reason,
        "route": route,
        "schema": ROUTE_SCHEMA,
        "terminal_contract_schema": contract_schema,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--terminal-contract", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    terminal_contract = None
    if args.terminal_contract is not None:
        terminal_contract = _mapping(
            json.loads(args.terminal_contract.read_text(encoding="utf-8")),
            "terminal state contract",
        )
        receipt = _mapping(
            report.get("terminal_state_receipt"), "terminal state receipt"
        )
        if _sha256_file(args.terminal_contract) != receipt.get("contract_sha256"):
            raise OperationEffectRouteError("terminal state contract hash differs")
    result = route_result(_mapping(report, "evaluation report"), terminal_contract)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
