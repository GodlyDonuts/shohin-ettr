#!/usr/bin/env python3
"""Plan one preregistered operation-effect successor without submitting work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from route_operation_effect_set_result import ROUTE_SCHEMA
from train_ettr_component_island import _canonical_bytes, _write_no_replace


PLAN_SCHEMA = "shohin-ettr-operation-effect-successor-plan-v1"
V15_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v15"
V18_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v18"
V19_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v19"
V20_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v20"
V21_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v21"


class OperationEffectSuccessorPlanError(RuntimeError):
    """A route receipt cannot authorize one exact bounded successor."""


_TRANSITIONS = {
    (V15_SCHEMA, "operation_family_island_curriculum"): {
        "family_gate": True,
        "family_island": True,
        "family_state_binding": False,
        "output_tag": "v19-family-island-u1000-a31-d11",
        "successor_schema": V19_SCHEMA,
        "warm_start": False,
    },
    (V19_SCHEMA, "joint_operation_family_rail_release"): {
        "family_gate": True,
        "family_island": False,
        "family_state_binding": False,
        "output_tag": "v18-family-joint-u1000-a31-d11",
        "successor_schema": V18_SCHEMA,
        "warm_start": True,
    },
    (V19_SCHEMA, "operation_role_state_bilinear_arbiter"): {
        "family_gate": True,
        "family_island": True,
        "family_state_binding": True,
        "output_tag": "v20-state-bound-family-u1000-a31-d11",
        "successor_schema": V20_SCHEMA,
        "warm_start": False,
    },
    (V20_SCHEMA, "joint_state_bound_family_rail_release"): {
        "family_gate": True,
        "family_island": False,
        "family_state_binding": True,
        "output_tag": "v21-state-bound-joint-u1000-a31-d11",
        "successor_schema": V21_SCHEMA,
        "warm_start": True,
    },
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OperationEffectSuccessorPlanError(f"{label} differs")
    return value


def plan_successor(
    route_receipt: Mapping[str, object],
    terminal_contract: Mapping[str, object],
) -> dict[str, object]:
    """Return one bounded transition or an explicit terminal stop."""

    if route_receipt.get("schema") != ROUTE_SCHEMA:
        raise OperationEffectSuccessorPlanError("operation effect route differs")
    route = route_receipt.get("route")
    contract_schema = terminal_contract.get("schema")
    if not isinstance(route, str) or contract_schema not in {
        V15_SCHEMA,
        V18_SCHEMA,
        V19_SCHEMA,
        V20_SCHEMA,
        V21_SCHEMA,
    }:
        raise OperationEffectSuccessorPlanError(
            "operation effect route lineage differs"
        )
    if route_receipt.get("terminal_contract_schema") != contract_schema:
        raise OperationEffectSuccessorPlanError(
            "operation effect route contract differs"
        )
    transition = _TRANSITIONS.get((str(contract_schema), route))
    if transition is None:
        return {
            "action": "stop",
            "predecessor_schema": contract_schema,
            "reason": route_receipt.get("reason"),
            "route": route,
            "schema": PLAN_SCHEMA,
        }
    return {
        "action": "submit_scientific_successor",
        "architecture_seed": 31,
        "data_seed": 11,
        "family_gate": transition["family_gate"],
        "family_island": transition["family_island"],
        "family_state_binding": transition["family_state_binding"],
        "output_tag": transition["output_tag"],
        "predecessor_schema": contract_schema,
        "route": route,
        "schema": PLAN_SCHEMA,
        "successor_schema": transition["successor_schema"],
        "updates": 1000,
        "warm_start": transition["warm_start"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--terminal-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        route = _mapping(json.loads(args.route.read_text()), "route receipt")
        contract = _mapping(
            json.loads(args.terminal_contract.read_text()),
            "terminal contract",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OperationEffectSuccessorPlanError(
            "operation effect successor input differs"
        ) from exc
    payload = plan_successor(route, contract)
    _write_no_replace(args.output, _canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
