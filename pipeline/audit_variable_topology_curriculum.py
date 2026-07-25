"""Audit the five-seed variable-topology semantic-type qualification."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


FAMILIES = ("affine_modular", "bitwise_rotate_xor", "permutation")
SEEDS = tuple(range(20260725, 20260730))
CELLS = (
    "collision",
    "composition",
    "joint",
    "law",
    "renderer",
    "topology",
)
ARMS = ("treatment", "direction_shuffled", "type_shuffled")


class VariableTopologyCurriculumAuditError(ValueError):
    """Raised when a report leaves the qualification contract."""


def _read(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VariableTopologyCurriculumAuditError(
            "report is not JSON"
        ) from exc
    if not isinstance(report, dict):
        raise VariableTopologyCurriculumAuditError("report root differs")
    return report, sha256(payload).hexdigest()


def audit_reports(input_dir: Path) -> dict[str, object]:
    expected = {
        (seed, family)
        for seed in SEEDS
        for family in FAMILIES
    }
    observed: dict[tuple[int, str], tuple[Path, dict[str, object], str]] = {}
    for path in sorted(input_dir.glob("semantic_type_holdout_*_seed*.json")):
        report, digest = _read(path)
        try:
            key = (int(report["seed"]), str(report["held_out_family"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise VariableTopologyCurriculumAuditError(
                "report identity differs"
            ) from exc
        if key in observed:
            raise VariableTopologyCurriculumAuditError(
                "duplicate report identity"
            )
        observed[key] = (path, report, digest)
    if set(observed) != expected:
        raise VariableTopologyCurriculumAuditError(
            "report matrix is incomplete"
        )

    totals = {
        arm: {"correct": 0, "invalid": 0, "total": 0}
        for arm in ARMS
    }
    per_family = {
        family: {
            arm: {"correct": 0, "total": 0}
            for arm in ARMS
        }
        for family in FAMILIES
    }
    positive_directions = {
        "direction_shuffled": 0,
        "type_shuffled": 0,
    }
    preparation_calls = 0
    manifest: list[dict[str, object]] = []
    for seed, family in sorted(observed):
        path, report, digest = observed[(seed, family)]
        if (
            report.get("status")
            != "variable_topology_semantic_type_curriculum"
            or report.get("candidate_time_oracle_calls") != 0
            or report.get("candidate_time_search_calls") != 0
            or report.get("candidate_time_verifier_calls") != 0
        ):
            raise VariableTopologyCurriculumAuditError(
                "candidate custody counters differ"
            )
        budget = report.get("equal_budget")
        parameters = report.get("parameter_receipt")
        if (
            not isinstance(budget, dict)
            or budget.get("base_rows") != 40
            or budget.get("counterfactual_rows") != 160
            or budget.get("initialization_identical") is not True
            or budget.get("optimizer_updates_per_arm") != 300
            or not isinstance(parameters, dict)
            or parameters.get("learned_compiler") != 152_933
            or parameters.get("complete_system") != 125_234_597
        ):
            raise VariableTopologyCurriculumAuditError(
                "budget or parameter receipt differs"
            )
        development = report.get("development")
        if not isinstance(development, dict):
            raise VariableTopologyCurriculumAuditError(
                "development report differs"
            )
        for arm in ARMS:
            result = development.get(arm)
            if (
                not isinstance(result, dict)
                or result.get("total") != 24
                or not isinstance(result.get("exact"), int)
                or not isinstance(result.get("invalid"), int)
            ):
                raise VariableTopologyCurriculumAuditError(
                    "arm report differs"
                )
            totals[arm]["correct"] += int(result["exact"])
            totals[arm]["invalid"] += int(result["invalid"])
            totals[arm]["total"] += int(result["total"])
            per_family[family][arm]["correct"] += int(result["exact"])
            per_family[family][arm]["total"] += int(result["total"])
        treatment = development["treatment"]
        cells = treatment.get("cell_exact")
        if (
            treatment.get("exact") != 24
            or treatment.get("invalid") != 0
            or treatment.get("collision_exact") != 8
            or treatment.get("collision_total") != 8
            or not isinstance(cells, dict)
            or any(
                not isinstance(cells.get(cell), dict)
                or cells[cell].get("correct") != 4
                or cells[cell].get("total") != 4
                for cell in CELLS
            )
        ):
            raise VariableTopologyCurriculumAuditError(
                "treatment cell gate differs"
            )
        for control in ("direction_shuffled", "type_shuffled"):
            if int(treatment["exact"]) <= int(development[control]["exact"]):
                raise VariableTopologyCurriculumAuditError(
                    "paired treatment direction is not positive"
                )
            positive_directions[control] += 1
        preparation_calls += int(report["preparation_exact_parser_calls"])
        manifest.append(
            {
                "family": family,
                "file": path.name,
                "seed": seed,
                "sha256": digest,
            }
        )

    treatment = totals["treatment"]
    if treatment != {"correct": 360, "invalid": 0, "total": 360}:
        raise VariableTopologyCurriculumAuditError(
            "aggregate treatment gate differs"
        )
    treatment_rate = treatment["correct"] / treatment["total"]
    margins = {
        control: (
            treatment_rate
            - totals[control]["correct"] / totals[control]["total"]
        )
        for control in ("direction_shuffled", "type_shuffled")
    }
    if any(margin < 0.10 for margin in margins.values()):
        raise VariableTopologyCurriculumAuditError(
            "aggregate causal margin differs"
        )
    return {
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "decision": (
            "global_semantic_partition_passes_variable_topology_gate"
        ),
        "manifest": manifest,
        "margins_over_control": margins,
        "parameter_receipt": {
            "complete_system": 125_234_597,
            "learned_compiler": 152_933,
        },
        "per_family": per_family,
        "positive_seed_fold_directions": {
            control: {"correct": count, "total": len(expected)}
            for control, count in positive_directions.items()
        },
        "preparation_exact_parser_calls": preparation_calls,
        "scope_boundary": (
            "systematic complete-table finite-machine compilation across "
            "variable topology; not incomplete-law induction, open-ended "
            "planning, or unrestricted natural-language reasoning"
        ),
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_reports(args.input_dir)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
