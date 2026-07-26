"""Independent audit of the five-seed renderer-curriculum qualification."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


FAMILIES = ("affine_modular", "bitwise_rotate_xor", "permutation")
SEEDS = tuple(range(20260725, 20260730))
CELLS = ("composition", "joint", "law", "renderer")


class RendererCurriculumAuditError(ValueError):
    """Raised when a report leaves the frozen qualification contract."""


def _read(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        report = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RendererCurriculumAuditError("report is not JSON") from exc
    if not isinstance(report, dict):
        raise RendererCurriculumAuditError("report root differs")
    return report, sha256(payload).hexdigest()


def audit_reports(input_dir: Path) -> dict[str, object]:
    expected = {
        (seed, family)
        for seed in SEEDS
        for family in FAMILIES
    }
    observed: dict[tuple[int, str], tuple[Path, dict[str, object], str]] = {}
    for path in sorted(
        input_dir.glob("renderer_curriculum_holdout_*_seed*.json")
    ):
        report, digest = _read(path)
        try:
            key = (int(report["seed"]), str(report["held_out_family"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RendererCurriculumAuditError(
                "report identity differs"
            ) from exc
        if key in observed:
            raise RendererCurriculumAuditError("duplicate report identity")
        observed[key] = (path, report, digest)
    if set(observed) != expected:
        raise RendererCurriculumAuditError("report matrix is incomplete")

    treatment_total = 0
    treatment_correct = 0
    control_total = 0
    control_correct = 0
    positive_directions = 0
    preparation_calls = 0
    per_family = {
        family: {
            "control_correct": 0,
            "total": 0,
            "treatment_correct": 0,
        }
        for family in FAMILIES
    }
    manifest: list[dict[str, object]] = []
    for seed, family in sorted(observed):
        path, report, digest = observed[(seed, family)]
        if (
            report.get("status")
            != "target_first_renderer_curriculum_smoke"
            or report.get("candidate_time_oracle_calls") != 0
            or report.get("candidate_time_search_calls") != 0
            or report.get("candidate_time_verifier_calls") != 0
        ):
            raise RendererCurriculumAuditError(
                "candidate custody counters differ"
            )
        equal_budget = report.get("equal_budget")
        parameters = report.get("parameter_receipt")
        if (
            not isinstance(equal_budget, dict)
            or equal_budget.get("base_rows") != 24
            or equal_budget.get("counterfactual_rows") != 24
            or equal_budget.get("initialization_identical") is not True
            or equal_budget.get("optimizer_updates_per_arm") != 300
            or not isinstance(parameters, dict)
            or parameters.get("learned_compiler") != 152_933
            or parameters.get("complete_system") != 125_234_597
        ):
            raise RendererCurriculumAuditError(
                "matched budget or parameter receipt differs"
            )
        development = report.get("development")
        if not isinstance(development, dict):
            raise RendererCurriculumAuditError("development report differs")
        treatment = development.get("renderer_curriculum_treatment")
        control = development.get("direction_shuffled_control")
        if not isinstance(treatment, dict) or not isinstance(control, dict):
            raise RendererCurriculumAuditError("qualification arms differ")
        if (
            treatment.get("correct", treatment.get("exact")) != 8
            or treatment.get("exact") != 8
            or treatment.get("total") != 8
            or treatment.get("invalid") != 0
            or control.get("total") != 8
            or control.get("invalid") != 0
        ):
            raise RendererCurriculumAuditError(
                "treatment/control aggregate differs"
            )
        cell_exact = treatment.get("cell_exact")
        if (
            not isinstance(cell_exact, dict)
            or any(
                not isinstance(cell_exact.get(cell), dict)
                or cell_exact[cell].get("correct") != 2
                or cell_exact[cell].get("total") != 2
                for cell in CELLS
            )
        ):
            raise RendererCurriculumAuditError(
                "per-cell treatment gate differs"
            )
        treatment_exact = int(treatment["exact"])
        control_exact = int(control["exact"])
        if treatment_exact <= control_exact:
            raise RendererCurriculumAuditError(
                "treatment direction is not positive"
            )
        positive_directions += 1
        treatment_correct += treatment_exact
        treatment_total += int(treatment["total"])
        control_correct += control_exact
        control_total += int(control["total"])
        per_family[family]["treatment_correct"] += treatment_exact
        per_family[family]["control_correct"] += control_exact
        per_family[family]["total"] += int(treatment["total"])
        preparation_calls += int(report["preparation_exact_parser_calls"])
        manifest.append(
            {
                "family": family,
                "file": path.name,
                "seed": seed,
                "sha256": digest,
            }
        )
    if treatment_correct != 120 or treatment_total != 120:
        raise RendererCurriculumAuditError("aggregate treatment gate differs")
    return {
        "candidate_time_oracle_calls": 0,
        "candidate_time_search_calls": 0,
        "candidate_time_verifier_calls": 0,
        "control": {
            "correct": control_correct,
            "rate": control_correct / control_total,
            "total": control_total,
        },
        "decision": (
            "renderer_curriculum_passes_five_seed_three_family_holdout_smoke"
        ),
        "manifest": manifest,
        "parameter_receipt": {
            "complete_system": 125_234_597,
            "learned_compiler": 152_933,
        },
        "per_family": per_family,
        "positive_seed_fold_directions": {
            "correct": positive_directions,
            "total": len(expected),
        },
        "preparation_exact_parser_calls": preparation_calls,
        "scope_boundary": (
            "bounded anonymous finite-machine compilation; target-first "
            "direction was covered under different symbols; not unrestricted "
            "natural-language or general reasoning"
        ),
        "treatment": {
            "correct": treatment_correct,
            "margin_over_control": (
                treatment_correct / treatment_total
                - control_correct / control_total
            ),
            "rate": treatment_correct / treatment_total,
            "total": treatment_total,
        },
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
