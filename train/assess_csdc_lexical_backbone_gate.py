#!/usr/bin/env python3
"""Assess the frozen Shohin-vs-SmolLM2 lexical transfer gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path, split: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    if value.get("split") != split or value.get("oracle") != "none":
        raise ValueError(f"unexpected evaluation contract: {path}")
    return value


def assess(
    shohin_compositional: dict[str, object],
    shohin_lexical: dict[str, object],
    smol_compositional: dict[str, object],
    smol_lexical: dict[str, object],
    shuffled_compositional: dict[str, object],
    shuffled_lexical: dict[str, object],
) -> dict[str, object]:
    del shuffled_compositional

    def program(result: dict[str, object]) -> float:
        return float(result["overall"]["semantic_program_exact"])

    def answer(result: dict[str, object]) -> float:
        return float(result["overall"]["answer_accuracy"])

    def all_four(result: dict[str, object]) -> int:
        return int(result["group_summary"]["all_four_semantic_program_exact"])

    metrics = {
        "shohin_compositional_program_exact": program(shohin_compositional),
        "shohin_lexical_program_exact": program(shohin_lexical),
        "shohin_lexical_answer": answer(shohin_lexical),
        "smollm2_compositional_program_exact": program(smol_compositional),
        "smollm2_lexical_program_exact": program(smol_lexical),
        "smollm2_lexical_answer": answer(smol_lexical),
        "smollm2_lexical_all_four": all_four(smol_lexical),
        "shuffled_lexical_program_exact": program(shuffled_lexical),
    }
    metrics["smollm2_minus_shohin_lexical_program_points"] = 100.0 * (
        metrics["smollm2_lexical_program_exact"]
        - metrics["shohin_lexical_program_exact"]
    )
    gates = {
        "shohin_compositional_program_at_least_98pct": (
            metrics["shohin_compositional_program_exact"] >= 0.98
        ),
        "smollm2_compositional_program_at_least_98pct": (
            metrics["smollm2_compositional_program_exact"] >= 0.98
        ),
        "smollm2_lexical_program_at_least_90pct": (
            metrics["smollm2_lexical_program_exact"] >= 0.90
        ),
        "smollm2_lexical_answer_at_least_95pct": (
            metrics["smollm2_lexical_answer"] >= 0.95
        ),
        "smollm2_lexical_program_advantage_at_least_15_points": (
            metrics["smollm2_minus_shohin_lexical_program_points"] >= 15.0
        ),
        "smollm2_lexical_all_four_at_least_410": (
            metrics["smollm2_lexical_all_four"] >= 410
        ),
        "shuffled_lexical_program_at_most_5pct": (
            metrics["shuffled_lexical_program_exact"] <= 0.05
        ),
    }
    passed = all(gates.values())
    return {
        "schema": "csdc_lexical_backbone_transfer_assessment_v1",
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": passed,
        "decision": (
            "advance_role_copy_predictions_to_frozen_csdc"
            if passed
            else "close_pretrained_residual_only_and_add_explicit_lexical_grounding"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shohin-dir", required=True)
    parser.add_argument("--smollm2-dir", required=True)
    parser.add_argument("--shuffled-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing assessment")

    def pair(directory: str) -> tuple[dict[str, object], dict[str, object]]:
        root = Path(directory)
        return (
            load_result(root / "development_compositional.json", "development_compositional"),
            load_result(root / "development_lexical_ood.json", "development_lexical_ood"),
        )

    result = assess(*pair(args.shohin_dir), *pair(args.smollm2_dir), *pair(args.shuffled_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
