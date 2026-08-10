#!/usr/bin/env python3
"""Run the frozen DTC1 draft-transaction development falsifier once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Sequence

import torch

from draft_transaction_compiler import (
    DraftTransactionError,
    compile_draft_transactions,
    reset_state_reads,
)
from learned_arithmetic_microcode import LearnedArithmeticError, LearnedDigitMicrocode
from typed_microcode_graph import TypedMicrocodeGraphError, execute_learned


SCHEMA = "shohin-dtc1-development-evaluation-v1"
DIRECT_REPORT_SCHEMA = "shohin-nmc1-development-evaluation-v1"


class DTC1EvaluationError(ValueError):
    """Frozen DTC1 custody or evaluation differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise DTC1EvaluationError("development data SHA-256 differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 666 or len({row["identity_sha256"] for row in rows}) != 666:
        raise DTC1EvaluationError("development population differs")
    return rows


def source_shuffle(
    rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["register_depth"])].append(row)
    mapping = {}
    for depth, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: str(row["identity_sha256"]))
        if len(ordered) < 2:
            raise DTC1EvaluationError(f"source-shuffle singleton depth {depth}")
        for target, donor in zip(ordered, ordered[1:] + ordered[:1], strict=True):
            mapping[str(target["identity_sha256"])] = donor
    return mapping


def load_microcode(path: Path) -> LearnedDigitMicrocode:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-lam1-learned-arithmetic-microcode-v1":
        raise DTC1EvaluationError("LAM checkpoint differs")
    model = LearnedDigitMicrocode()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.transition_exact() != (1400, 1400):
        raise DTC1EvaluationError("LAM transition receipt differs")
    model.freeze_discrete()
    return model


def candidate_fraction(value) -> Fraction:
    numerator = int("".join(str(digit) for digit in reversed(value.numerator)))
    denominator = int("".join(str(digit) for digit in reversed(value.denominator)))
    result = Fraction(numerator, denominator)
    return -result if value.negative else result


def load_direct_report(
    path: Path, expected_sha256: str, expected_data_sha256: str
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise DTC1EvaluationError("direct report SHA-256 differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    details = report.get("details")
    if (
        report.get("schema") != DIRECT_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != "direct"
        or report.get("control") != "normal"
        or report.get("public_test_opened") is not False
        or report.get("development_data_sha256") != expected_data_sha256
        or report.get("checkpoint_sha256")
        != "8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53"
        or report.get("seed") != 2026081053
        or report.get("max_new_tokens") != 512
        or not isinstance(details, list)
        or len(details) != 666
    ):
        raise DTC1EvaluationError("direct report custody differs")
    by_identity = {str(detail["identity_sha256"]): detail for detail in details}
    if len(by_identity) != 666:
        raise DTC1EvaluationError("direct identities differ")
    return by_identity, report


def execute_candidate(microcode, graph, intervention: str = "normal") -> Fraction:
    return candidate_fraction(execute_learned(microcode, graph, intervention=intervention))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise DTC1EvaluationError("refusing existing output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluate_view(
    rows,
    drafts,
    donors,
    microcode,
    control: str,
) -> tuple[Counter[str], list[dict[str, object]]]:
    counts: Counter[str] = Counter()
    details: list[dict[str, object]] = []
    for target in rows:
        target_identity = str(target["identity_sha256"])
        donor = donors[target_identity]
        source_row = donor if control == "source_draft_shuffled" else target
        draft_identity = (
            str(donor["identity_sha256"])
            if control in {"draft_shuffled", "source_draft_shuffled"}
            else target_identity
        )
        source = str(source_row["original_question"])
        draft_detail = drafts[draft_identity]
        expected = Fraction(str(target["gold_answer"]))
        counts["rows"] += 1
        detail: dict[str, object] = {
            "identity_sha256": target_identity,
            "source_identity_sha256": str(source_row["identity_sha256"]),
            "draft_identity_sha256": draft_identity,
            "direct_correct": bool(drafts[target_identity]["answer_correct"]),
        }
        try:
            graph, receipt = compile_draft_transactions(
                source, str(draft_detail["completion"])
            )
        except DraftTransactionError as error:
            counts["compile_invalid"] += 1
            counts[f"compile_invalid:{error}"] += 1
            detail["compile_error"] = str(error)
            detail["correct"] = False
            details.append(detail)
            continue
        counts["compiled_rows"] += 1
        counts["annotations"] += receipt.annotations
        counts["accepted_transactions"] += receipt.accepted
        counts["rejected_transactions"] += len(receipt.rejected)
        counts["state_reads"] += receipt.state_reads
        counts["source_reads"] += receipt.source_reads
        counts["literal_reads"] += receipt.literal_reads
        counts["linked_rows"] += int(receipt.state_reads > 0)
        for reason in receipt.rejected:
            counts[f"rejected:{reason}"] += 1
        detail.update(
            {
                "annotations": receipt.annotations,
                "accepted_transactions": receipt.accepted,
                "rejected_reasons": list(receipt.rejected),
                "state_reads": receipt.state_reads,
                "source_reads": receipt.source_reads,
                "literal_reads": receipt.literal_reads,
            }
        )
        try:
            prediction = execute_candidate(microcode, graph)
        except (LearnedArithmeticError, TypedMicrocodeGraphError, ZeroDivisionError) as error:
            counts["execution_invalid"] += 1
            counts[f"execution_invalid:{type(error).__name__}"] += 1
            detail["execution_error"] = type(error).__name__
            detail["correct"] = False
            details.append(detail)
            continue
        correct = prediction == expected
        counts["executable_rows"] += 1
        counts["correct"] += int(correct)
        detail["prediction"] = str(prediction)
        detail["correct"] = correct
        if control == "normal":
            direct_correct = bool(draft_detail["answer_correct"])
            counts["direct_correct"] += int(direct_correct)
            counts["repair"] += int(correct and not direct_correct)
            counts["break"] += int(direct_correct and not correct)
            if receipt.state_reads:
                counts["linked_executable_rows"] += 1
                counts["linked_correct"] += int(correct)
                try:
                    reset_prediction = execute_candidate(
                        microcode, reset_state_reads(graph)
                    )
                    reset_correct = reset_prediction == expected
                except (
                    LearnedArithmeticError,
                    TypedMicrocodeGraphError,
                    ZeroDivisionError,
                ):
                    reset_correct = False
                    reset_prediction = None
                counts["state_reset_linked_correct"] += int(reset_correct)
                detail["state_reset_prediction"] = (
                    None if reset_prediction is None else str(reset_prediction)
                )
                detail["state_reset_correct"] = reset_correct
            try:
                opcode_prediction = execute_candidate(
                    microcode, graph, intervention="opcode_permuted"
                )
                opcode_correct = opcode_prediction == expected
            except (
                LearnedArithmeticError,
                TypedMicrocodeGraphError,
                ZeroDivisionError,
            ):
                opcode_correct = False
                opcode_prediction = None
            counts["opcode_permuted_correct"] += int(opcode_correct)
            detail["opcode_permuted_prediction"] = (
                None if opcode_prediction is None else str(opcode_prediction)
            )
            detail["opcode_permuted_correct"] = opcode_correct
        details.append(detail)
    return counts, details


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    rows = load_rows(args.data, args.expected_data_sha256)
    drafts, direct_report = load_direct_report(
        args.draft_report,
        args.expected_draft_report_sha256,
        args.expected_data_sha256,
    )
    donors = source_shuffle(rows)
    microcode = load_microcode(args.lam_checkpoint)
    views = {}
    all_details = {}
    for control in ("normal", "draft_shuffled", "source_draft_shuffled"):
        counts, details = evaluate_view(rows, drafts, donors, microcode, control)
        views[control] = dict(sorted(counts.items()))
        all_details[control] = details
    normal = views["normal"]
    linked_correct = int(normal.get("linked_correct", 0))
    reset_retained = int(normal.get("state_reset_linked_correct", 0))
    state_reset_loss = (
        1.0 - reset_retained / linked_correct if linked_correct else 0.0
    )
    normal_correct = int(normal.get("correct", 0))
    gates = {
        "compiled_and_executable_at_least_500": int(normal.get("executable_rows", 0))
        >= 500,
        "aligned_at_least_direct_267": normal_correct >= 267,
        "aligned_beats_draft_shuffle_by_100": normal_correct
        - int(views["draft_shuffled"].get("correct", 0))
        >= 100,
        "source_draft_shuffle_at_most_67": int(
            views["source_draft_shuffled"].get("correct", 0)
        )
        <= 67,
        "at_least_100_linked_rows": int(normal.get("linked_rows", 0)) >= 100,
        "state_reset_loss_at_least_20_points": state_reset_loss >= 0.20,
        "opcode_loss_at_least_30_points": (
            normal_correct - int(normal.get("opcode_permuted_correct", 0))
        )
        / 666
        >= 0.30,
        "zero_aligned_execution_invalid": int(normal.get("execution_invalid", 0))
        == 0,
    }
    interface_qualified = all(gates.values())
    capability_improved = interface_qualified and normal_correct >= 280
    report = {
        "schema": SCHEMA,
        "status": "capability_pass" if capability_improved else (
            "interface_pass" if interface_qualified else "fail"
        ),
        "holdout_used": False,
        "public_test_opened": False,
        "custody": {
            "development_data_sha256": args.expected_data_sha256,
            "draft_report_sha256": args.expected_draft_report_sha256,
            "direct_checkpoint_sha256": direct_report["checkpoint_sha256"],
            "direct_seed": direct_report["seed"],
            "direct_max_new_tokens": direct_report["max_new_tokens"],
            "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        },
        "views": views,
        "scores": {
            "aligned_correct": normal_correct,
            "draft_shuffled_correct": int(views["draft_shuffled"].get("correct", 0)),
            "source_draft_shuffled_correct": int(
                views["source_draft_shuffled"].get("correct", 0)
            ),
            "direct_reference_correct": 267,
            "repairs": int(normal.get("repair", 0)),
            "breaks": int(normal.get("break", 0)),
            "state_reset_loss_on_linked_correct": state_reset_loss,
            "opcode_permuted_correct": int(normal.get("opcode_permuted_correct", 0)),
        },
        "interface_gates": gates,
        "interface_qualified": interface_qualified,
        "capability_improved": capability_improved,
        "elapsed_seconds": time.time() - started,
        "details": all_details,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--draft-report", type=Path, required=True)
    parser.add_argument("--expected-draft-report-sha256", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
