#!/usr/bin/env python3
"""Score pinned LiveBench with its objective task-specific graders.

The current upstream top-level scorer imports API and agentic-coding stacks that
are irrelevant to the pinned 2024-11-25 release. This adapter dispatches only
the official graders represented on that board.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import base64
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any
import zlib

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK = "livebench"
REPORT_SCHEMA = "shohin-dense-public-livebench-score-v2"


def instruction_score(result: Any) -> float:
    flags = list(result.follow_instruction_list)
    if not flags:
        raise OfficialScoringError("LiveBench instruction list is empty")
    return (float(bool(result.follow_all_instructions)) + sum(map(bool, flags)) / len(flags)) / 2


def import_graders(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from livebench.if_runner.instruction_following_eval import evaluation_main
    from livebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from livebench.lcb_runner.utils.extraction_utils import extract_code
    from livebench.process_results.data_analysis.cta.utils import cta_process_results
    from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
    from livebench.process_results.data_analysis.tablereformat.utils import table_process_results
    from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results
    from livebench.process_results.math.math_competitions.utils import aime_process_results, mathcontest_process_results
    from livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
    from livebench.process_results.reasoning.spatial.utils import spatial_process_results
    from livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
    from livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
    from livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
    from livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
    from livebench.process_results.writing.typos.utils import typos_process_results
    return locals()


def lcb_generation_score(question: dict[str, Any], answer: str, graders: dict[str, Any]) -> float:
    extracted = graders["extract_code"](model_output=answer, lmstyle=None)
    partial = question.get("partial_solution")
    if partial and not extracted.startswith(partial):
        extracted = partial + "\n" + extracted
    public = json.loads(question["public_test_cases"])
    try:
        private = json.loads(question["private_test_cases"])
    except Exception:
        private = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(question["private_test_cases"])))
        )
    metadata = json.loads(question["original_json"]["metadata"])
    sample = {
        "input_output": json.dumps(
            {
                "inputs": [row["input"] for row in public + private],
                "outputs": [row["output"] for row in public + private],
                "fn_name": metadata.get("func_name"),
            }
        )
    }
    metrics, _, _ = graders["codegen_metrics"](
        [sample], [[extracted]], k_list=[1], num_process_evaluate=1, timeout=6
    )
    return float(metrics["pass@1"] == 1.0)


def legacy_if_scores(
    *,
    questions: list[dict[str, Any]],
    completions: dict[str, str],
    model_id: str,
    work_root: Path,
    evaluator: Any,
) -> dict[str, float]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        by_task.setdefault(str(question["task"]), []).append(question)
    scores: dict[str, float] = {}
    for task, task_questions in sorted(by_task.items()):
        model_answers = {
            model_id: {
                str(question["question_id"]): {
                    "question_id": question["question_id"],
                    "choices": [{"turns": [re.sub(
                        r"<think>.*?</think>", "",
                        completions[str(question["question_id"])], flags=re.DOTALL
                    ).strip()]}],
                }
                for question in task_questions
            }
        }
        result_root = work_root / model_id / task
        result_root.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(io.StringIO()):
            results = evaluator.evaluator(
                task_questions, model_answers, str(result_root), model_id
            )["strict"]
        if len(results) != len(task_questions):
            raise OfficialScoringError("LiveBench instruction result coverage differs")
        for result in results:
            scores[str(result.question_id)] = instruction_score(result)
    return scores


def ordinary_score(question: dict[str, Any], answer: str, graders: dict[str, Any]) -> float:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    task = str(question["task"])
    subtask = str(question.get("subtask") or task)
    text = str(question["turns"][0])
    truth = question.get("ground_truth")
    splits = subtask.split("_")
    if subtask == "cta":
        value = graders["cta_process_results"](truth, answer, False)
    elif subtask == "tablereformat":
        value = graders["table_process_results"](text, truth, answer, "v1", False)
    elif subtask == "tablejoin":
        value = graders["joinmap_process_results"](text, truth, answer, False)
    elif "amps_hard" in subtask or "amps_hard" in task.lower():
        value = graders["amps_hard_process_results"](truth, answer, False)
    elif splits[0] in {"amc", "smc"} or (len(splits) > 1 and splits[1] == "amc"):
        value = graders["mathcontest_process_results"](truth, answer, text, False)
    elif splits[0] == "aime":
        value = graders["aime_process_results"](truth, answer, False)
    elif splits[0] in {"imo", "usamo"}:
        value = graders["proof_rearrangement_process_results"](
            truth, answer, edit_distance=True, debug=False
        )
    elif subtask == "web_of_lies_v2":
        value = graders["web_of_lies_process_results"](truth, answer, False)
    elif "zebra_puzzle" in subtask:
        value = graders["get_zebra_puzzle_evaluator"](
            question["livebench_release_date"]
        )(truth, answer, False)
    elif subtask == "spatial":
        value = graders["spatial_process_results"](truth, answer, False)
    elif subtask == "typos":
        value = graders["typos_process_results"](truth, answer, False)
    elif subtask == "connections":
        value = graders["get_connections_puzzle_evaluator"](
            question["livebench_release_date"]
        )(truth, answer, False)
    elif subtask == "plot_unscrambling":
        value = graders["plot_unscrambling_process_results"](truth, answer, False)
    elif subtask in {"LCB_generation", "coding_completion"}:
        value = lcb_generation_score(question, answer, graders)
    else:
        raise OfficialScoringError(f"unsupported pinned LiveBench task: {subtask}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise OfficialScoringError("LiveBench score is outside [0, 1]")
    return value


def score(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SHOHIN_CODE_SANDBOX") != "1":
        raise OfficialScoringError(
            "LiveBench includes generated-code execution; use the frozen sandbox wrapper"
        )
    graders = import_graders(args.livebench_root)
    questions, assessors, ledgers = load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=BENCHMARK,
    )
    upstream = [row.get("assessor") for row in assessors]
    if not all(isinstance(row, dict) for row in upstream):
        raise OfficialScoringError("LiveBench assessor schema differs")
    hashes: dict[str, str] = {}
    sums: dict[str, float] = {}
    error_counts: dict[str, int] = {}
    for stage in STAGES:
        completion_by_qid = {
            str(row["question_id"]): generated["completion"]
            for row, generated in zip(upstream, ledgers[stage], strict=True)
        }
        if_questions = [
            row for row in upstream
            if row.get("category") == "instruction_following"
            and str(row.get("livebench_release_date", "")) < "2025-11-25"
        ]
        if_scores = legacy_if_scores(
            questions=if_questions,
            completions=completion_by_qid,
            model_id=f"shohin-{stage}",
            work_root=args.work_root,
            evaluator=graders["evaluation_main"],
        )
        if_ids = {str(row["question_id"]) for row in if_questions}
        rows = []
        errors = 0
        for question, assessor_row, source, generated in zip(
            questions, assessors, upstream, ledgers[stage], strict=True
        ):
            qid = str(source["question_id"])
            try:
                value = if_scores[qid] if qid in if_ids else ordinary_score(
                    source, generated["completion"], graders
                )
                scorer_error = None
            except Exception as exc:
                value = 0.0
                scorer_error = f"{type(exc).__name__}: {exc}"
                errors += 1
            rows.append(official_score_row(
                stage=stage,
                identity=question["id"],
                benchmark=BENCHMARK,
                metric="official_objective_score",
                stratum=assessor_row["stratum"],
                score=value,
                details={
                    "question_id": qid,
                    "task": source.get("task"),
                    "subtask": source.get("subtask"),
                    "scorer_error": scorer_error,
                },
            ))
        hashes[stage] = write_official_scores(
            output_root=args.output_root, benchmark=BENCHMARK, stage=stage, rows=rows
        )
        sums[stage] = sum(float(row["score"]) for row in rows)
        error_counts[stage] = errors
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmark": BENCHMARK,
        "rows": len(questions),
        "release": args.release,
        "livebench_commit": args.livebench_commit,
        "score_sums": sums,
        "scorer_error_counts": error_counts,
        "official_score_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--livebench-root", type=Path, required=True)
    parser.add_argument("--livebench-commit", required=True)
    parser.add_argument("--release", required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(score(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
