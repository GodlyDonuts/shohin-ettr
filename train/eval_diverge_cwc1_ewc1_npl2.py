#!/usr/bin/env python3
"""Compose qualified CWC1 selection, frozen EWC structure, and confirmed NPL2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

import eval_diverge_npl2_development as npl2
from diverge_cwc1_npl2_data import validate_wrapper_record
from diverge_npl2_runtime import TypedEpisode
from eval_diverge_cwc1 import (
    _load_arm as load_cwc_arm,
    _predict as predict_cwc,
    _scrubbed,
    sha256_path,
)
from eval_diverge_ewc1 import _predict as predict_ewc
from eval_diverge_ewc1_npl2 import build_typed_cache, _load_world_model


DEVELOPMENT_SCHEMA = "shohin-diverge-cwc1-ewc1-npl2-development-v1"
CONFIRMATION_SCHEMA = "shohin-diverge-cwc1-ewc1-npl2-confirmation-seed-v1"


def _option(arguments: Sequence[str], name: str) -> str:
    try:
        index = arguments.index(name)
        return arguments[index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit(f"CWC1/EWC1/NPL2 missing base option {name}") from error


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("CWC1/EWC1/NPL2 data hash differs")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _align_wrapper(
    public: Sequence[Mapping[str, Any]], wrappers: Sequence[Mapping[str, Any]]
) -> None:
    cursor = 0
    for episode in public:
        for phase in ("acquisition", "transfer"):
            for phase_index, program in enumerate(episode[phase]):
                if cursor >= len(wrappers):
                    raise SystemExit("CWC1 wrapper is shorter than NPL2 WORLD")
                row = wrappers[cursor]
                validate_wrapper_record(row)
                expected = (
                    str(episode["episode_id"]),
                    phase,
                    phase_index,
                    str(program["program_id"]),
                    str(program["source_sha256"]),
                )
                observed = (
                    str(row["episode_id"]),
                    str(row["phase"]),
                    int(row["phase_index"]),
                    str(row["program_id"]),
                    str(row["program_source_sha256"]),
                )
                if observed != expected:
                    raise SystemExit("CWC1 wrapper does not align with NPL2 WORLD")
                cursor += 1
    if cursor != len(wrappers):
        raise SystemExit("CWC1 wrapper is longer than NPL2 WORLD")


def _world_work(
    wrappers: Sequence[Mapping[str, Any]], positions: Sequence[int]
) -> list[dict[str, Any]]:
    work = []
    for row, position in zip(wrappers, positions, strict=True):
        program = row["candidate_programs"][int(position)]
        source = str(program["source_text"])
        work.append(
            {
                "source_text": source,
                "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
                "aliases": [str(value) for value in row["aliases"]],
                "registers": [str(value) for value in row["registers"]],
            }
        )
    return work


def _structure_score(
    wrappers: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    predictions: Sequence[tuple[tuple[int, int], tuple[int, ...]]],
    *,
    compare_to_true: bool,
) -> dict[str, Any]:
    initial_exact = 0
    operation_exact = 0
    joint_exact = 0
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row, position, prediction in zip(
        wrappers, positions, predictions, strict=True
    ):
        target_position = int(row["target_position"])
        gold_position = target_position if compare_to_true else int(position)
        gold = row["candidate_programs"][gold_position]
        initial_ok = tuple(int(value) for value in gold["initial_state"]) == prediction[0]
        operation_ok = tuple(int(value) for value in gold["symbols"]) == prediction[1]
        joint_ok = initial_ok and operation_ok
        initial_exact += initial_ok
        operation_exact += operation_ok
        joint_exact += joint_ok
        key = f"{row['renderer'][0]}:{row['renderer'][1]}"
        by_renderer[key]["rows"] += 1
        by_renderer[key]["joint_exact"] += joint_ok
    rows = len(wrappers)
    return {
        "rows": rows,
        "initial_exact": initial_exact,
        "initial_rate": initial_exact / rows,
        "operation_exact": operation_exact,
        "operation_rate": operation_exact / rows,
        "joint_exact": joint_exact,
        "joint_rate": joint_exact / rows,
        "renderer_minimum_joint_rate": min(
            value["joint_exact"] / value["rows"] for value in by_renderer.values()
        ),
    }


@torch.no_grad()
def _compile_world(
    *,
    wrappers: Sequence[Mapping[str, Any]],
    public: Sequence[Mapping[str, Any]],
    cwc_model: torch.nn.Module,
    ewc_model: torch.nn.Module,
    device: torch.device,
    cwc_batch_size: int,
    ewc_batch_size: int,
) -> tuple[dict[str, TypedEpisode], dict[str, Any]]:
    normal_scores, partner_scores, projection_residual = predict_cwc(
        cwc_model, wrappers, arm="involution", device=device, batch_size=cwc_batch_size
    )
    targets = torch.tensor(
        [int(row["target_position"]) for row in wrappers], dtype=torch.long
    )
    normal_predictions = normal_scores.argmax(-1)
    partner_predictions = partner_scores.argmax(-1)
    selected_positions = [int(value) for value in normal_predictions.tolist()]
    opposite_positions = [1 - value for value in selected_positions]
    scrubbed = [_scrubbed(row) for row in wrappers]
    scrubbed_scores, _, scrubbed_residual = predict_cwc(
        cwc_model, scrubbed, arm="involution", device=device, batch_size=cwc_batch_size
    )
    scrubbed_predictions = scrubbed_scores.argmax(-1)
    target_margin = scrubbed_scores.gather(1, targets[:, None]).squeeze(1) - (
        scrubbed_scores.gather(1, (1 - targets)[:, None]).squeeze(1)
    )

    selected_work = _world_work(wrappers, selected_positions)
    opposite_work = _world_work(wrappers, opposite_positions)
    selected_predictions = predict_ewc(
        ewc_model, selected_work, device=device, batch_size=ewc_batch_size
    )
    opposite_predictions = predict_ewc(
        ewc_model, opposite_work, device=device, batch_size=ewc_batch_size
    )
    typed_cache = build_typed_cache(public, selected_predictions)
    selector_exact = int((normal_predictions == targets).sum())
    partner_mapped_exact = int((partner_predictions == (1 - targets)).sum())
    partner_original_exact = int((partner_predictions == targets).sum())
    scrubbed_exact = int((scrubbed_predictions == targets).sum())
    rows = len(wrappers)
    selector = {
        "rows": rows,
        "normal_exact": selector_exact,
        "normal_rate": selector_exact / rows,
        "counterfactual_mapped_exact": partner_mapped_exact,
        "counterfactual_mapped_rate": partner_mapped_exact / rows,
        "counterfactual_against_original_exact": partner_original_exact,
        "counterfactual_against_original_rate": partner_original_exact / rows,
        "directive_scrub_exact": scrubbed_exact,
        "directive_scrub_rate": scrubbed_exact / rows,
        "directive_scrub_mean_signed_margin": float(target_margin.mean()),
        "directive_scrub_max_absolute_margin": float(target_margin.abs().max()),
        "projection_max_absolute_error": max(projection_residual, scrubbed_residual),
    }
    structure = {
        "selected_against_true": _structure_score(
            wrappers, selected_positions, selected_predictions, compare_to_true=True
        ),
        "selected_candidate_parse": _structure_score(
            wrappers, selected_positions, selected_predictions, compare_to_true=False
        ),
        "forced_opposite_against_true": _structure_score(
            wrappers, opposite_positions, opposite_predictions, compare_to_true=True
        ),
        "forced_opposite_candidate_parse": _structure_score(
            wrappers, opposite_positions, opposite_predictions, compare_to_true=False
        ),
    }
    metrics = {
        "selector": selector,
        "structure": structure,
        "gate": {
            "selector_at_least_99_percent": selector["normal_rate"] >= 0.99,
            "counterfactual_mapped_at_least_99_percent": selector[
                "counterfactual_mapped_rate"
            ]
            >= 0.99,
            "counterfactual_changes_physical_world": selector[
                "counterfactual_against_original_rate"
            ]
            <= 0.01,
            "directive_scrub_is_chance": 0.49 <= selector["directive_scrub_rate"] <= 0.51,
            "directive_scrub_has_zero_margin": selector[
                "directive_scrub_max_absolute_margin"
            ]
            <= 1e-6,
            "projection_is_exact": selector["projection_max_absolute_error"] == 0.0,
            "selected_structure_at_least_99_percent": structure[
                "selected_against_true"
            ]["joint_rate"]
            >= 0.99,
            "forced_opposite_is_not_true": structure["forced_opposite_against_true"][
                "joint_rate"
            ]
            <= 0.01,
            "ewc_transcribes_decoy_at_least_99_percent": structure[
                "forced_opposite_candidate_parse"
            ]["joint_rate"]
            >= 0.99,
        },
    }
    metrics["gate"]["passed"] = all(metrics["gate"].values())
    return typed_cache, metrics


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cwc-root", type=Path, required=True)
    parser.add_argument("--cwc-checkpoint-sha256", required=True)
    parser.add_argument("--cwc-training-report-sha256", required=True)
    parser.add_argument("--cwc-result", type=Path, required=True)
    parser.add_argument("--cwc-result-sha256", required=True)
    parser.add_argument("--cwc-wrapper-data", type=Path, required=True)
    parser.add_argument("--cwc-wrapper-data-sha256", required=True)
    parser.add_argument("--ewc-checkpoint", type=Path, required=True)
    parser.add_argument("--ewc-checkpoint-sha256", required=True)
    parser.add_argument("--ewc-result", type=Path, required=True)
    parser.add_argument("--ewc-result-sha256", required=True)
    parser.add_argument("--cwc-batch-size", type=int, default=256)
    parser.add_argument("--ewc-batch-size", type=int, default=512)
    args, remaining = parser.parse_known_args()
    if not torch.cuda.is_available():
        raise SystemExit("CWC1/EWC1/NPL2 integration requires CUDA")
    if sha256_path(args.cwc_result) != args.cwc_result_sha256:
        raise SystemExit("CWC1 qualification result hash differs")
    cwc_result = json.loads(args.cwc_result.read_text(encoding="utf-8"))
    if cwc_result.get("split") != "confirmation" or not cwc_result.get("all_pass"):
        raise SystemExit("CWC1/EWC1/NPL2 requires confirmed CWC1")
    if sha256_path(args.ewc_result) != args.ewc_result_sha256:
        raise SystemExit("EWC1 structural result hash differs")
    ewc_result = json.loads(args.ewc_result.read_text(encoding="utf-8"))
    if (
        ewc_result.get("split") != "development"
        or ewc_result.get("all_pass") is not False
        or not ewc_result.get("gates", {}).get("normal_joint")
        or ewc_result.get("gates", {}).get("source_scrub_collapse") is not False
    ):
        raise SystemExit("EWC1 structural-baseline boundary differs")

    public_path = Path(_option(remaining, "--public-data"))
    public_sha256 = _option(remaining, "--public-data-sha256")
    public = _load_jsonl(public_path, public_sha256)
    wrappers = _load_jsonl(args.cwc_wrapper_data, args.cwc_wrapper_data_sha256)
    _align_wrapper(public, wrappers)
    device = torch.device("cuda")
    cwc, _, checkpoint_sha256, report_sha256 = load_cwc_arm(
        args.cwc_root, "involution", device
    )
    if (
        checkpoint_sha256 != args.cwc_checkpoint_sha256
        or report_sha256 != args.cwc_training_report_sha256
    ):
        raise SystemExit("CWC1 trained-arm receipt differs")
    ewc, ewc_checkpoint = _load_world_model(
        args.ewc_checkpoint, args.ewc_checkpoint_sha256, device
    )
    typed_cache, compilation = _compile_world(
        wrappers=wrappers,
        public=public,
        cwc_model=cwc,
        ewc_model=ewc,
        device=device,
        cwc_batch_size=args.cwc_batch_size,
        ewc_batch_size=args.ewc_batch_size,
    )
    if not compilation["gate"]["passed"]:
        raise SystemExit("CWC1/EWC1 WORLD composition gate failed")
    cwc_state_sha256 = str(
        torch.load(
            args.cwc_root / "checkpoint_0001000.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_sha256"]
    )
    ewc_state_sha256 = str(ewc_checkpoint["model_state_sha256"])
    del cwc, ewc, wrappers, public
    torch.cuda.empty_cache()

    def learned_world(candidate: Mapping[str, Any]) -> TypedEpisode:
        try:
            return typed_cache[str(candidate["episode_id"])]
        except KeyError as error:
            raise RuntimeError("CWC1/EWC1/NPL2 WORLD cache miss") from error

    npl2.typed_episode_from_public = learned_world
    npl2.WORLD_OWNER_RECEIPT = (
        f"cwc1:{cwc_state_sha256}:ewc1-structural:{ewc_state_sha256}"
    )
    npl2.WORLD_OWNER_CUSTODY = {
        "selector": "confirmed-cwc1-involution",
        "selector_checkpoint": str(args.cwc_root / "checkpoint_0001000.pt"),
        "selector_checkpoint_sha256": args.cwc_checkpoint_sha256,
        "selector_training_report_sha256": args.cwc_training_report_sha256,
        "selector_qualification_result": str(args.cwc_result),
        "selector_qualification_result_sha256": args.cwc_result_sha256,
        "structural_extractor": "closed-ewc1-structural-baseline",
        "structural_checkpoint": str(args.ewc_checkpoint),
        "structural_checkpoint_sha256": args.ewc_checkpoint_sha256,
        "structural_result": str(args.ewc_result),
        "structural_result_sha256": args.ewc_result_sha256,
        "wrapper_data": str(args.cwc_wrapper_data),
        "wrapper_data_sha256": args.cwc_wrapper_data_sha256,
        "compilation": compilation,
        "complete_candidate_commit_before_structure": True,
        "fieldwise_candidate_averaging": False,
        "source_deleted_after_compilation": True,
        "ewc_not_qualified_as_semantic_owner": True,
    }
    npl2.DEVELOPMENT_SCHEMA = DEVELOPMENT_SCHEMA
    npl2.CONFIRMATION_SEED_SCHEMA = CONFIRMATION_SCHEMA
    sys.argv = [sys.argv[0], *remaining]
    npl2.main()


if __name__ == "__main__":
    main()
