#!/usr/bin/env python3
"""Verify a synthesized Q36 answer against two independently derived alternatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from hf_q36_mtr_generate_drafts import (
    _atomic_json,
    _atomic_lines,
    load_sources,
    sha256_file,
)
from hf_q36_mtr_synthesize_trajectories import validate_aligned_metadata
from q36_mtr_roles import MODEL_REVISION

SCHEMA = "shohin-q36-mtr-model-draft-v1"
REPORT_SCHEMA = "shohin-q36-mtr-hierarchical-synthesis-shard-v1"
HIERARCHY_SCHEMA = "shohin-q36-mtr-hierarchical-synthesis-v1"
SHARDS = 16
ROWS = 1_289
SEED = 2026081423
INCUMBENT_CHALLENGER_SEED = 2026081424
INCUMBENT_CYCLIC_SEED = 2026081425
INCUMBENT_INTERPOLATION_SEED = 2026081426
MULTI_TRAJECTORY_ADJUDICATION_SEED = 2026081427
GUIDED_MULTI_TRAJECTORY_ADJUDICATION_SEED = 2026081428
MAX_NEW_TOKENS = 768
TASKS = {"bbh_logic", "math500", "mbpp"}
ADJUDICATION_ARMS = (
    "hierarchy",
    "interpolation",
    "direct",
    "offset_one",
    "level_two",
    "challenger",
)


class Q36MTRHierarchicalSynthesisError(RuntimeError):
    """Hierarchical synthesis inputs, generation, or custody differ."""


def hierarchical_prompt(
    source_prompt: str,
    synthesis: str,
    stacked: str,
    self_refinement: str,
) -> str:
    values = (source_prompt, synthesis, stacked, self_refinement)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise Q36MTRHierarchicalSynthesisError("hierarchical prompt input differs")
    return (
        "Produce the single most reliable answer to the original problem. Candidate A "
        "is the current integrated solution produced by reconciling three independent "
        "reasoning trajectories. Preserve Candidate A unless checking it against the "
        "original problem reveals a concrete error. Candidates B and C are independent "
        "alternatives: use them as evidence to locate and repair such an error, not as "
        "votes. Recompute disputed steps yourself. Do not mention the candidates or this "
        "review process, and return only one final solution in the original problem's "
        "requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        f"Candidate A — integrated solution:\n{synthesis}\n\n"
        f"Candidate B — preserved alternative:\n{stacked}\n\n"
        f"Candidate C — independent refinement:\n{self_refinement}\n\n"
        "Return the verified final solution in the original problem's requested output "
        "format."
    )


def incumbent_challenger_prompt(
    source_prompt: str,
    incumbent: str,
    challenger: str,
    direct_synthesis: str,
) -> str:
    values = (source_prompt, incumbent, challenger, direct_synthesis)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise Q36MTRHierarchicalSynthesisError(
            "incumbent-challenger prompt input differs"
        )
    return (
        "Produce the single most reliable answer to the original problem. Candidate A "
        "is the incumbent verified solution and should be preserved unless a concrete, "
        "recomputed error is established. Candidate B is a deeper synthesis obtained "
        "from multiple cyclic reconciliations, and Candidate C is the original direct "
        "three-trajectory synthesis. Use B and C only to identify a specific weakness "
        "in A; never change A merely because alternatives agree. If a weakness is "
        "found, recompute the disputed reasoning from the original problem and repair "
        "only what is necessary. Do not mention the candidates or this review process, "
        "and return one final solution in the original problem's requested output "
        "format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        f"Candidate A — incumbent verified solution:\n{incumbent}\n\n"
        f"Candidate B — cyclic deep synthesis:\n{challenger}\n\n"
        f"Candidate C — direct synthesis:\n{direct_synthesis}\n\n"
        "Return the verified final solution in the original problem's requested output "
        "format."
    )


def incumbent_cyclic_prompt(
    source_prompt: str,
    incumbent: str,
    cyclic_offset_one: str,
    cyclic_offset_two: str,
) -> str:
    values = (source_prompt, incumbent, cyclic_offset_one, cyclic_offset_two)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise Q36MTRHierarchicalSynthesisError("incumbent-cyclic prompt input differs")
    return (
        "Produce the single most reliable answer to the original problem. Candidate A "
        "is the incumbent verified solution and should be preserved unless a concrete, "
        "recomputed error is established. Candidates B and C are independent cyclic "
        "reconciliations of the underlying reasoning trajectories. Use either cyclic "
        "candidate to identify a specific weakness in A, but never change A because of "
        "surface agreement or voting. Recompute every disputed step from the original "
        "problem; if A is wrong, repair only what is necessary. Do not mention the "
        "candidates or this review process, and return one final solution in the "
        "original problem's requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        f"Candidate A — incumbent verified solution:\n{incumbent}\n\n"
        f"Candidate B — cyclic reconciliation one:\n{cyclic_offset_one}\n\n"
        f"Candidate C — cyclic reconciliation two:\n{cyclic_offset_two}\n\n"
        "Return the verified final solution in the original problem's requested output "
        "format."
    )


def incumbent_interpolation_prompt(
    source_prompt: str,
    incumbent: str,
    interpolated_synthesis: str,
    direct_synthesis: str,
) -> str:
    values = (source_prompt, incumbent, interpolated_synthesis, direct_synthesis)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise Q36MTRHierarchicalSynthesisError(
            "incumbent-interpolation prompt input differs"
        )
    return (
        "Produce the single most reliable answer to the original problem. Candidate A "
        "is the incumbent verified solution and should be preserved unless a concrete, "
        "recomputed error is established. Candidate B is an independently synthesized "
        "solution from a conservatively interpolated reviser, and Candidate C is the "
        "original direct synthesis. Use B and C only as evidence for a specific error "
        "in A; do not vote or replace A for stylistic differences. Recompute the "
        "disputed reasoning from the original problem and repair only what is necessary. "
        "Do not mention the candidates or this review process, and return one final "
        "solution in the original problem's requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        f"Candidate A — incumbent verified solution:\n{incumbent}\n\n"
        f"Candidate B — interpolated-reviser synthesis:\n{interpolated_synthesis}\n\n"
        f"Candidate C — direct synthesis:\n{direct_synthesis}\n\n"
        "Return the verified final solution in the original problem's requested output "
        "format."
    )


def normalized_candidate_answer(task: str, completion: str) -> str | None:
    from hf_product_reasoning_eval import (
        _normalize_math,
        _normalize_short_answer,
        extract_boxed,
        extract_short_answer,
    )

    if task == "bbh_logic":
        return _normalize_short_answer(extract_short_answer(completion))
    if task == "math500":
        return _normalize_math(extract_boxed(completion))
    if task == "mbpp":
        return None
    raise Q36MTRHierarchicalSynthesisError("adjudication task differs")


def multi_trajectory_adjudication_plan(
    source_prompt: str, candidates: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None, dict[str, Any]]:
    if tuple(candidates) != ADJUDICATION_ARMS:
        raise Q36MTRHierarchicalSynthesisError("adjudication arm order differs")
    if not isinstance(source_prompt, str) or not source_prompt.strip():
        raise Q36MTRHierarchicalSynthesisError("adjudication source differs")
    identities = {row.get("identity_sha256") for row in candidates.values()}
    tasks = {row.get("task") for row in candidates.values()}
    if len(identities) != 1 or len(tasks) != 1:
        raise Q36MTRHierarchicalSynthesisError("adjudication identity differs")
    task = next(iter(tasks))
    if task == "mbpp":
        return (
            "interpolation",
            None,
            {
                "decision": "preserve_executable_control",
                "unique_answers": None,
                "maximum_support": None,
            },
        )
    grouped: dict[str, list[str]] = {}
    unparsed = []
    for arm, row in candidates.items():
        completion = row.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise Q36MTRHierarchicalSynthesisError("adjudication completion differs")
        answer = normalized_candidate_answer(task, completion)
        if answer is None:
            unparsed.append(arm)
        else:
            grouped.setdefault(answer, []).append(arm)
    if len(grouped) == 1 and not unparsed:
        return (
            "hierarchy",
            None,
            {
                "decision": "preserve_unanimous_answer",
                "unique_answers": 1,
                "maximum_support": 6,
            },
        )
    proposals = []
    for index, (answer, arms) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (
                -len(item[1]),
                min(ADJUDICATION_ARMS.index(arm) for arm in item[1]),
            ),
        ),
        start=1,
    ):
        representative = arms[0]
        proposals.append(
            f"Proposal {index} — supported by {len(arms)} of 6 independent trajectories "
            f"({', '.join(arms)}):\n{candidates[representative]['completion']}"
        )
    for arm in unparsed:
        proposals.append(
            f"Proposal {len(proposals) + 1} — unparsed independent trajectory "
            f"({arm}):\n{candidates[arm]['completion']}"
        )
    prompt = (
        "Resolve a disagreement among independently derived solutions to the original "
        "problem. Support counts are evidence, not proof. Recompute the decisive steps "
        "from the original problem, preserve the plurality answer unless you establish "
        "a concrete error, and repair only that error. Do not mention the proposals, "
        "support counts, or review process. Return one final solution in the original "
        "problem's requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        + "\n\n".join(proposals)
        + "\n\nReturn the independently verified final solution."
    )
    return (
        None,
        prompt,
        {
            "decision": "model_owned_disagreement_adjudication",
            "unique_answers": len(grouped) + len(unparsed),
            "maximum_support": max((len(arms) for arms in grouped.values()), default=0),
        },
    )


def guided_multi_trajectory_adjudication_plan(
    source_prompt: str,
    candidates: dict[str, dict[str, Any]],
    guidance: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any]]:
    if tuple(candidates) != ADJUDICATION_ARMS:
        raise Q36MTRHierarchicalSynthesisError("guided adjudication arm order differs")
    identities = {row.get("identity_sha256") for row in candidates.values()}
    tasks = {row.get("task") for row in candidates.values()}
    if len(identities) != 1 or guidance.get("identity_sha256") not in identities:
        raise Q36MTRHierarchicalSynthesisError("guided adjudication identity differs")
    metadata = guidance.get("nested_pattern_consensus")
    if (
        guidance.get("schema") != SCHEMA
        or not isinstance(metadata, dict)
        or metadata.get("schema") != "shohin-q36-mtr-nested-pattern-consensus-v1"
        or metadata.get("heldout_identity_labels_read") != 0
        or metadata.get("selected") not in ADJUDICATION_ARMS
        or isinstance(metadata.get("estimated_reliability"), bool)
        or not isinstance(metadata.get("estimated_reliability"), (int, float))
        or not math.isfinite(metadata["estimated_reliability"])
        or not isinstance(guidance.get("completion"), str)
        or not guidance["completion"].strip()
        or not isinstance(source_prompt, str)
        or not source_prompt.strip()
    ):
        raise Q36MTRHierarchicalSynthesisError("guided adjudication prior differs")
    task = guidance.get("task")
    if tasks != {task}:
        raise Q36MTRHierarchicalSynthesisError("guided adjudication task differs")
    if task == "mbpp":
        return (
            "interpolation",
            None,
            {
                "decision": "preserve_executable_control",
                "unique_answers": None,
                "maximum_support": None,
                "guidance_selected": metadata["selected"],
            },
        )
    if task not in {"bbh_logic", "math500"}:
        raise Q36MTRHierarchicalSynthesisError("guided adjudication task differs")
    grouped: dict[tuple[str, str], list[str]] = {}
    for arm, row in candidates.items():
        answer = normalized_candidate_answer(task, row["completion"])
        key = (
            ("normalized", answer)
            if answer is not None
            else ("raw", row["completion"].strip())
        )
        grouped.setdefault(key, []).append(arm)
    if len(grouped) == 1:
        return (
            "hierarchy",
            None,
            {
                "decision": "preserve_unanimous_answer",
                "unique_answers": 1,
                "maximum_support": 6,
                "guidance_selected": metadata["selected"],
            },
        )
    normalized_guidance = normalized_candidate_answer(task, guidance["completion"])
    guided_answer = (
        ("normalized", normalized_guidance)
        if normalized_guidance is not None
        else ("raw", guidance["completion"].strip())
    )
    guided_arm = metadata["selected"]
    if guided_arm not in grouped.get(guided_answer, []):
        raise Q36MTRHierarchicalSynthesisError("guided adjudication answer differs")
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            item[0] != guided_answer,
            -len(item[1]),
            min(ADJUDICATION_ARMS.index(arm) for arm in item[1]),
        ),
    )
    proposals = []
    for index, (answer_key, arms) in enumerate(ordered, start=1):
        representative = guided_arm if answer_key == guided_answer else arms[0]
        label = (
            "cross-fitted incumbent" if answer_key == guided_answer else "alternative"
        )
        proposals.append(
            f"Proposal {index} — {label}, supported by {len(arms)} of 6 independent "
            f"trajectories ({', '.join(arms)}):\n"
            f"{candidates[representative]['completion']}"
        )
    prompt = (
        "Independently solve the original problem, then adjudicate the proposed "
        "solutions. Proposal 1 is an incumbent selected by a reliability model trained "
        "without this identity or its shard; that is a useful prior, not proof. Verify "
        "the decisive reasoning yourself. Preserve Proposal 1 unless recomputation "
        "establishes a concrete error, and then repair the error using the strongest "
        "alternative evidence. Support counts are evidence, not truth. Do not mention "
        "the proposals, reliability prior, support counts, or review process. Return "
        "one final solution in the original problem's requested output format.\n\n"
        f"Original problem:\n{source_prompt}\n\n"
        + "\n\n".join(proposals)
        + "\n\nReturn the independently verified final solution."
    )
    return (
        None,
        prompt,
        {
            "decision": "model_owned_guided_disagreement_adjudication",
            "unique_answers": len(grouped),
            "maximum_support": max((len(arms) for arms in grouped.values()), default=0),
            "guidance_selected": guided_arm,
            "guidance_estimated_reliability": metadata["estimated_reliability"],
            "guidance_heldout_identity_labels_read": 0,
        },
    )


def mode_contract(mode: str) -> dict[str, Any]:
    if mode == "retention_controls":
        return {
            "seed": SEED,
            "path_counts": (16, 1, 8),
            "roles": (
                "integrated_synthesis",
                "stacked_preserved",
                "self_refinement",
            ),
            "interpretation": "hierarchical_synthesis_with_conservative_retention",
        }
    if mode == "incumbent_challenger":
        return {
            "seed": INCUMBENT_CHALLENGER_SEED,
            "path_counts": (16, 16, 16),
            "roles": (
                "incumbent_verified",
                "cyclic_deep_synthesis",
                "direct_synthesis",
            ),
            "interpretation": "incumbent_challenger_conservative_verification",
        }
    if mode == "incumbent_cyclic":
        return {
            "seed": INCUMBENT_CYCLIC_SEED,
            "path_counts": (16, 16, 16),
            "roles": (
                "incumbent_verified",
                "cyclic_offset_one",
                "cyclic_offset_two",
            ),
            "interpretation": "incumbent_cyclic_conservative_verification",
        }
    if mode == "incumbent_interpolation":
        return {
            "seed": INCUMBENT_INTERPOLATION_SEED,
            "path_counts": (16, 16, 16),
            "roles": (
                "incumbent_verified",
                "interpolated_synthesis",
                "direct_synthesis",
            ),
            "interpretation": "incumbent_interpolation_conservative_verification",
        }
    if mode == "multi_trajectory_adjudication":
        return {
            "seed": MULTI_TRAJECTORY_ADJUDICATION_SEED,
            "path_counts": (16, 16, 16, 16, 16, 16),
            "roles": (
                "hierarchy",
                "interpolation",
                "direct",
                "offset_one",
                "level_two",
                "challenger",
            ),
            "interpretation": "selective_model_owned_multi_trajectory_adjudication",
        }
    if mode == "guided_multi_trajectory_adjudication":
        return {
            "seed": GUIDED_MULTI_TRAJECTORY_ADJUDICATION_SEED,
            "path_counts": (16, 16, 16, 16, 16, 16),
            "roles": ADJUDICATION_ARMS,
            "interpretation": "crossfit_guided_model_owned_trajectory_adjudication",
        }
    raise Q36MTRHierarchicalSynthesisError("hierarchical mode differs")


def load_candidate_group(
    paths: list[Path], *, expected_paths: int
) -> dict[str, dict[str, Any]]:
    if len(paths) != expected_paths:
        raise Q36MTRHierarchicalSynthesisError("candidate path geometry differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise Q36MTRHierarchicalSynthesisError("candidate path differs")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("identity_sha256")
                if row.get("split", "development") != "development":
                    continue
                if (
                    row.get("schema") not in {SCHEMA, "shohin-q36-mtr-candidate-v1"}
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in result
                    or row.get("task") not in TASKS
                    or not isinstance(row.get("completion"), str)
                    or not row["completion"].strip()
                    or isinstance(row.get("generated_tokens"), bool)
                    or not isinstance(row.get("generated_tokens"), int)
                    or row["generated_tokens"] <= 0
                    or not isinstance(row.get("max_token_exhausted"), bool)
                ):
                    raise Q36MTRHierarchicalSynthesisError("candidate payload differs")
                result[identity] = row
    if len(result) != ROWS:
        raise Q36MTRHierarchicalSynthesisError("candidate coverage differs")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import (
        GENERATED_ONLY_SEQUENCE_CONTRACT,
        _generate_completions,
        _generation_stop_token_ids,
        _render_prompt,
    )
    from hf_q36_mtr_evaluate import (
        load_q36_adapter_model,
        q36_nonpadding_prompt_tokens,
    )

    contract = mode_contract(args.mode)
    additional_groups = (
        args.offset_one_candidates,
        args.level_two_candidates,
        args.challenger_candidates,
    )
    adjudication_modes = {
        "multi_trajectory_adjudication",
        "guided_multi_trajectory_adjudication",
    }
    if args.mode in adjudication_modes:
        if any(group is None for group in additional_groups):
            raise Q36MTRHierarchicalSynthesisError(
                "adjudication candidate groups are missing"
            )
    elif any(group is not None for group in additional_groups):
        raise Q36MTRHierarchicalSynthesisError(
            "unexpected adjudication candidate groups"
        )
    if (
        args.model_revision != MODEL_REVISION
        or args.seed != contract["seed"]
        or args.shard_count != SHARDS
        or args.max_new_tokens != MAX_NEW_TOKENS
        or args.batch_size != 2
        or not 0 <= args.shard_index < SHARDS
    ):
        raise Q36MTRHierarchicalSynthesisError("generation settings differ")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTRHierarchicalSynthesisError("environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTRHierarchicalSynthesisError("environment contract differs")

    all_sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    sources = {
        row["identity_sha256"]: row
        for row in all_sources
        if row["split"] == "development"
    }
    candidate_path_groups = (
        (
            args.synthesis_candidates,
            args.stacked_candidates,
            args.self_refinement_candidates,
            args.offset_one_candidates,
            args.level_two_candidates,
            args.challenger_candidates,
        )
        if args.mode in adjudication_modes
        else (
            args.synthesis_candidates,
            args.stacked_candidates,
            args.self_refinement_candidates,
        )
    )
    groups = {
        role: load_candidate_group(paths, expected_paths=expected)
        for role, paths, expected in zip(
            contract["roles"],
            candidate_path_groups,
            contract["path_counts"],
            strict=True,
        )
    }
    guidance = None
    if args.mode == "guided_multi_trajectory_adjudication":
        if args.guidance_candidates is None:
            raise Q36MTRHierarchicalSynthesisError("guidance candidates are missing")
        guidance = load_candidate_group(args.guidance_candidates, expected_paths=1)
    elif args.guidance_candidates is not None:
        raise Q36MTRHierarchicalSynthesisError("unexpected guidance candidates")
    if len(sources) != ROWS or any(
        set(group) != set(sources) for group in groups.values()
    ):
        raise Q36MTRHierarchicalSynthesisError("identity coverage differs")
    if guidance is not None and set(guidance) != set(sources):
        raise Q36MTRHierarchicalSynthesisError("guidance identity coverage differs")
    ordered_identities = sorted(sources)
    row_start = ROWS * args.shard_index // args.shard_count
    row_end = ROWS * (args.shard_index + 1) // args.shard_count
    shard_identities = ordered_identities[row_start:row_end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = load_q36_adapter_model(
        args.model_root, args.aligned_checkpoint
    )
    validate_aligned_metadata(metadata)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    output_by_identity: dict[str, dict[str, Any]] = {}
    prompt_tokens = generated_tokens = exhausted = 0
    preserved_rows = adjudicated_rows = 0
    started = time.monotonic()
    prompts_by_identity: dict[str, str] = {}
    plans: dict[str, dict[str, Any]] = {}
    if args.mode in adjudication_modes:
        for identity in shard_identities:
            candidates = {role: groups[role][identity] for role in contract["roles"]}
            if args.mode == "guided_multi_trajectory_adjudication":
                assert guidance is not None
                preserve, prompt, plan = guided_multi_trajectory_adjudication_plan(
                    sources[identity]["source_prompt"],
                    candidates,
                    guidance[identity],
                )
            else:
                preserve, prompt, plan = multi_trajectory_adjudication_plan(
                    sources[identity]["source_prompt"], candidates
                )
            plans[identity] = plan
            if preserve is None:
                if prompt is None:
                    raise Q36MTRHierarchicalSynthesisError(
                        "adjudication prompt is missing"
                    )
                prompts_by_identity[identity] = prompt
                continue
            selected = candidates[preserve]
            selected_completion_sha256 = hashlib.sha256(
                selected["completion"].encode()
            ).hexdigest()
            preservation_receipt = (
                f"{plan['decision']}:{identity}:{preserve}:"
                f"{selected_completion_sha256}"
            )
            output_by_identity[identity] = {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": selected["task"],
                "prompt_sha256": hashlib.sha256(
                    preservation_receipt.encode()
                ).hexdigest(),
                "owner_checkpoint_sha256": sha256_file(args.aligned_checkpoint),
                "model_revision": MODEL_REVISION,
                "completion": selected["completion"],
                "generated_tokens": selected["generated_tokens"],
                "max_token_exhausted": selected["max_token_exhausted"],
                "finish_reason": selected["finish_reason"],
                "wall_seconds": 0.0,
                "hierarchical_synthesis": {
                    "schema": HIERARCHY_SCHEMA,
                    "input_roles": list(groups),
                    "development_labels_read": 0,
                    "adjudication": {**plan, "selected": preserve},
                },
            }
            preserved_rows += 1
            exhausted += int(selected["max_token_exhausted"])
        generation_identities = [
            identity for identity in shard_identities if identity in prompts_by_identity
        ]
    else:
        prompt_builder = {
            "retention_controls": hierarchical_prompt,
            "incumbent_challenger": incumbent_challenger_prompt,
            "incumbent_cyclic": incumbent_cyclic_prompt,
            "incumbent_interpolation": incumbent_interpolation_prompt,
        }[args.mode]
        generation_identities = shard_identities
        prompts_by_identity = {
            identity: prompt_builder(
                sources[identity]["source_prompt"],
                *(group[identity]["completion"] for group in groups.values()),
            )
            for identity in generation_identities
        }

    for offset in range(0, len(generation_identities), args.batch_size):
        identities = generation_identities[offset : offset + args.batch_size]
        prompts = [prompts_by_identity[identity] for identity in identities]
        rendered = [
            _render_prompt(tokenizer, prompt, True, False) for prompt in prompts
        ]
        prompt_tokens += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        batch_started = time.monotonic()
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
            add_special_tokens=False,
        )
        batch_wall_seconds = (time.monotonic() - batch_started) / len(identities)
        for identity, prompt, completion, (token_count, hit_limit) in zip(
            identities, prompts, completions, usage, strict=True
        ):
            if not isinstance(completion, str) or not completion.strip():
                raise Q36MTRHierarchicalSynthesisError(
                    "generation emitted an empty completion"
                )
            source = sources[identity]
            synthesis = {
                "schema": HIERARCHY_SCHEMA,
                "input_roles": list(groups),
                "development_labels_read": 0,
            }
            if args.mode in adjudication_modes:
                synthesis["adjudication"] = {**plans[identity], "selected": "model"}
                adjudicated_rows += 1
            output_by_identity[identity] = {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": source["task"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "owner_checkpoint_sha256": sha256_file(args.aligned_checkpoint),
                "model_revision": MODEL_REVISION,
                "completion": completion,
                "generated_tokens": int(token_count),
                "max_token_exhausted": bool(hit_limit),
                "finish_reason": "length" if hit_limit else "stop",
                "wall_seconds": batch_wall_seconds,
                "hierarchical_synthesis": synthesis,
            }
            generated_tokens += int(token_count)
            exhausted += int(hit_limit)
    outputs = [output_by_identity[identity] for identity in shard_identities]
    if len(outputs) != len(shard_identities):
        raise Q36MTRHierarchicalSynthesisError("adjudication output coverage differs")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.output, outputs)
    candidate_paths = dict(zip(contract["roles"], candidate_path_groups, strict=True))
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": contract["interpretation"],
        "mode": args.mode,
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "aligned_checkpoint": str(args.aligned_checkpoint.resolve()),
        "aligned_checkpoint_sha256": sha256_file(args.aligned_checkpoint),
        "aligned_update": metadata["update"],
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "freeze_identity_receipts": freeze_report["identity_receipts"],
        "train_source_sha256": sha256_file(args.train_source),
        "development_source_sha256": sha256_file(args.development_source),
        "candidate_sha256": {
            name: [sha256_file(path) for path in paths]
            for name, paths in candidate_paths.items()
        },
        "guidance_sha256": (
            [sha256_file(path) for path in args.guidance_candidates]
            if args.guidance_candidates is not None
            else None
        ),
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "rendered_chat_tokenization": "add_special_tokens_false",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_rows": ROWS,
        "row_start": row_start,
        "row_end": row_end,
        "rows": len(outputs),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(row["identity_sha256"] for row in outputs) + "\n").encode()
        ).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "preserved_rows": preserved_rows,
        "model_adjudicated_rows": adjudicated_rows,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "development_labels_read": 0,
        "capability_scored": False,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "retention_controls",
            "incumbent_challenger",
            "incumbent_cyclic",
            "incumbent_interpolation",
            "multi_trajectory_adjudication",
            "guided_multi_trajectory_adjudication",
        ),
        default="retention_controls",
    )
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--aligned-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument(
        "--synthesis-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--stacked-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--self-refinement-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--offset-one-candidates", type=Path, action="append")
    parser.add_argument("--level-two-candidates", type=Path, action="append")
    parser.add_argument("--challenger-candidates", type=Path, action="append")
    parser.add_argument("--guidance-candidates", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=SHARDS)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
