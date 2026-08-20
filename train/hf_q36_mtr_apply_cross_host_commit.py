#!/usr/bin/env python3
"""Apply the frozen Q36 model-owned commit policy to another MoE host."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from hf_aqc1_train_commit import IndependentCommitHead, select_candidate
from hf_pcf1_train_commit import hidden_states
from hf_q36_mtr_evaluate import load_q36_adapter_model, validate_adapter
from hf_q36_mtr_train_commit import (
    COMMIT_PROJECTION_CONTRACT,
    HEAD_WIDTH,
    MAX_SEQUENCE_LENGTH,
    MODEL_SCHEMA,
    Q36MTRCommitError,
    commit_token_rows,
    restore_commit_state,
)
from q36_mtr_roles import MODEL_REVISION, TRAINABLE_PARAMETERS

SELECTION_SCHEMA = "shohin-q36-cross-host-semantic-commit-selection-v1"
REPORT_SCHEMA = "shohin-q36-cross-host-semantic-commit-application-v1"
TASKS = ("math500", "bbh_logic", "mbpp", "mmlu_pro")
LINEAGES = ("revision", "unchanged")
REVISION_RELIABILITY_VETOES = ("none", "empty_or_exhausted")
HOSTS: dict[str, dict[str, Any]] = {
    "gpt_oss_120b_screen": {
        "rows": 256,
        "shards": 4,
        "candidate_schema": "shohin-gpt-oss-120b-fixed-draft-candidate-v1",
        "source_schema": "shohin-q36-mtr-external-validation-source-v1",
        "source_split": "external_validation",
    },
    "gpt_oss_120b_confirmation": {
        "rows": 256,
        "shards": 4,
        "candidate_schema": "shohin-gpt-oss-120b-fixed-draft-candidate-v1",
        "source_schema": "shohin-q36-mtr-external-validation-source-v1",
        "source_split": "external_validation",
    },
    "mixtral_8x22b_validation": {
        "rows": 1_023,
        "shards": 16,
        "candidate_schema": "shohin-mixtral-8x22b-fixed-draft-candidate-v1",
        "source_schema": "shohin-q36-mtr-external-validation-source-v1",
        "source_split": "external_validation",
    },
}


class CrossHostCommitError(RuntimeError):
    """The cross-host semantic-commit inputs or execution differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CrossHostCommitError(f"missing or linked input: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CrossHostCommitError("cross-host JSONL row differs")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise CrossHostCommitError(f"unreadable input: {path}") from error
    if not rows:
        raise CrossHostCommitError(f"empty input: {path}")
    return rows


def load_source(
    path: Path, contract: dict[str, Any], identities: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    for row in _jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != contract["source_schema"]
            or row.get("split") != contract["source_split"]
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in observed
            or row.get("task") not in TASKS
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or any(field in row for field in ("assessor", "answer", "gold", "correct"))
        ):
            raise CrossHostCommitError("cross-host source projection differs")
        observed.add(identity)
        if identities is None or identity in identities:
            result[identity] = row
    if len(result) != contract["rows"] or (
        identities is not None and set(result) != identities
    ):
        raise CrossHostCommitError("cross-host source coverage differs")
    return result


def load_candidates(
    paths: list[Path], lineage: str, contract: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if lineage not in LINEAGES or len(paths) != contract["shards"]:
        raise CrossHostCommitError("cross-host candidate geometry differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != contract["candidate_schema"]
                or row.get("arm") != lineage
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in result
                or row.get("task") not in TASKS
                or not isinstance(row.get("completion"), str)
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise CrossHostCommitError("cross-host candidate differs")
            result[identity] = row
    if len(result) != contract["rows"]:
        raise CrossHostCommitError("cross-host candidate coverage differs")
    return result


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise CrossHostCommitError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CrossHostCommitError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_environment(args: argparse.Namespace) -> None:
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise CrossHostCommitError("cross-host environment bytes differ")
    payload = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "shohin-q36-mtr-environment-v1"
        or payload.get("status") != "pass"
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise CrossHostCommitError("cross-host environment contract differs")


def select_pair(
    margin: float,
    reverse_margin: float,
    candidates: list[dict[str, Any]],
    revision_margin_threshold: float,
) -> tuple[int, int]:
    """Return direct/reversed choices under a semantic revision threshold."""
    if revision_margin_threshold == 0.0:
        return (
            select_candidate(margin, candidates),
            select_candidate(reverse_margin, list(reversed(candidates))),
        )
    return (
        0 if margin > revision_margin_threshold else 1,
        1 if -reverse_margin > revision_margin_threshold else 0,
    )


def apply_revision_reliability_veto(
    chosen: int,
    reversed_choice: int,
    revision_candidate: dict[str, Any],
    mode: str,
) -> tuple[int, int, bool]:
    """Fall back to unchanged when the revision failed to terminate reliably."""
    if mode not in REVISION_RELIABILITY_VETOES:
        raise CrossHostCommitError("cross-host revision reliability veto differs")
    unreliable = (
        not revision_candidate["completion"].strip()
        or revision_candidate["max_token_exhausted"]
    )
    vetoed = mode == "empty_or_exhausted" and chosen == 0 and unreliable
    return (1, 0, True) if vetoed else (chosen, reversed_choice, False)


def apply(args: argparse.Namespace) -> dict[str, Any]:
    contract = HOSTS[args.host]
    if (
        args.model_revision != MODEL_REVISION
        or args.max_sequence_length != MAX_SEQUENCE_LENGTH
        or not math.isfinite(args.revision_margin_threshold)
        or args.revision_margin_threshold < 0.0
        or args.revision_reliability_veto not in REVISION_RELIABILITY_VETOES
        or args.batch_identities <= 0
        or any(
            path.exists() or path.is_symlink()
            for path in (args.output, args.selections, args.report)
        )
    ):
        raise CrossHostCommitError("cross-host pinned settings differ")
    _validate_environment(args)
    owners = {
        "revision": load_candidates(args.revision_candidates, "revision", contract),
        "unchanged": load_candidates(args.unchanged_candidates, "unchanged", contract),
    }
    identities = set(owners["revision"])
    if any(set(owner) != identities for owner in owners.values()):
        raise CrossHostCommitError("cross-host identities differ")
    source = load_source(args.source, contract, identities)
    for identity in identities:
        if (
            len(
                {
                    source[identity]["task"],
                    *(owner[identity]["task"] for owner in owners.values()),
                }
            )
            != 1
        ):
            raise CrossHostCommitError("cross-host task binding differs")

    payload = torch.load(args.commit_checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != MODEL_SCHEMA
        or not isinstance(metadata, dict)
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("head_width") != HEAD_WIDTH
        or metadata.get("commit_projection_contract") != COMMIT_PROJECTION_CONTRACT
        or metadata.get("adapter_checkpoint_sha256")
        != sha256_file(args.adapter_checkpoint)
    ):
        raise CrossHostCommitError("cross-host commit checkpoint differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, loader = load_q36_adapter_model(
        args.model_root, args.adapter_checkpoint
    )
    if loader != "causal":
        raise CrossHostCommitError("cross-host model loader differs")
    trainable_receipt = validate_adapter(model, adapter_metadata, "revision")
    trainable = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if sum(parameter.numel() for _, parameter in trainable) != TRAINABLE_PARAMETERS:
        raise CrossHostCommitError("cross-host adapter geometry differs")
    hidden_size = int(model.text_model.embed_tokens.embedding_dim)
    head = IndependentCommitHead(hidden_size, HEAD_WIDTH).to("cuda:0")
    try:
        restore_commit_state(trainable, head, payload)
    except Q36MTRCommitError as error:
        raise CrossHostCommitError("cross-host commit restore differs") from error
    model.eval()
    head.eval()

    selected_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    reliability_veto_counts: Counter[str] = Counter()
    prompt_truncated = 0
    maximum_swap_error = 0.0
    ordered = sorted(identities)
    with torch.inference_mode():
        for start in range(0, len(ordered), args.batch_identities):
            local_identities = ordered[start : start + args.batch_identities]
            encoded: list[list[int]] = []
            pair_rows: list[dict[str, Any]] = []
            for identity in local_identities:
                pair = {
                    "question": source[identity]["source_prompt"],
                    "candidates": [
                        {
                            "lineage": lineage,
                            "completion": owners[lineage][identity]["completion"],
                        }
                        for lineage in LINEAGES
                    ],
                }
                local, truncated = commit_token_rows(
                    tokenizer, pair, args.max_sequence_length
                )
                encoded.extend(local)
                pair_rows.append(pair)
                prompt_truncated += truncated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = hidden_states(model, encoded, tokenizer.pad_token_id)
                paired = hidden.reshape(-1, 2, hidden.shape[-1])
                direct = head.margin(paired[:, 0], paired[:, 1]).float()
                reverse = head.margin(paired[:, 1], paired[:, 0]).float()
            maximum_swap_error = max(
                maximum_swap_error, float((direct + reverse).abs().max().cpu())
            )
            for identity, pair, margin, reverse_margin in zip(
                local_identities,
                pair_rows,
                direct.tolist(),
                reverse.tolist(),
                strict=True,
            ):
                chosen, reversed_choice = select_pair(
                    margin,
                    reverse_margin,
                    pair["candidates"],
                    args.revision_margin_threshold,
                )
                semantic_chosen = chosen
                chosen, reversed_choice, reliability_vetoed = (
                    apply_revision_reliability_veto(
                        chosen,
                        reversed_choice,
                        owners["revision"][identity],
                        args.revision_reliability_veto,
                    )
                )
                consistent = chosen == 1 - reversed_choice or (
                    pair["candidates"][0]["completion"]
                    == pair["candidates"][1]["completion"]
                )
                lineage = LINEAGES[chosen]
                selected_counts[lineage] += 1
                if reliability_vetoed:
                    reliability_veto_counts["revision_to_unchanged"] += 1
                selected_rows.append(owners[lineage][identity])
                selection_rows.append(
                    {
                        "schema": SELECTION_SCHEMA,
                        "host": args.host,
                        "identity_sha256": identity,
                        "task": source[identity]["task"],
                        "selected_index": chosen,
                        "selected_lineage": lineage,
                        "semantic_selected_lineage": LINEAGES[semantic_chosen],
                        "margin": float(margin),
                        "revision_margin_threshold": args.revision_margin_threshold,
                        "revision_reliability_veto": args.revision_reliability_veto,
                        "reliability_veto_applied": reliability_vetoed,
                        "revision_candidate_empty": not owners["revision"][identity][
                            "completion"
                        ].strip(),
                        "revision_candidate_max_token_exhausted": owners["revision"][
                            identity
                        ]["max_token_exhausted"],
                        "order_consistent": consistent,
                    }
                )
    if maximum_swap_error != 0.0 or not all(
        row["order_consistent"] for row in selection_rows
    ):
        raise CrossHostCommitError("cross-host order consistency differs")
    output_sha256 = _atomic_lines(args.output, selected_rows)
    selections_sha256 = _atomic_lines(args.selections, selection_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": args.host,
        "rows": len(selection_rows),
        "selected": dict(sorted(selected_counts.items())),
        "lineages": list(LINEAGES),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "revision_candidate_sha256s": [
            sha256_file(path) for path in args.revision_candidates
        ],
        "unchanged_candidate_sha256s": [
            sha256_file(path) for path in args.unchanged_candidates
        ],
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "selections": str(args.selections.resolve()),
        "selections_sha256": selections_sha256,
        "commit_checkpoint_sha256": sha256_file(args.commit_checkpoint),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "prompt_truncated": prompt_truncated,
        "maximum_swap_error": maximum_swap_error,
        "order_consistent": len(selection_rows),
        "revision_margin_threshold": args.revision_margin_threshold,
        "revision_reliability_veto": args.revision_reliability_veto,
        "reliability_veto_counts": dict(sorted(reliability_veto_counts.items())),
        "model_visible_fields": [
            "question",
            "candidate_a.completion",
            "candidate_b.completion",
        ],
        "deterministic_control_fields": [
            "revision.completion_is_empty",
            "revision.max_token_exhausted",
        ],
        "commit_projection_contract": COMMIT_PROJECTION_CONTRACT,
        "task_correctness_or_host_label_visible": False,
        "assessor_access_count": 0,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "trainable_parameter_name_sha256": trainable_receipt[
            "trainable_parameter_name_sha256"
        ],
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(HOSTS), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--commit-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--revision-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--unchanged-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--batch-identities", type=int, default=2)
    parser.add_argument("--revision-margin-threshold", type=float, default=0.0)
    parser.add_argument(
        "--revision-reliability-veto",
        choices=REVISION_RELIABILITY_VETOES,
        default="none",
    )
    return parser.parse_args()


def main() -> int:
    print(json.dumps(apply(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
