#!/usr/bin/env python3
"""Generate one matched Q36-MTR calibration or label-free development arm."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from hf_pcf1_evaluate import (
    nonpadding_prompt_tokens,
    self_refinement_prompt,
    shard_bounds,
)
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    SANDBOX_CONFIG_SHA256,
    atomic_json as sandbox_atomic_json,
    mbpp_allocation_setup_receipts_sha256,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from q36_mtr_roles import (
    CONTROLLED_LAYERS,
    MODEL_REVISION,
    Q36MTRRoleError,
    TRAINABLE_PARAMETERS,
    validate_contract,
)

DATA_SCHEMA = "shohin-q36-mtr-eval-v1"
DATA_REPORT_SCHEMA = "shohin-q36-mtr-data-report-v1"
CANDIDATE_SCHEMA = "shohin-q36-mtr-candidate-v1"
REPORT_SCHEMA = "shohin-q36-mtr-evaluation-v1"
ARMS = ("revision", "unchanged", "self_refinement", "draft_hidden")
SPLITS = ("calibration", "development")
TASKS = ("math500", "bbh_logic", "mbpp")
EVALUATION_SEED = 2026080816
EXPECTED_SHARDS = {"calibration": 4, "development": 8}
EXPECTED_FULL_ROWS = {"calibration": 5_824, "development": 1_289}
ROLE_BY_ARM = {
    "revision": "aligned",
    "unchanged": "owner",
    "self_refinement": "owner",
    "draft_hidden": "draft_hidden",
}


class Q36MTREvaluationError(RuntimeError):
    """The Q36-MTR matched evaluation boundary differs."""


def reject_protected_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(term in rendered for term in ("holdout", "product", "public")):
        raise Q36MTREvaluationError(f"protected path supplied to Q36-MTR: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTREvaluationError(f"refusing existing Q36 candidate output: {path}")
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
        raise Q36MTREvaluationError(f"refusing existing Q36 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def model_visible_runtime_fields(arm: str) -> list[str]:
    if arm not in ARMS:
        raise Q36MTREvaluationError("Q36-MTR arm differs")
    return (
        ["source_prompt", "internal_draft.completion"]
        if arm == "self_refinement"
        else ["question"]
    )


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = str(row.get("identity_sha256", ""))
        if (
            row.get("schema") != DATA_SCHEMA
            or row.get("split") != split
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or row.get("runtime_fields")
            != (
                ["question"]
                if split == "calibration"
                else ["question", "source_prompt"]
            )
            or row.get("internal_draft_visible") is not True
            or row.get("external_candidate_text_visible") is not False
            or row.get("candidates") != []
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise Q36MTREvaluationError("Q36-MTR evaluation row differs")
        draft = row.get("internal_draft")
        if (
            not isinstance(draft, dict)
            or draft.get("identity_sha256") != identity
            or not isinstance(draft.get("completion"), str)
            or draft["completion"] not in row["question"]
        ):
            raise Q36MTREvaluationError("Q36-MTR draft binding differs")
        if split == "calibration":
            if not isinstance(row.get("assessor"), dict):
                raise Q36MTREvaluationError("Q36-MTR calibration assessor is absent")
        elif any(
            field in row
            for field in ("assessor", "answer", "correct", "gold", "response", "target")
        ):
            raise Q36MTREvaluationError("Q36-MTR development exposes supervision")
        identities.add(identity)
        rows.append(row)
    if len(rows) != EXPECTED_FULL_ROWS[split] or {row["task"] for row in rows} != set(
        TASKS
    ):
        raise Q36MTREvaluationError("Q36-MTR evaluation coverage differs")
    return rows


def validate_adapter(model: Any, metadata: Any, arm: str) -> dict[str, Any]:
    role = ROLE_BY_ARM[arm]
    if not isinstance(metadata, dict):
        raise Q36MTREvaluationError("Q36-MTR adapter metadata is absent")
    try:
        validate_contract(metadata, role)
    except Q36MTRRoleError as error:
        raise Q36MTREvaluationError(str(error)) from error
    trainables = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    names = [name for name, _ in trainables]
    name_sha256 = hashlib.sha256("\n".join(names).encode()).hexdigest()
    layer_count = len(model.text_model.layers)
    expected_indices = list(range(layer_count - CONTROLLED_LAYERS, layer_count))
    expected_draft_bytes = role != "owner"
    expected_draft_information = role == "aligned"
    if (
        int(metadata.get("update", -1)) != 256
        or metadata.get("trainable_parameters") != TRAINABLE_PARAMETERS
        or sum(int(parameter.numel()) for _, parameter in trainables)
        != TRAINABLE_PARAMETERS
        or metadata.get("trainable_parameter_name_sha256") != name_sha256
        or metadata.get("controlled_layer_indices") != expected_indices
        or metadata.get("draft_control")
        != ("draft_unavailable" if arm == "draft_hidden" else "normal")
        or metadata.get("draft_token_bytes_present") is not expected_draft_bytes
        or metadata.get("draft_information_available") is not expected_draft_information
        or metadata.get("internal_draft_visible") is not expected_draft_information
        or metadata.get("draft_attention_applied") is not (arm == "draft_hidden")
    ):
        raise Q36MTREvaluationError("Q36-MTR adapter trainables differ")
    return {
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_parameter_name_sha256": name_sha256,
        "controlled_layer_indices": expected_indices,
        "role": role,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    for path in (
        args.data,
        args.data_report,
        args.candidates_output,
        args.report,
        args.adapter_checkpoint,
        args.environment_receipt,
    ):
        reject_protected_path(path)
    if args.sandbox_probe_output is not None:
        reject_protected_path(args.sandbox_probe_output)
    if (
        args.arm not in ARMS
        or args.split not in SPLITS
        or args.model_revision != MODEL_REVISION
        or args.seed != EVALUATION_SEED
        or args.shard_count != EXPECTED_SHARDS[args.split]
        or args.batch_size != 1
        or not 0 <= args.shard_index < args.shard_count
        or (args.split == "calibration" and args.arm not in {"revision", "unchanged"})
    ):
        raise Q36MTREvaluationError("Q36-MTR evaluation settings differ")
    if args.candidates_output.exists() or args.report.exists():
        raise Q36MTREvaluationError("Q36-MTR evaluation output exists")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTREvaluationError("Q36-MTR environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTREvaluationError("Q36-MTR environment contract differs")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split)
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("model_revision") != MODEL_REVISION
        or data_report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or not isinstance(expected, dict)
        or Path(str(expected.get("path", ""))).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise Q36MTREvaluationError("Q36-MTR data report differs")
    all_rows = load_rows(args.data, args.split)
    row_start, row_end = shard_bounds(
        len(all_rows), args.shard_index, args.shard_count, args.batch_size
    )
    rows = all_rows[row_start:row_end]

    sandbox_payload = None
    sandbox_receipt_sha256 = None
    setup_receipts: list[dict[str, Any]] = []
    if args.split == "calibration":
        if args.sandbox_probe_output is None:
            raise Q36MTREvaluationError("Q36-MTR calibration sandbox receipt is absent")
        sandbox_payload = qualify_allocation()
        sandbox_receipt_sha256 = sandbox_atomic_json(
            args.sandbox_probe_output, sandbox_payload
        )
        setup_receipts = qualify_mbpp_assessor_setups([row["assessor"] for row in rows])
    elif args.sandbox_probe_output is not None:
        raise Q36MTREvaluationError("Q36-MTR development must not score code")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, "causal", quantization="nf4"
    )
    trainable_receipt = validate_adapter(model, metadata, args.arm)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    started = time.monotonic()
    for row in rows:
        question = (
            self_refinement_prompt(row)
            if args.arm == "self_refinement"
            else str(row["question"])
        )
        rendered = [_render_prompt(tokenizer, question, True, False)]
        counters["prompt_tokens"] += nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            model, tokenizer, rendered, True, "greedy", 768, stop_ids
        )
        completion = completions[0]
        token_count, exhausted = usage[0]
        candidate = {
            "schema": CANDIDATE_SCHEMA,
            "arm": args.arm,
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "completion": completion,
            "generated_tokens": token_count,
            "max_token_exhausted": exhausted,
        }
        if args.split == "calibration":
            score = score_completion(row["assessor"], completion)
            candidate.update(score)
            counters["correct"] += int(score["correct"])
            counters["sandbox_executions"] += int(row["task"] == "mbpp")
        candidates.append(candidate)
        counters["rows"] += 1
        counters["generated_tokens"] += token_count
        counters["max_token_exhausted"] += int(exhausted)
        counters["empty_completions"] += int(not completion.strip())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_sha256 = _atomic_lines(args.candidates_output, candidates)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": args.split,
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **trainable_receipt,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "runtime_fields": model_visible_runtime_fields(args.arm),
        "assessor_fields_visible_to_model": False,
        "assessment_mode": (
            "calibration_immediate"
            if args.split == "calibration"
            else "development_deferred"
        ),
        "assessor_board_access_count": 0,
        "generation_mode": "greedy",
        "max_new_tokens": 768,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": row_start,
        "row_end": row_end,
        "full_row_count": len(all_rows),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
        "sandbox_probe_sha256": (
            sandbox_payload.get("probe_sha256") if sandbox_payload else None
        ),
        "sandbox_status": (
            "passed" if args.split == "calibration" else "not_applicable_no_scoring"
        ),
        "mbpp_setup_receipts": setup_receipts,
        "mbpp_setup_receipts_sha256": (
            mbpp_allocation_setup_receipts_sha256(setup_receipts)
            if args.split == "calibration"
            else None
        ),
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sandbox-probe-output", type=Path)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"arm": report["arm"], "split": report["split"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
