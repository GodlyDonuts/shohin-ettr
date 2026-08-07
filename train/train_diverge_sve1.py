#!/usr/bin/env python3
"""Train one matched DIVERGE-SVE1 spanless value-event arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_data import scan_integer_spans
from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_oqb1_runtime import exact_occurrence_quotient
from diverge_sve1_data import (
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SEED,
    validate_training_record,
)
from diverge_sve1_runtime import (
    CHECKPOINT_SCHEMA,
    EVIDENCE_BLANK_ID,
    INITIAL_BLANK_ID,
    SpanlessValueEventConfig,
    SpanlessValueEventTransducer,
    greedy_ctc_decode,
    tensorize_event_sources,
)


REPORT_SCHEMA = "shohin-diverge-sve1-training-report-v1"
UPDATES = 1_500
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CONTROL_OFFSET = 50_021


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("SVE1 training data hash differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        validate_training_record(row)
    return rows


def _load_data_report(
    path: Path,
    expected_sha256: str,
    data: Path,
    data_sha256: str,
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("SVE1 data report hash differs")
    report = json.loads(path.read_text())
    training = report.get("files", {}).get("training", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or not report.get("zero_source_name_and_identity_overlap")
        or not report.get("training_deterministic_regeneration")
        or Path(str(training.get("path", ""))).name != data.name
        or training.get("sha256") != data_sha256
        or int(training.get("rows", -1)) != TRAIN_ROWS
    ):
        raise RuntimeError("SVE1 data report custody differs")
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError("SVE1 temporary checkpoint already exists")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _targets(
    rows: Sequence[Mapping[str, Any]], key: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [tuple(int(value) for value in row[key]) for row in rows]
    return (
        torch.tensor(
            [value for sequence in sequences for value in sequence],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
            device=device,
        ),
    )


def _frame_targets(
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    text_key: str,
    target_key: str,
    blank_id: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-only alignment; no span reaches the candidate runtime."""
    output = torch.full(
        (len(source_rows), width), blank_id, dtype=torch.long, device=device
    )
    for row_index, (source, target) in enumerate(
        zip(source_rows, target_rows, strict=True)
    ):
        quotient, _, _ = exact_occurrence_quotient(
            str(source[text_key]), source["register_table"], mode="coherent"
        )
        spans = scan_integer_spans(quotient)
        events = tuple(int(value) for value in target[target_key])
        if len(spans) != len(events):
            raise RuntimeError("SVE1 teacher alignment geometry differs")
        for event, (start, end) in zip(events, spans, strict=True):
            output[row_index, start + 1 : end + 1] = event
    return output


def _ctc_and_frame_loss(
    model: SpanlessValueEventTransducer,
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "evidence":
        text_key = "evidence_text"
        target_key = "evidence_event_targets"
        expected = (2, 2)
        blank = EVIDENCE_BLANK_ID
    elif kind == "initial":
        text_key = "initial_text"
        target_key = "initial_event_targets"
        expected = (1, 1)
        blank = INITIAL_BLANK_ID
    else:
        raise RuntimeError("SVE1 training event kind differs")
    tensors = tensorize_event_sources(
        source_rows,
        device,
        text_key=text_key,
        expected_occurrences=expected,
    )
    if not bool(tensors[3].all()):
        raise RuntimeError("SVE1 training quotient is incomplete")
    logits = model(tensors[0], tensors[1], kind=kind)  # type: ignore[arg-type]
    target_values, target_lengths = _targets(target_rows, target_key, device)
    ctc = torch.nn.functional.ctc_loss(
        logits.log_softmax(dim=-1).transpose(0, 1),
        target_values,
        tensors[2],
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )
    frame_targets = _frame_targets(
        source_rows,
        target_rows,
        text_key=text_key,
        target_key=target_key,
        blank_id=blank,
        width=logits.shape[1],
        device=device,
    )
    frame = torch.nn.functional.cross_entropy(
        logits[tensors[1]], frame_targets[tensors[1]]
    )
    return frame + 0.1 * ctc, frame, ctc


@torch.no_grad()
def _sample_accuracy(
    model: SpanlessValueEventTransducer,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, float | int]:
    sample = rows[: min(2_048, len(rows))]
    predicted: dict[str, list[tuple[int, ...]]] = {"evidence": [], "initial": []}
    for kind, text_key, expected, blank in (
        ("evidence", "evidence_text", (2, 2), EVIDENCE_BLANK_ID),
        ("initial", "initial_text", (1, 1), INITIAL_BLANK_ID),
    ):
        for start in range(0, len(sample), 64):
            batch = sample[start : start + 64]
            tensors = tensorize_event_sources(
                batch,
                device,
                text_key=text_key,
                expected_occurrences=expected,
            )
            predicted[kind].extend(
                greedy_ctc_decode(
                    model(tensors[0], tensors[1], kind=kind),  # type: ignore[arg-type]
                    tensors[2],
                    blank_id=blank,
                )
            )
    evidence = initial = joint = 0
    for index, row in enumerate(sample):
        evidence_ok = predicted["evidence"][index] == tuple(
            int(value) for value in row["evidence_event_targets"]
        )
        initial_ok = predicted["initial"][index] == tuple(
            int(value) for value in row["initial_event_targets"]
        )
        evidence += evidence_ok
        initial += initial_ok
        joint += evidence_ok and initial_ok
    total = len(sample)
    return {
        "evidence_exact": evidence,
        "initial_exact": initial,
        "joint_exact": joint,
        "total": total,
        "joint_rate": joint / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--arm", choices=("treatment", "shuffled_targets"), required=True
    )
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SVE1 output: {args.output}")
    if (
        args.updates != UPDATES
        or args.batch_size != BATCH_SIZE
        or not math.isclose(args.learning_rate, LEARNING_RATE)
    ):
        raise SystemExit("SVE1 frozen training schedule differs")
    args.output.mkdir(parents=True)
    _load_data_report(
        args.data_report,
        args.data_report_sha256,
        args.data,
        args.data_sha256,
    )
    rows = _load_jsonl(args.data, args.data_sha256)
    if len(rows) != TRAIN_ROWS or not torch.cuda.is_available():
        raise SystemExit("SVE1 training geometry/CUDA differs")
    device = torch.device("cuda")

    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    model = SpanlessValueEventTransducer().to(device)
    initial_state_sha256 = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(TRAIN_SEED + 1)
    history = []
    charged = 0
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            len(rows), (args.batch_size,), generator=generator
        ).tolist()
        source_rows = [rows[index] for index in indices]
        target_rows = (
            source_rows
            if args.arm == "treatment"
            else [rows[(index + CONTROL_OFFSET) % len(rows)] for index in indices]
        )
        evidence_loss, evidence_frame, evidence_ctc = _ctc_and_frame_loss(
            model, source_rows, target_rows, device, kind="evidence"
        )
        initial_loss, initial_frame, initial_ctc = _ctc_and_frame_loss(
            model, source_rows, target_rows, device, kind="initial"
        )
        loss = evidence_loss + initial_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("SVE1 training became nonfinite")
        optimizer.step()
        charged += args.batch_size
        if update in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, args.updates):
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "evidence_frame_loss": float(evidence_frame.detach()),
                    "evidence_ctc_loss": float(evidence_ctc.detach()),
                    "initial_frame_loss": float(initial_frame.detach()),
                    "initial_ctc_loss": float(initial_ctc.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    elapsed = time.perf_counter() - started
    model.eval()
    sample = _sample_accuracy(model, rows, device)
    final_state_sha256 = module_state_sha256(model)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "arm": args.arm,
        "config": asdict(SpanlessValueEventConfig()),
        "model_state": model.state_dict(),
        "model_state_sha256": final_state_sha256,
        "initial_state_sha256": initial_state_sha256,
        "data_sha256": args.data_sha256,
        "data_report_sha256": args.data_report_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "control_offset": CONTROL_OFFSET,
    }
    checkpoint_path = args.output / "checkpoint.pt"
    _atomic_torch(checkpoint_path, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": args.source_commit,
        "arm": args.arm,
        "model": model.record(),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "data_report_sha256": args.data_report_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": charged,
        "learning_rate": args.learning_rate,
        "control_offset": CONTROL_OFFSET,
        "elapsed_seconds": elapsed,
        "examples_per_second": charged / elapsed,
        "history": history,
        "training_sample": sample,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    os.chmod(checkpoint_path, 0o444)
    os.chmod(report_path, 0o444)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "sample_rate": sample["joint_rate"],
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
