#!/usr/bin/env python3
"""Evaluate PSET1 edit programs, causal interventions, and executed trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from hf_product_reasoning_eval import _load_model
from pset1_pointer_transducer import EditProgram, KEEP, PSET1Config, PSET1Error, PSET1PointerHead, REPLACE, execute_program
from pset1_runtime import BYTE_EOS, host_hidden, load_rows, pad_characters, pad_ids, sha256_file, tokenize_rows


REPORT_SCHEMA = "shohin-pset1-pointer-evaluation-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def decode_replacement(
    head: PSET1PointerHead,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    characters: torch.Tensor,
    pointers: torch.Tensor,
) -> tuple[list[str | None], list[bool], int]:
    batch = source.shape[0]
    generated = torch.full((batch, 1), BYTE_EOS, device=source.device, dtype=torch.long)
    values = [[] for _ in range(batch)]
    finished = [False] * batch
    for step in range(head.config.max_replacement_tokens + 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head.replacement_logits(source, source_mask, characters, pointers, generated)
        next_ids = logits[:, -1].float().argmax(dim=-1)
        for index, token in enumerate(next_ids.tolist()):
            if finished[index]:
                continue
            if token == BYTE_EOS:
                finished[index] = True
            elif step < head.config.max_replacement_tokens:
                values[index].append(token)
        if all(finished):
            break
        if step == head.config.max_replacement_tokens:
            break
        generated = torch.cat((generated, next_ids[:, None]), dim=1)
    decoded = []
    for value in values:
        try:
            decoded.append(bytes(value).decode("utf-8"))
        except UnicodeDecodeError:
            decoded.append(None)
    return decoded, finished, sum(len(value) for value in values)


def shuffled_donors(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["corruption_family"]].append(row)
    output = {}
    for family_rows in grouped.values():
        ordered = sorted(family_rows, key=lambda row: (row["members"]["fault"]["draft_token_count"], row["source_identity_sha256"]))
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            if donor["source_identity_sha256"] == row["source_identity_sha256"]:
                raise RuntimeError("PSET1 shuffled donor identity matches")
            output[row["source_identity_sha256"]] = donor
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("PSET1 output exists or shard differs")
    rows, data_report = load_rows(args.data, args.data_report, "diagnostic")
    tokenized = tokenize_rows(AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True), rows)
    donors = shuffled_donors(tokenized)
    selected = [row for index, row in enumerate(tokenized) if index % args.shard_count == args.shard_index]
    payload = torch.load(args.pointer_checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata.get("arm") != args.expected_arm or metadata.get("host_checkpoint_sha256") != sha256_file(args.host_checkpoint):
        raise RuntimeError("PSET1 pointer checkpoint differs")
    config = PSET1Config(**metadata["config"])
    head = PSET1PointerHead(config).to("cuda:0")
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    host, host_metadata, loader = _load_model(args.model_root, args.host_checkpoint, "causal")
    if host_metadata.get("dset1_arm") != "aligned":
        raise RuntimeError("PSET1 host differs")
    host.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    generated_bytes = 0
    for row in selected:
        presented = donors[row["source_identity_sha256"]] if args.intervention == "shuffled" else row
        members = [presented["members"]["clean"], presented["members"]["fault"]]
        source_ids, source_mask = pad_ids([row["source_ids"]], tokenizer.pad_token_id, torch.device("cuda:0"))
        draft_ids, draft_mask = pad_ids([member["draft_ids"] for member in members], tokenizer.pad_token_id, torch.device("cuda:0"))
        mapping, character_ids, character_mask = pad_characters(members, torch.device("cuda:0"))
        source_hidden = host_hidden(host, source_ids, source_mask).expand(2, -1, -1)
        source_mask_pair = source_mask.expand(2, -1)
        draft_hidden = host_hidden(host, draft_ids, draft_mask)
        if args.intervention == "hidden":
            draft_hidden = torch.zeros_like(draft_hidden)
            character_ids = torch.zeros_like(character_ids)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            source, characters, action_logits, pointer_logits = head.encode(
                source_hidden, source_mask_pair, draft_hidden, draft_mask,
                mapping, character_ids, character_mask,
            )
        actions = action_logits.float().argmax(dim=-1)
        pointers = pointer_logits.float().argmax(dim=-1)
        replacements, finished, byte_count = decode_replacement(
            head, source, source_mask_pair, characters, pointers
        )
        generated_bytes += byte_count
        for index, member_name in enumerate(("clean", "fault")):
            gold_member = row["members"][member_name]
            presented_member = members[index]
            replacement = replacements[index]
            program = EditProgram(
                int(actions[index]),
                None if actions[index].item() == KEEP else int(pointers[index, 0]),
                None if actions[index].item() == KEEP else int(pointers[index, 1]),
                "" if actions[index].item() == KEEP or replacement is None else replacement,
            )
            executed = ""
            error = None
            try:
                executed = execute_program(presented_member["draft"], presented_member["offsets"], program)
            except PSET1Error as exc:
                error = str(exc)
            expected_action = KEEP if member_name == "clean" else REPLACE
            program_exact = (
                program.action == expected_action
                and (
                    member_name == "clean"
                    or (
                        program.start == gold_member["pointer_start"]
                        and program.end == gold_member["pointer_end"]
                        and program.replacement == row["new_surface"]
                    )
                )
            )
            force_keep_breaks = member_name == "fault" and executed == row["final_response"] and presented_member["draft"] != row["final_response"]
            results.append({
                "source_identity_sha256": row["source_identity_sha256"],
                "pair_member": member_name,
                "corruption_family": row["corruption_family"],
                "predicted_action": program.action,
                "predicted_start": program.start,
                "predicted_end": program.end,
                "predicted_replacement": program.replacement,
                "replacement_finished": finished[index],
                "program_exact": program_exact,
                "execution_correct": executed == row["final_response"],
                "execution_error": error,
                "force_keep_breaks_correct_repair": force_keep_breaks,
            })
    counts = Counter()
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    pairs = defaultdict(list)
    for result in results:
        pairs[result["source_identity_sha256"]].append(result)
        for key in ("program_exact", "execution_correct", "replacement_finished", "force_keep_breaks_correct_repair"):
            counts[key] += int(result[key])
            family[result["corruption_family"]][key] += int(result[key])
            member[result["pair_member"]][key] += int(result[key])
        family[result["corruption_family"]]["rows"] += 1
        member[result["pair_member"]]["rows"] += 1
    consistency = sum(len(value) == 2 and all(item["program_exact"] for item in value) for value in pairs.values())
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.expected_arm,
        "intervention": args.intervention,
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pair_count": len(selected),
        "row_count": len(results),
        **counts,
        "counterfactual_consistent_pairs": consistency,
        "execution_errors": dict(Counter(result["execution_error"] for result in results if result["execution_error"])),
        "family_counts": {key: dict(value) for key, value in family.items()},
        "member_counts": {key: dict(value) for key, value in member.items()},
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "pointer_checkpoint_sha256": sha256_file(args.pointer_checkpoint),
        "host_checkpoint_sha256": sha256_file(args.host_checkpoint),
        "pointer_metadata": metadata,
        "model_loader": loader,
        "generated_bytes": generated_bytes,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--host-checkpoint", type=Path, required=True)
    parser.add_argument("--pointer-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-arm", choices=["aligned", "permuted"], required=True)
    parser.add_argument("--intervention", choices=["normal", "hidden", "shuffled"], default="normal")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026080917)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
