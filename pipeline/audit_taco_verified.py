#!/usr/bin/env python3
"""Re-run every supplied TACO test for a bounded curated code artifact.

The first curation pass uses a few tests to cheaply identify plausible programs.
This audit is the admission gate: it finds the source record for every retained
program and executes all bounded supplied stdin/stdout cases before emitting a
new immutable derivative. It never mutates its input.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from curate_apps import run_solution
from curate_taco_verified import parse_cases


def read_candidates(path):
    by_id = {}
    malformed = missing = duplicate_ids = 0
    with open(path, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            problem_id = row.get("problem_id")
            response = str(row.get("response") or "").strip()
            if problem_id is None or not response:
                missing += 1
                continue
            key = str(problem_id)
            if key in by_id:
                duplicate_ids += 1
                continue
            by_id[key] = row
    if malformed or missing or duplicate_ids:
        raise SystemExit(
            f"invalid input: malformed={malformed} missing={missing} duplicate_ids={duplicate_ids}"
        )
    return by_id


def read_completed_partial(path, candidates):
    """Load already full-verified rows for an explicit interrupted-job resume.

    The partial output is authoritative only for rows that are still present in
    the immutable curation input and whose response bytes have not changed.
    This prevents a retry from silently mixing another source or a hand-edited
    solution into the eventual frozen artifact.
    """
    completed = {}
    malformed = missing = duplicate_ids = unexpected_ids = changed_responses = 0
    with open(path, errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            problem_id = row.get("problem_id")
            response = str(row.get("response") or "").strip()
            if problem_id is None or not response or int(row.get("full_verified_cases") or 0) <= 0:
                missing += 1
                continue
            key = str(problem_id)
            if key in completed:
                duplicate_ids += 1
                continue
            original = candidates.get(key)
            if original is None:
                unexpected_ids += 1
                continue
            if response != str(original.get("response") or "").strip():
                changed_responses += 1
                continue
            completed[key] = row
    if malformed or missing or duplicate_ids or unexpected_ids or changed_responses:
        raise SystemExit(
            "invalid partial: "
            f"malformed={malformed} missing={missing} duplicate_ids={duplicate_ids} "
            f"unexpected_ids={unexpected_ids} changed_responses={changed_responses}"
        )
    return completed


def verify_source_row(row, source, max_case_chars, timeout):
    """Return a full-test-verified immutable derivative or its rejection key."""
    cases = parse_cases(source.get("input_output"), max_tests=0,
                        max_case_chars=max_case_chars)
    if not cases:
        return None, "missing_cases"
    if not run_solution(str(row["response"]), cases, timeout):
        return None, "execution"
    clean = dict(row)
    clean["full_verified_cases"] = len(cases)
    clean["training_group"] = "code"
    clean["verification"] = "execution_verified"
    return clean, None


def audit_source_stream(
    *,
    candidates,
    stream,
    output,
    resumed_rows,
    workers,
    progress_every,
    max_source_rows,
    max_case_chars,
    timeout,
):
    """Overlap source discovery with bounded verification and durable output.

    Keeping only a small FIFO of futures preserves deterministic source-order
    output while ensuring a time-limited audit writes resumable progress during
    the source scan instead of waiting for the entire dataset traversal.
    """
    remaining = dict(candidates)
    pending = deque()
    kept = resumed_rows
    matched = resumed_rows
    checked = resumed_rows
    source_rows = 0
    drops = {key: 0 for key in ("missing_cases", "execution", "source_unmatched")}

    def verify(match):
        row, source = match
        return verify_source_row(row, source, max_case_chars, timeout)

    def record(future):
        nonlocal checked, kept
        clean, drop = future.result()
        checked += 1
        if drop:
            drops[drop] += 1
        else:
            output.write(json.dumps(clean, ensure_ascii=False) + "\n")
            kept += 1
        if checked % progress_every == 0:
            output.flush()
            os.fsync(output.fileno())
            print(
                f"[taco-full-audit] matched={matched} checked={checked} kept={kept} "
                f"source_rows_scanned={source_rows} source_remaining={len(remaining)} "
                f"verification_pending={len(pending)} resumed={resumed_rows} "
                f"workers={workers}",
                flush=True,
            )

    max_pending = max(workers * 2, 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for source in stream:
            source_rows += 1
            if max_source_rows and source_rows > max_source_rows:
                break
            key = str(source.get("id"))
            row = remaining.pop(key, None)
            if row is None:
                continue
            matched += 1
            pending.append(executor.submit(verify, (row, source)))
            if len(pending) >= max_pending:
                record(pending.popleft())
            if not remaining:
                break
        while pending:
            record(pending.popleft())

    output.flush()
    os.fsync(output.fileno())
    drops["source_unmatched"] = len(remaining)
    print(
        f"[taco-full-audit] matched={matched} checked={checked} kept={kept} "
        f"source_rows_scanned={source_rows} source_remaining={len(remaining)} "
        f"verification_pending=0 resumed={resumed_rows} workers={workers}",
        flush=True,
    )
    return {
        "source_rows_scanned": source_rows,
        "matched_source_rows": matched,
        "checked_source_rows": checked,
        "kept": kept,
        "remaining": remaining,
        "drops": drops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="likaixin/TACO-verified")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-case-chars", type=int, default=20_000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-source-rows", type=int, default=0,
                        help="0 scans until every input problem is found")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel full-test workers; keep at or below allocated CPUs")
    parser.add_argument("--resume-partial", action="store_true",
                        help="append to a checked interrupted .partial output instead of refusing it")
    args = parser.parse_args()
    if (args.timeout <= 0 or args.max_case_chars <= 0 or args.max_source_rows < 0
            or args.progress_every <= 0 or args.workers <= 0):
        raise ValueError("timeout, max-case-chars, and progress-every must be positive; "
                         "max-source-rows non-negative")

    from datasets import load_dataset

    candidates = read_candidates(args.input)
    out_path = Path(args.out)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out_path}")
    if partial.exists() and not args.resume_partial:
        raise SystemExit(f"refusing stale partial without --resume-partial: {partial}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = read_completed_partial(partial, candidates) if partial.exists() else {}
    candidates = {key: row for key, row in candidates.items() if key not in completed}
    resumed = len(completed)
    stream = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
        revision=args.revision,
    )
    with partial.open("a" if resumed else "w") as out:
        result = audit_source_stream(
            candidates=candidates,
            stream=stream,
            output=out,
            resumed_rows=resumed,
            workers=args.workers,
            progress_every=args.progress_every,
            max_source_rows=args.max_source_rows,
            max_case_chars=args.max_case_chars,
            timeout=args.timeout,
        )
    if result["remaining"]:
        raise SystemExit(
            f"[taco-full-audit] source records missing for {len(result['remaining'])} input rows; "
            f"durable partial retained at {partial}"
        )
    os.replace(partial, out_path)
    print(json.dumps({
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "input_rows": result["matched_source_rows"] + len(result["remaining"]),
        "source_rows_scanned": result["source_rows_scanned"],
        "matched_source_rows": result["matched_source_rows"],
        "checked_source_rows": result["checked_source_rows"],
        "kept": result["kept"],
        "resumed_preverified_rows": resumed,
        "dropped": result["drops"],
        "out": str(out_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
