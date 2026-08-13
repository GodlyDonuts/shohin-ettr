#!/usr/bin/env python3
"""Seal the terminal Q36 result into durable evidence within the final CPU job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from build_q36_mtr_custody import EVIDENCE_SCHEMA
from compare_q36_mtr import CUSTODY_SCHEMA, OUTPUT_SCHEMA
from q36_mtr_contract import SCHEMA as GRAPH_SCHEMA, validate_graph

SCHEMA = "shohin-q36-mtr-terminal-evidence-v1"


class Q36MTRTerminalEvidenceError(RuntimeError):
    """The Q36 terminal result cannot be durably sealed as claimed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, schema: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRTerminalEvidenceError("Q36 terminal evidence input differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRTerminalEvidenceError("Q36 terminal evidence schema differs")
    return value


def _output_root(authorized: Path, output: Path) -> Path:
    if (
        not authorized.is_absolute()
        or authorized.is_symlink()
        or not authorized.is_dir()
        or not output.is_absolute()
        or output.exists()
        or output.is_symlink()
    ):
        raise Q36MTRTerminalEvidenceError("Q36 terminal evidence root differs")
    root = authorized.resolve(strict=True)
    target = output.resolve(strict=False)
    if root in {Path("/"), Path.home().resolve()}:
        raise Q36MTRTerminalEvidenceError("Q36 terminal evidence root is too broad")
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise Q36MTRTerminalEvidenceError(
            "Q36 terminal evidence escapes its authorized root"
        ) from error
    if not relative.parts:
        raise Q36MTRTerminalEvidenceError(
            "Q36 terminal evidence equals its authorized root"
        )
    return target


def seal(args: argparse.Namespace) -> dict:
    output = _output_root(args.authorized_root, args.output_root)
    graph = _load(args.graph_contract, GRAPH_SCHEMA)
    validate_graph(graph)
    custody = _load(args.final_custody, CUSTODY_SCHEMA)
    preterminal = _load(args.preterminal_evidence, EVIDENCE_SCHEMA)
    result = _load(args.final_result, OUTPUT_SCHEMA)
    if (
        result.get("status") != "complete"
        or result.get("formal_result") not in {"PASS", "FAIL"}
        or not isinstance(result.get("gate_pass"), bool)
        or result.get("gate_pass") != (result.get("formal_result") == "PASS")
        or result.get("run_id") != custody.get("run_id")
        or result.get("stop_after_gate") is not True
        or result.get("automatic_retry_authorized") is not False
        or result.get("automatic_confirmation_authorized") is not False
        or result.get("automatic_successor_authorized") is not False
        or result.get("holdout_access_authorized") is not False
        or result.get("product_access_authorized") is not False
        or result.get("next_action") != "stop_and_preserve_evidence"
        or result.get("inputs", {}).get("final_custody", {}).get("sha256")
        != sha256_file(args.final_custody)
        or result.get("inputs", {}).get("graph_contract", {}).get("sha256")
        != sha256_file(args.graph_contract)
        or custody.get("status") != "complete"
        or custody.get("custody_verified") is not True
        or preterminal.get("status") != "complete"
        or preterminal.get("verified") is not True
        or preterminal.get("run_id") != result.get("run_id")
        or preterminal.get("source_commit") != graph.get("source_commit")
    ):
        raise Q36MTRTerminalEvidenceError("Q36 terminal result custody differs")
    inputs = {
        "graph_contract": args.graph_contract,
        "preterminal_evidence": args.preterminal_evidence,
        "final_custody": args.final_custody,
        "final_result": args.final_result,
    }
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True, mode=0o700)
    records = []
    try:
        for name, source in inputs.items():
            destination = temporary / f"{name}{''.join(source.suffixes)}"
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1 << 20)
                writer.flush()
                os.fsync(writer.fileno())
            if sha256_file(destination) != sha256_file(source):
                raise Q36MTRTerminalEvidenceError("Q36 terminal mirror differs")
            os.chmod(destination, 0o444)
            records.append(
                {
                    "name": name,
                    "primary": str(source.resolve()),
                    "mirror": str((output / destination.name).resolve()),
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                }
            )
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "verified": True,
            "run_id": result["run_id"],
            "source_commit": graph["source_commit"],
            "formal_result": result["formal_result"],
            "gate_pass": result["gate_pass"],
            "records": records,
            "preterminal_evidence_sha256": sha256_file(args.preterminal_evidence),
            "final_custody_sha256": sha256_file(args.final_custody),
            "final_result_sha256": sha256_file(args.final_result),
            "stop_after_gate": True,
            "retry_authorized": False,
            "successor_authorized": False,
            "holdout_access_authorized": False,
            "product_access_authorized": False,
        }
        manifest = temporary / "manifest.json"
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(manifest, 0o444)
        os.chmod(temporary, 0o555)
        os.replace(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--preterminal-evidence", type=Path, required=True)
    parser.add_argument("--final-custody", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = seal(parser.parse_args())
    print(json.dumps({"status": result["status"], "result": result["formal_result"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
