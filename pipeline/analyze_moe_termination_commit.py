#!/usr/bin/env python3
"""Replay a label-free termination-aware commit over scored MoE candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


class TerminationCommitError(RuntimeError):
    """The candidate or scoring boundary differs from the declared replay."""


ARMS = ("unchanged", "self_refinement", "revision")
TASKS = ("bbh_logic", "math500", "mbpp")


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TerminationCommitError(f"missing regular input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(
    paths: list[Path], arm: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not paths:
        raise TerminationCommitError(f"{arm} candidates are absent")
    rows: dict[str, dict[str, Any]] = {}
    hashes: list[str] = []
    for path in paths:
        hashes.append(sha256_file(path))
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TerminationCommitError(
                        f"{arm} candidate {path}:{line_number} is not JSON"
                    ) from exc
                identity = row.get("identity_sha256")
                task = row.get("task")
                generated_tokens = row.get("generated_tokens")
                if (
                    not isinstance(row.get("schema"), str)
                    or row.get("arm") != arm
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or any(
                        character not in "0123456789abcdef" for character in identity
                    )
                    or identity in rows
                    or task not in TASKS
                    or not isinstance(row.get("completion"), str)
                    or isinstance(generated_tokens, bool)
                    or not isinstance(generated_tokens, int)
                    or generated_tokens < 0
                    or not isinstance(row.get("max_token_exhausted"), bool)
                ):
                    raise TerminationCommitError(
                        f"{arm} candidate {path}:{line_number} differs"
                    )
                rows[identity] = row
    if not rows:
        raise TerminationCommitError(f"{arm} candidate set is empty")
    return rows, hashes


def load_outcomes(path: Path) -> tuple[dict[str, dict[str, Any]], str, str]:
    score_sha256 = sha256_file(path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TerminationCommitError("score report is not JSON") from exc
    outcomes: dict[str, dict[str, Any]] = {}
    for row in report.get("outcomes", []):
        identity = row.get("identity_sha256")
        task = row.get("task")
        correct = row.get("correct")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or identity in outcomes
            or task not in TASKS
            or not isinstance(correct, dict)
            or any(not isinstance(correct.get(arm), bool) for arm in ARMS)
        ):
            raise TerminationCommitError("score outcome differs")
        outcomes[identity] = row
    if not outcomes:
        raise TerminationCommitError("score outcomes are empty")
    schema = report.get("schema")
    if not isinstance(schema, str):
        raise TerminationCommitError("score schema differs")
    return outcomes, score_sha256, schema


def identity_digest(identities: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{identity}\n" for identity in identities).encode()
    ).hexdigest()


def exact_sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def replay(
    *,
    host: str,
    score: Path,
    candidate_paths: dict[str, list[Path]],
) -> dict[str, Any]:
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_hashes: dict[str, list[str]] = {}
    for arm in ARMS:
        arms[arm], candidate_hashes[arm] = load_candidates(candidate_paths[arm], arm)
    outcomes, score_sha256, score_schema = load_outcomes(score)
    identity_set = set(outcomes)
    if any(set(arms[arm]) != identity_set for arm in ARMS):
        raise TerminationCommitError("candidate and score identity coverage differs")
    for identity, outcome in outcomes.items():
        if any(arms[arm][identity]["task"] != outcome["task"] for arm in ARMS):
            raise TerminationCommitError("candidate and score task binding differs")

    identities = sorted(identity_set)
    selectors: dict[str, Any] = {}
    for baseline in ("unchanged", "self_refinement"):
        selected: dict[str, str] = {}
        for identity in identities:
            base = arms[baseline][identity]
            revision = arms["revision"][identity]
            selected[identity] = (
                "revision"
                if base["max_token_exhausted"] and not revision["max_token_exhausted"]
                else baseline
            )
        baseline_correct = [
            identity
            for identity in identities
            if outcomes[identity]["correct"][baseline]
        ]
        selected_correct = [
            identity
            for identity in identities
            if outcomes[identity]["correct"][selected[identity]]
        ]
        retained = sum(
            outcomes[identity]["correct"][selected[identity]]
            for identity in baseline_correct
        )
        wins = sum(
            outcomes[identity]["correct"][selected[identity]]
            and not outcomes[identity]["correct"][baseline]
            for identity in identities
        )
        losses = sum(
            outcomes[identity]["correct"][baseline]
            and not outcomes[identity]["correct"][selected[identity]]
            for identity in identities
        )
        selected_revision = sorted(
            identity for identity in identities if selected[identity] == "revision"
        )
        selectors[baseline] = {
            "baseline_correct": len(baseline_correct),
            "selected_correct": len(selected_correct),
            "gain_correct": len(selected_correct) - len(baseline_correct),
            "retained_baseline_correct": retained,
            "baseline_correct_retention": retained / len(baseline_correct),
            "paired_wins": wins,
            "paired_losses": losses,
            "exact_two_sided_sign_p": exact_sign_test(wins, losses),
            "revision_selected": len(selected_revision),
            "revision_selected_identity_sha256": identity_digest(selected_revision),
            "domains": {
                task: {
                    "correct": sum(
                        outcomes[identity]["correct"][selected[identity]]
                        for identity in identities
                        if outcomes[identity]["task"] == task
                    ),
                    "total": sum(
                        outcomes[identity]["task"] == task for identity in identities
                    ),
                }
                for task in TASKS
            },
        }

    return {
        "schema": "shohin-moe-termination-aware-commit-analysis-v1",
        "status": "complete_retrospective_replay",
        "host": host,
        "row_count": len(identities),
        "ordered_identity_sha256": identity_digest(identities),
        "rule": {
            "name": "select_revision_only_when_baseline_exhausted_and_revision_not_exhausted",
            "model_visible_features": [
                "baseline_max_token_exhausted",
                "revision_max_token_exhausted",
            ],
            "uses_task_label": False,
            "uses_correctness_at_selection": False,
            "uses_assessor_at_selection": False,
            "uses_completion_text": False,
        },
        "selectors": selectors,
        "evidence": {
            "score_path": str(score.resolve()),
            "score_schema": score_schema,
            "score_sha256": score_sha256,
            "candidate_paths": {
                arm: [str(path.resolve()) for path in candidate_paths[arm]]
                for arm in ARMS
            },
            "candidate_sha256s": candidate_hashes,
        },
        "interpretation_boundary": {
            "development_only": True,
            "predeclared_confirmation": False,
            "qualified_release": False,
            "claim": "retrospective_label_free_trajectory_state_diagnostic",
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TerminationCommitError("refusing existing output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--score", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = replay(
        host=args.host,
        score=args.score,
        candidate_paths={arm: getattr(args, f"{arm}_candidates") for arm in ARMS},
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
