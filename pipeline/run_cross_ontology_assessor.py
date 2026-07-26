#!/usr/bin/env python3
"""Model-free final assessor for detached cross-ontology answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat


ASSESSMENT_SCHEMA = "shohin-ettr-detached-assessment-v1"
ANSWER_SCHEMA = "shohin-ettr-late-query-answer-v1"
EXPECTED_SCHEMA = "shohin-ettr-independent-expected-v1"


class CrossOntologyAssessmentError(ValueError):
    """A detached assessment artifact differs from its contract."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _read(path: Path) -> object:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise CrossOntologyAssessmentError(
            f"assessor input is not immutable regular file: {path}"
        )
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossOntologyAssessmentError(
            f"assessor input is malformed: {path}"
        ) from exc
    if payload != _canonical_json_bytes(value):
        raise CrossOntologyAssessmentError(
            f"assessor input is not canonical: {path}"
        )
    return value


def _write_once(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise CrossOntologyAssessmentError(
            "assessment output already exists"
        ) from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)


def assess(
    *,
    candidate_path: Path,
    expected_path: Path,
    output_path: Path,
) -> None:
    candidate = _read(candidate_path)
    expected = _read(expected_path)
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"schema", "token_ids"}
        or candidate["schema"] != ANSWER_SCHEMA
        or not isinstance(candidate["token_ids"], list)
    ):
        raise CrossOntologyAssessmentError(
            "candidate answer schema differs"
        )
    if (
        not isinstance(expected, dict)
        or set(expected)
        != {
            "disposition",
            "expected_token_ids",
            "schema",
        }
        or expected["schema"] != EXPECTED_SCHEMA
        or expected["disposition"]
        not in {
            "ambiguous",
            "coherent_alternate",
            "contradictory",
            "singleton",
        }
        or not isinstance(expected["expected_token_ids"], list)
    ):
        raise CrossOntologyAssessmentError(
            "independent expected schema differs"
        )
    exact = candidate["token_ids"] == expected["expected_token_ids"]
    _write_once(
        output_path,
        {
            "disposition": expected["disposition"],
            "exact": exact,
            "schema": ASSESSMENT_SCHEMA,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assess(
        candidate_path=arguments.candidate,
        expected_path=arguments.expected,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
