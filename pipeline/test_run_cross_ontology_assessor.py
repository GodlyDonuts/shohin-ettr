from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from run_cross_ontology_assessor import (
    ANSWER_SCHEMA,
    ASSESSMENT_SCHEMA,
    EXPECTED_SCHEMA,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o444)


def test_assessor_runs_after_candidate_with_no_model_inputs(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    expected = tmp_path / "expected.json"
    output = tmp_path / "assessment.json"
    _write(
        candidate,
        {
            "schema": ANSWER_SCHEMA,
            "token_ids": [[3, 1, 4]],
        },
    )
    assert not expected.exists()
    _write(
        expected,
        {
            "disposition": "coherent_alternate",
            "expected_token_ids": [[3, 1, 4]],
            "schema": EXPECTED_SCHEMA,
        },
    )
    runner = Path(__file__).with_name(
        "run_cross_ontology_assessor.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "--candidate",
            str(candidate),
            "--expected",
            str(expected),
            "--output",
            str(output),
        ],
        check=True,
        cwd=runner.parent.parent,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text()) == {
        "disposition": "coherent_alternate",
        "exact": True,
        "schema": ASSESSMENT_SCHEMA,
    }
    assert output.stat().st_mode & 0o222 == 0


def test_assessor_source_has_no_candidate_runtime_imports() -> None:
    source = Path(__file__).with_name(
        "run_cross_ontology_assessor.py"
    ).read_text()
    for forbidden in (
        "import torch",
        "from model import",
        "endogenous_typed_theory_reactor",
        "cross_ontology_horn_board",
        "cross_ontology_rewrite_board",
        "cross_ontology_resource_board",
        "--checkpoint",
        "--state",
        "--source",
        "--world",
        "--query",
    ):
        assert forbidden not in source
