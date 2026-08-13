from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from compare_q36_mtr import OUTPUT_SCHEMA
from q36_mtr_contract import MODEL_REVISION
from render_q36_mtr_publication_figure import (
    PAIRED_CSV_NAME,
    Q36MTRFigureError,
    SCALING_CSV_NAME,
    SVG_NAME,
    render,
)
from score_q36_mtr import build_publication_analysis


def _analysis() -> dict:
    rows = []
    tasks = ("math500", "bbh_logic", "mbpp")
    for index in range(1_289):
        rows.append(
            {
                "identity_sha256": hashlib.sha256(
                    f"figure-outcome-{index}".encode()
                ).hexdigest(),
                "task": tasks[index % len(tasks)],
                "correct": {
                    "revision": index % 5 != 0,
                    "unchanged": index % 3 == 0,
                    "self_refinement": index % 4 == 0,
                    "draft_hidden": index % 7 == 0,
                    "learned_commit": index % 6 != 0,
                },
            }
        )
    return build_publication_analysis(rows)


def _result(path: Path, mutate=None) -> Path:
    analysis = _analysis()
    if mutate is not None:
        mutate(analysis)
    value = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "run_id": "q36-figure-test",
        "model_revision": MODEL_REVISION,
        "formal_result": "PASS",
        "gate_pass": True,
        "publication_analysis": analysis,
        "publication_analysis_non_gating": True,
        "stop_after_gate": True,
        "automatic_retry_authorized": False,
        "automatic_confirmation_authorized": False,
        "automatic_successor_authorized": False,
        "next_action": "stop_and_preserve_evidence",
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_render_q36_publication_figure_is_atomic_manifested_and_deterministic(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path / "result.json")
    outputs = []
    manifests = []
    for name in ("figure-a", "figure-b"):
        output = tmp_path / name
        manifests.append(
            render(argparse.Namespace(final_result=result, output_root=output))
        )
        outputs.append(output)
    assert manifests[0] == manifests[1]
    for output in outputs:
        assert {path.name for path in output.iterdir()} == {
            SVG_NAME,
            SCALING_CSV_NAME,
            PAIRED_CSV_NAME,
            "manifest.json",
        }
        assert not any(path.stat().st_mode & 0o222 for path in output.rglob("*"))
        svg = (output / SVG_NAME).read_text(encoding="utf-8")
        assert "Dense 9B" in svg
        assert "MoE 35B-A3B" in svg
        assert "not a direct compute-scaling law" in svg
        assert "nan" not in svg.casefold()
        scaling = list(
            csv.DictReader(
                io.StringIO((output / SCALING_CSV_NAME).read_text(encoding="utf-8"))
            )
        )
        assert len(scaling) == 8
        assert {row["arm"] for row in scaling if "35B-A3B" in row["model"]} == {
            "unchanged",
            "trained_revision",
            "learned_commit",
            "self_refinement",
            "draft_hidden",
        }
        assert all(
            row["cross_board_absolute_comparison_authorized"] == "false"
            for row in scaling
        )


def test_render_q36_publication_figure_rejects_tampered_statistics(
    tmp_path: Path,
) -> None:
    result = _result(
        tmp_path / "result.json",
        lambda analysis: analysis["comparisons"]["revision_vs_unchanged"][
            "overall"
        ].__setitem__("net_correct", 999),
    )
    with pytest.raises(Q36MTRFigureError):
        render(
            argparse.Namespace(
                final_result=result,
                output_root=tmp_path / "publication",
            )
        )


def test_render_q36_publication_figure_refuses_existing_output(tmp_path: Path) -> None:
    result = _result(tmp_path / "result.json")
    output = tmp_path / "publication"
    output.mkdir()
    with pytest.raises(Q36MTRFigureError):
        render(argparse.Namespace(final_result=result, output_root=output))
