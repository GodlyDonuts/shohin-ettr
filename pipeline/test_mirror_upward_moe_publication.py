from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mirror_upward_moe_publication import UpwardMoEMirrorError, mirror
from q36_mtr_evidence import verify_evidence_snapshot


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path.resolve()


def _fixture(tmp_path: Path):
    authorized = tmp_path / "authorized"
    authorized.mkdir()
    analysis = _write(tmp_path / "analysis.json", b"analysis\n")
    figure = tmp_path / "figure"
    manifest = _write(figure / "manifest.json", b"manifest\n")
    svg = _write(figure / "shohin-upward-moe-scaling.svg", b"svg\n")
    csv = _write(figure / "shohin-upward-moe-scaling-points.csv", b"csv\n")
    accounting = [
        _write(tmp_path / f"accounting-{index}.json", f"accounting-{index}\n".encode())
        for index in range(2)
    ]
    points = [
        _write(tmp_path / f"point-{index}.json", f"point-{index}\n".encode())
        for index in range(4)
    ]
    publication_value = {
        "schema": "shohin-upward-moe-publication-evidence-v1",
        "status": "complete",
        "analysis": {
            "sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
            "point_source_sha256s": [
                hashlib.sha256(path.read_bytes()).hexdigest() for path in points
            ],
        },
        "figure_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "figure_records": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (svg, csv)
        ],
        "accounting_records": [
            {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in accounting
        ],
        "scientific_scores_changed": False,
        "automatic_successor_authorized": False,
    }
    publication = _write(
        tmp_path / "publication.json",
        (json.dumps(publication_value) + "\n").encode(),
    )
    return authorized, publication, analysis, figure.resolve(), accounting, points


def test_mirrors_exact_publication_and_replays_membership(tmp_path: Path) -> None:
    authorized, publication, analysis, figure, accounting, points = _fixture(tmp_path)
    output = (authorized / "upward-moe-result").resolve()
    result = mirror(
        authorized_root=authorized.resolve(),
        output_root=output,
        publication_receipt=publication,
        analysis=analysis,
        figure_root=figure,
        accounting=accounting,
        source_points=points,
    )
    assert result["status"] == "complete"
    assert result["artifact_count"] == 11
    replay = verify_evidence_snapshot(output / "manifest.json", result)
    assert replay["exact_membership"] is True
    assert not any(path.stat().st_mode & 0o222 for path in output.rglob("*"))


def test_refuses_tampered_point_or_escaping_output(tmp_path: Path) -> None:
    authorized, publication, analysis, figure, accounting, points = _fixture(tmp_path)
    points[0].write_text("tampered\n")
    with pytest.raises(UpwardMoEMirrorError, match="source point"):
        mirror(
            authorized_root=authorized.resolve(),
            output_root=(authorized / "bad").resolve(),
            publication_receipt=publication,
            analysis=analysis,
            figure_root=figure,
            accounting=accounting,
            source_points=points,
        )


def test_refuses_incomplete_four_point_set(tmp_path: Path) -> None:
    authorized, publication, analysis, figure, accounting, points = _fixture(tmp_path)
    with pytest.raises(UpwardMoEMirrorError, match="geometry"):
        mirror(
            authorized_root=authorized.resolve(),
            output_root=(authorized / "incomplete").resolve(),
            publication_receipt=publication,
            analysis=analysis,
            figure_root=figure,
            accounting=accounting,
            source_points=points[:-1],
        )
    with pytest.raises(UpwardMoEMirrorError, match="escapes"):
        mirror(
            authorized_root=authorized.resolve(),
            output_root=(tmp_path / "outside").resolve(),
            publication_receipt=publication,
            analysis=analysis,
            figure_root=figure,
            accounting=accounting,
            source_points=points,
        )


def test_job_is_cpu_only_runtime_bound_and_nonrequeueing() -> None:
    source = (
        Path(__file__).with_name("jobs") / "mirror_upward_moe_publication.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_verify_runtime" in source
    assert '[[ "$OUTPUT_ROOT" == "$AUTHORIZED_ROOT"/*' in source
    assert "${#point_paths[@]} -ge 3" in source
