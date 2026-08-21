from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_sai4b_public_gate import BENCHMARKS, SaiPublicGateError, run


def _write_reports(
    tmp_path: Path, scores: dict[str, tuple[float, float, float]]
) -> list[Path]:
    paths = []
    for benchmark in BENCHMARKS:
        original, control, candidate = scores[benchmark]
        path = tmp_path / f"{benchmark}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "sai-4b-public-benchmark-score-v1",
                    "status": "complete",
                    "benchmark": benchmark,
                    "benchmark_version": "frozen-v1",
                    "rows": 100,
                    "benchmark_source_sha256": "1" * 64,
                    "identity_order_sha256": "2" * 64,
                    "prompt_contract_sha256": "3" * 64,
                    "decoding_contract_sha256": "4" * 64,
                    "original_checkpoint_sha256": "5" * 64,
                    "equal_compute_checkpoint_sha256": "6" * 64,
                    "candidate_checkpoint_sha256": "7" * 64,
                    "original_score": original,
                    "equal_compute_score": control,
                    "candidate_score": candidate,
                }
            )
            + "\n"
        )
        paths.append(path)
    return paths


def test_historical_always_revise_result_is_rejected(tmp_path: Path) -> None:
    scores = {
        "humaneval_plus": (68.9024, 70.7317, 73.1707),
        "mbpp_plus": (5.5556, 5.8201, 6.0847),
        "ifeval": (81.8854, 72.2736, 75.0462),
        "musr": (78.0423, 68.3862, 44.8413),
        "correctbench": (35.7240, 32.3410, 14.8850),
    }
    payload = run(_write_reports(tmp_path, scores), tmp_path / "gate.json")
    assert payload["decision"] == "reject_sai_candidate"
    assert payload["macro"]["candidate_vs_original_points"] == pytest.approx(-11.21636)
    assert payload["macro"]["candidate_vs_equal_compute_points"] == pytest.approx(
        -7.10494
    )
    assert not payload["checks"]["musr_nonnegative_vs_both"]
    assert not payload["checks"]["correctbench_nonnegative_vs_both"]
    assert payload["stop_candidate"] is True


def test_broad_nonregressing_candidate_promotes(tmp_path: Path) -> None:
    scores = {
        benchmark: (50.0 + index, 50.5 + index, 52.0 + index)
        for index, benchmark in enumerate(BENCHMARKS)
    }
    payload = run(_write_reports(tmp_path, scores), tmp_path / "gate.json")
    assert payload["decision"] == "promote_sai_candidate"
    assert payload["promote_to_full_confirmation"] is True
    assert payload["architecture_locked"] is False


def test_material_single_benchmark_regression_vetoes_macro_gain(tmp_path: Path) -> None:
    scores = {benchmark: (50.0, 50.0, 54.0) for benchmark in BENCHMARKS}
    scores["musr"] = (50.0, 50.0, 48.0)
    payload = run(_write_reports(tmp_path, scores), tmp_path / "gate.json")
    assert payload["macro"]["candidate_vs_original_points"] > 1.0
    assert payload["decision"] == "reject_sai_candidate"
    assert not payload["checks"]["no_benchmark_regresses_over_1_point_vs_original"]


def test_duplicate_or_unmatched_reports_fail_closed(tmp_path: Path) -> None:
    scores = {benchmark: (50.0, 50.0, 52.0) for benchmark in BENCHMARKS}
    paths = _write_reports(tmp_path, scores)
    paths[-1] = paths[0]
    with pytest.raises(SaiPublicGateError, match="one report per"):
        run(paths, tmp_path / "gate.json")


def test_invalid_score_fails_closed(tmp_path: Path) -> None:
    scores = {benchmark: (50.0, 50.0, 52.0) for benchmark in BENCHMARKS}
    scores["musr"] = (50.0, 50.0, float("nan"))
    with pytest.raises(SaiPublicGateError, match="outside"):
        run(_write_reports(tmp_path, scores), tmp_path / "gate.json")
