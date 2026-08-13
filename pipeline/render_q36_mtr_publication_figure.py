#!/usr/bin/env python3
"""Render deterministic Q36 publication SVG/CSV assets from one terminal result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from compare_q36_mtr import OUTPUT_SCHEMA
from score_q36_mtr import PUBLICATION_CLAIMS, validate_publication_analysis

SCHEMA = "shohin-q36-mtr-publication-figure-manifest-v1"
SVG_NAME = "shohin-q36-architecture-transfer.svg"
SCALING_CSV_NAME = "shohin-q36-scaling-points.csv"
PAIRED_CSV_NAME = "shohin-q36-paired-effects.csv"
COMMON_ARMS = ("unchanged", "trained_revision", "learned_commit")
COMPARISON_LABELS = {
    "revision_vs_unchanged": "Revision − unchanged",
    "revision_vs_self_refinement": "Revision − self-refine",
    "revision_vs_draft_hidden": "Revision − draft-hidden",
    "learned_commit_vs_revision": "Commit − revision",
    "learned_commit_vs_unchanged": "Commit − unchanged",
}


class Q36MTRFigureError(RuntimeError):
    """The terminal result cannot support the frozen publication figure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Q36MTRFigureError("Q36 terminal result path differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36MTRFigureError("Q36 terminal result is unreadable") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != OUTPUT_SCHEMA
        or value.get("status") != "complete"
        or value.get("formal_result") not in {"PASS", "FAIL"}
        or value.get("gate_pass") != (value.get("formal_result") == "PASS")
        or value.get("publication_analysis_non_gating") is not True
        or value.get("stop_after_gate") is not True
        or value.get("automatic_retry_authorized") is not False
        or value.get("automatic_confirmation_authorized") is not False
        or value.get("automatic_successor_authorized") is not False
        or value.get("next_action") != "stop_and_preserve_evidence"
    ):
        raise Q36MTRFigureError("Q36 terminal result contract differs")
    try:
        validate_publication_analysis(value.get("publication_analysis"))
    except RuntimeError as error:
        raise Q36MTRFigureError("Q36 publication analysis differs") from error
    return value


def _csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _scaling_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for point in analysis["scaling_graph_data"]["points"]:
        for arm in point["scores"]:
            rows.append(
                {
                    "model": point["model"],
                    "board": point["board"],
                    "arm": arm,
                    "correct": point["scores"][arm],
                    "total": point["total"],
                    "percent": f'{point["arm_percentages"][arm]:.9f}',
                    "cross_board_absolute_comparison_authorized": "false",
                }
            )
    return rows


def _paired_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    claim_for_comparison = {
        comparison: claim for claim, comparison in PUBLICATION_CLAIMS.items()
    }
    for name in COMPARISON_LABELS:
        comparison = analysis["comparisons"][name]
        claim_name = claim_for_comparison.get(name)
        claim = (
            analysis["claim_evidence"]["claims"][claim_name]
            if claim_name is not None
            else None
        )
        for scope, summary in (
            ("overall", comparison["overall"]),
            *tuple((domain, value) for domain, value in comparison["domains"].items()),
        ):
            interval = summary["paired_wald_95_ci_percentage_points"]
            probability = summary["mcnemar_exact_two_sided"]
            rows.append(
                {
                    "comparison": name,
                    "scope": scope,
                    "treatment": comparison["treatment"],
                    "control": comparison["control"],
                    "rows": summary["rows"],
                    "treatment_only_correct": summary["treatment_only_correct"],
                    "control_only_correct": summary["control_only_correct"],
                    "net_correct": summary["net_correct"],
                    "risk_difference_percentage_points": (
                        f'{summary["risk_difference_percentage_points"]:.9f}'
                    ),
                    "ci95_lower_percentage_points": f"{interval[0]:.9f}",
                    "ci95_upper_percentage_points": f"{interval[1]:.9f}",
                    "mcnemar_exact_p": f'{probability["value"]:.17g}',
                    "mcnemar_exact_numerator": probability["numerator"],
                    "mcnemar_exact_denominator": probability["denominator"],
                    "publication_claim": claim_name or "none",
                    "holm_rejected": (
                        str(claim["holm_rejected"]).lower()
                        if claim is not None
                        else "not_applicable"
                    ),
                    "publication_claim_supported": (
                        str(claim["publication_claim_supported"]).lower()
                        if claim is not None
                        else "not_applicable"
                    ),
                }
            )
    return rows


def _svg(analysis: dict[str, Any], formal_result: str) -> bytes:
    points = analysis["scaling_graph_data"]["points"]
    dense, moe = points
    width, height = 1200, 720
    left, top = 78, 110
    panel_width, panel_height = 500, 455
    baseline = top + panel_height
    arm_labels = {
        "unchanged": "Unchanged",
        "trained_revision": "Revision",
        "learned_commit": "Commit",
    }
    colors = {
        "unchanged": "#64748b",
        "trained_revision": "#2563eb",
        "learned_commit": "#d97706",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Shohin dense-to-MoE architecture transfer</title>',
        '<desc id="desc">Qualified dense reference and source-disjoint Q36 MoE development results, plus paired causal effect intervals. Different boards are not a compute scaling law.</desc>',
        '<rect width="1200" height="720" fill="#ffffff"/>',
        '<text x="600" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600" fill="#0f172a">Shohin architecture transfer: dense 9B → MoE 35B-A3B</text>',
        f'<text x="600" y="59" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">Terminal result: {html.escape(formal_result)} · paired effects are descriptive and non-gating</text>',
        f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#cbd5e1"/>',
        f'<text x="{left}" y="{top - 18}" font-family="sans-serif" font-size="16" font-weight="600" fill="#0f172a">A. Architecture transfer scores</text>',
    ]
    for tick in range(0, 101, 20):
        y = baseline - tick / 100 * panel_height
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{tick}%</text>',
            ]
        )
    group_centers = (left + 145, left + 365)
    for group_index, point in enumerate((dense, moe)):
        center = group_centers[group_index]
        offsets = (-54, 0, 54)
        for arm, offset in zip(COMMON_ARMS, offsets, strict=True):
            value = float(point["arm_percentages"][arm])
            bar_height = value / 100 * panel_height
            x = center + offset - 19
            y = baseline - bar_height
            parts.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="38" height="{bar_height:.2f}" fill="{colors[arm]}"/>',
                    f'<text x="{x + 19:.2f}" y="{y - 7:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0f172a">{value:.1f}%</text>',
                ]
            )
        model_label = "Dense 9B" if group_index == 0 else "MoE 35B-A3B"
        board_label = (
            "qualified product n=538" if group_index == 0 else "development n=1,289"
        )
        parts.extend(
            [
                f'<text x="{center}" y="{baseline + 25}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#0f172a">{model_label}</text>',
                f'<text x="{center}" y="{baseline + 43}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{board_label}</text>',
            ]
        )
    legend_y = baseline + 77
    for index, arm in enumerate(COMMON_ARMS):
        x = left + 70 + index * 145
        parts.extend(
            [
                f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{colors[arm]}"/>',
                f'<text x="{x + 18}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#334155">{arm_labels[arm]}</text>',
            ]
        )
    panel2_left = 650
    effects = [analysis["comparisons"][name]["overall"] for name in COMPARISON_LABELS]
    supported_comparisons = {
        value["comparison"]
        for value in analysis["claim_evidence"]["claims"].values()
        if value["publication_claim_supported"]
    }
    extrema = [0.0]
    for effect in effects:
        extrema.extend(effect["paired_wald_95_ci_percentage_points"])
    minimum, maximum = min(extrema), max(extrema)
    padding = max(2.0, (maximum - minimum) * 0.12)
    low, high = minimum - padding, maximum + padding
    if math.isclose(low, high):
        low, high = -1.0, 1.0

    def effect_y(value: float) -> float:
        return top + panel_height - (value - low) / (high - low) * panel_height

    parts.extend(
        [
            f'<rect x="{panel2_left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#cbd5e1"/>',
            f'<text x="{panel2_left}" y="{top - 18}" font-family="sans-serif" font-size="16" font-weight="600" fill="#0f172a">B. Matched Q36 effects (95% CI)</text>',
        ]
    )
    for index in range(5):
        tick = low + (high - low) * index / 4
        y = effect_y(tick)
        parts.extend(
            [
                f'<line x1="{panel2_left}" y1="{y:.2f}" x2="{panel2_left + panel_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{panel2_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{tick:+.1f}</text>',
            ]
        )
    parts.append(
        f'<line x1="{panel2_left}" y1="{effect_y(0):.2f}" x2="{panel2_left + panel_width}" y2="{effect_y(0):.2f}" stroke="#64748b" stroke-dasharray="4 4"/>'
    )
    for index, (name, effect) in enumerate(
        zip(COMPARISON_LABELS, effects, strict=True)
    ):
        x = panel2_left + 55 + index * 98
        value = effect["risk_difference_percentage_points"]
        interval = effect["paired_wald_95_ci_percentage_points"]
        y_value, y_low, y_high = (
            effect_y(value),
            effect_y(interval[0]),
            effect_y(interval[1]),
        )
        marker = "*" if name in supported_comparisons else ""
        parts.extend(
            [
                f'<line x1="{x}" y1="{y_high:.2f}" x2="{x}" y2="{y_low:.2f}" stroke="#2563eb" stroke-width="2"/>',
                f'<line x1="{x - 6}" y1="{y_high:.2f}" x2="{x + 6}" y2="{y_high:.2f}" stroke="#2563eb" stroke-width="2"/>',
                f'<line x1="{x - 6}" y1="{y_low:.2f}" x2="{x + 6}" y2="{y_low:.2f}" stroke="#2563eb" stroke-width="2"/>',
                f'<circle cx="{x}" cy="{y_value:.2f}" r="5" fill="#2563eb"/>',
                f'<text x="{x}" y="{y_value - 10:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#0f172a">{value:+.1f} pp{marker}</text>',
            ]
        )
        words = COMPARISON_LABELS[name].split(" ")
        parts.extend(
            [
                f'<text x="{x}" y="{baseline + 25}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#334155">{html.escape(" ".join(words[:2]))}</text>',
                f'<text x="{x}" y="{baseline + 39}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#334155">{html.escape(" ".join(words[2:]))}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="18" y="350" transform="rotate(-90 18 350)" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Correct answers (%)</text>',
            '<text x="600" y="350" transform="rotate(-90 600 350)" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Paired risk difference (percentage points)</text>',
            '<text x="600" y="678" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">* Preregistered claim supported after Holm-Bonferroni correction.</text>',
            '<text x="600" y="700" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">Boards are source-disjoint and differ in composition; panel A shows architecture transfer, not a direct compute-scaling law.</text>',
            "</svg>\n",
        ]
    )
    return "".join(parts).encode("utf-8")


def render(args: argparse.Namespace) -> dict[str, Any]:
    if (
        not args.output_root.is_absolute()
        or args.output_root.exists()
        or args.output_root.is_symlink()
        or args.output_root.parent.is_symlink()
        or not args.output_root.parent.is_dir()
    ):
        raise Q36MTRFigureError("Q36 publication output root differs")
    result = _load_result(args.final_result)
    analysis = result["publication_analysis"]
    scaling = _csv_bytes(
        _scaling_rows(analysis),
        (
            "model",
            "board",
            "arm",
            "correct",
            "total",
            "percent",
            "cross_board_absolute_comparison_authorized",
        ),
    )
    paired = _csv_bytes(
        _paired_rows(analysis),
        (
            "comparison",
            "scope",
            "treatment",
            "control",
            "rows",
            "treatment_only_correct",
            "control_only_correct",
            "net_correct",
            "risk_difference_percentage_points",
            "ci95_lower_percentage_points",
            "ci95_upper_percentage_points",
            "mcnemar_exact_p",
            "mcnemar_exact_numerator",
            "mcnemar_exact_denominator",
            "publication_claim",
            "holm_rejected",
            "publication_claim_supported",
        ),
    )
    svg = _svg(analysis, result["formal_result"])
    payloads = {SVG_NAME: svg, SCALING_CSV_NAME: scaling, PAIRED_CSV_NAME: paired}
    temporary = args.output_root.with_name(
        f".{args.output_root.name}.tmp.{os.getpid()}"
    )
    temporary.mkdir(mode=0o700)
    try:
        records = []
        for name, value in payloads.items():
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o444)
            records.append(
                {"name": name, "sha256": sha256_bytes(value), "bytes": len(value)}
            )
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "run_id": result["run_id"],
            "model_revision": result["model_revision"],
            "formal_result": result["formal_result"],
            "final_result_sha256": sha256_file(args.final_result),
            "records": records,
            "claim_evidence": {
                "draft_visibility_causal_supported": analysis["claim_evidence"][
                    "draft_visibility_causal_supported"
                ],
                "dense_pattern_replication_supported": analysis["claim_evidence"][
                    "dense_pattern_replication_supported"
                ],
                "publication_claim_supported": {
                    name: value["publication_claim_supported"]
                    for name, value in analysis["claim_evidence"]["claims"].items()
                },
            },
            "publication_analysis_non_gating": True,
            "cross_board_absolute_score_comparison_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_successor_authorized": False,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(manifest_path, 0o444)
        os.chmod(temporary, 0o555)
        os.replace(temporary, args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = render(parser.parse_args())
    print(json.dumps({"status": result["status"], "result": result["formal_result"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
