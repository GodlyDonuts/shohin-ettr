#!/usr/bin/env python3
"""Render deterministic publication assets for the upward MoE scaling screen."""

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

ANALYSIS_SCHEMA = "shohin-upward-moe-scaling-analysis-v1"
MANIFEST_SCHEMA = "shohin-upward-moe-scaling-figure-manifest-v1"
SVG_NAME = "shohin-upward-moe-scaling.svg"
CSV_NAME = "shohin-upward-moe-scaling-points.csv"
Z95 = 1.959963984540054


class UpwardMoEFigureError(RuntimeError):
    """The completed scaling analysis cannot support a publication figure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UpwardMoEFigureError(f"{label} differs")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpwardMoEFigureError(f"{label} differs")
    result = float(value)
    if not math.isfinite(result):
        raise UpwardMoEFigureError(f"{label} differs")
    return result


def _paired_interval(point: dict[str, Any]) -> tuple[float, float]:
    rows = _integer(point.get("rows"), "rows", 2)
    treatment = _integer(point.get("treatment_correct"), "treatment correct")
    control = _integer(point.get("unchanged_correct"), "unchanged correct")
    wins = _integer(point.get("paired_wins"), "paired wins")
    losses = _integer(point.get("paired_losses"), "paired losses")
    both_correct = treatment - wins
    if (
        treatment > rows
        or control > rows
        or both_correct < 0
        or both_correct != control - losses
        or wins + losses + both_correct > rows
    ):
        raise UpwardMoEFigureError("paired point accounting differs")
    mean = (wins - losses) / rows
    sum_squares = wins + losses
    variance = max(0.0, (sum_squares - rows * mean * mean) / (rows - 1))
    standard_error = math.sqrt(variance / rows)
    return (
        max(-100.0, 100.0 * (mean - Z95 * standard_error)),
        min(100.0, 100.0 * (mean + Z95 * standard_error)),
    )


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise UpwardMoEFigureError("retention accounting differs")
    estimate = successes / trials
    denominator = 1.0 + Z95 * Z95 / trials
    center = (estimate + Z95 * Z95 / (2.0 * trials)) / denominator
    radius = (
        Z95
        * math.sqrt(
            estimate * (1.0 - estimate) / trials + Z95 * Z95 / (4.0 * trials * trials)
        )
        / denominator
    )
    return (100.0 * max(0.0, center - radius), 100.0 * min(1.0, center + radius))


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise UpwardMoEFigureError("analysis path differs")
    try:
        analysis = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UpwardMoEFigureError("analysis is unreadable") from error
    if (
        not isinstance(analysis, dict)
        or analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete_curve"
        or analysis.get("minimum_points_for_curve") != 3
        or analysis.get("architecture_series") != "trained_revision"
        or not isinstance(analysis.get("curve"), dict)
        or not isinstance(analysis.get("points"), list)
        or len(analysis["points"]) < 3
        or analysis.get("point_count") != len(analysis["points"])
    ):
        raise UpwardMoEFigureError("analysis contract differs")
    points: list[dict[str, Any]] = []
    for raw in analysis["points"]:
        if not isinstance(raw, dict):
            raise UpwardMoEFigureError("point differs")
        host = raw.get("host")
        if not isinstance(host, str) or not host:
            raise UpwardMoEFigureError("host differs")
        total = _integer(raw.get("total_parameters"), "total parameters", 1)
        active = _integer(raw.get("active_parameters"), "active parameters", 1)
        if active > total:
            raise UpwardMoEFigureError("parameter geometry differs")
        rows = _integer(raw.get("rows"), "rows", 2)
        treatment = _integer(raw.get("treatment_correct"), "treatment correct")
        unchanged = _integer(raw.get("unchanged_correct"), "unchanged correct", 1)
        retained = _integer(raw.get("unchanged_correct_retained"), "unchanged retained")
        gain = _finite(raw.get("gain_over_unchanged_percentage_points"), "gain")
        if (
            treatment > rows
            or unchanged > rows
            or retained > unchanged
            or not math.isclose(gain, 100.0 * (treatment - unchanged) / rows)
        ):
            raise UpwardMoEFigureError("point score accounting differs")
        retention = _finite(raw.get("unchanged_correct_retention"), "retention")
        if not math.isclose(retention, retained / unchanged):
            raise UpwardMoEFigureError("retention differs")
        interval = _paired_interval(raw)
        retention_interval = _wilson(retained, unchanged)
        points.append(
            {
                **raw,
                "total_parameters": total,
                "active_parameters": active,
                "gain_over_unchanged_percentage_points": gain,
                "gain_ci95_percentage_points": interval,
                "retention_percentage": retention * 100.0,
                "retention_ci95_percentage": retention_interval,
            }
        )
    points.sort(key=lambda value: value["active_parameters"])
    if (
        len({point["host"] for point in points}) != len(points)
        or len({point["total_parameters"] for point in points}) != len(points)
        or len({point["active_parameters"] for point in points}) != len(points)
    ):
        raise UpwardMoEFigureError("scaling geometry differs")
    return analysis, points


def _csv(points: list[dict[str, Any]]) -> bytes:
    fields = (
        "host",
        "total_parameters",
        "active_parameters",
        "rows",
        "revision_correct",
        "unchanged_correct",
        "gain_percentage_points",
        "gain_ci95_lower_percentage_points",
        "gain_ci95_upper_percentage_points",
        "paired_wins",
        "paired_losses",
        "mcnemar_exact_two_sided_p",
        "unchanged_correct_retention_percent",
        "retention_ci95_lower_percent",
        "retention_ci95_upper_percent",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for point in points:
        writer.writerow(
            {
                "host": point["host"],
                "total_parameters": point["total_parameters"],
                "active_parameters": point["active_parameters"],
                "rows": point["rows"],
                "revision_correct": point["treatment_correct"],
                "unchanged_correct": point["unchanged_correct"],
                "gain_percentage_points": (
                    f'{point["gain_over_unchanged_percentage_points"]:.9f}'
                ),
                "gain_ci95_lower_percentage_points": (
                    f'{point["gain_ci95_percentage_points"][0]:.9f}'
                ),
                "gain_ci95_upper_percentage_points": (
                    f'{point["gain_ci95_percentage_points"][1]:.9f}'
                ),
                "paired_wins": point["paired_wins"],
                "paired_losses": point["paired_losses"],
                "mcnemar_exact_two_sided_p": (
                    f'{point["mcnemar_exact_two_sided_p"]:.17g}'
                ),
                "unchanged_correct_retention_percent": (
                    f'{point["retention_percentage"]:.9f}'
                ),
                "retention_ci95_lower_percent": (
                    f'{point["retention_ci95_percentage"][0]:.9f}'
                ),
                "retention_ci95_upper_percent": (
                    f'{point["retention_ci95_percentage"][1]:.9f}'
                ),
            }
        )
    return stream.getvalue().encode("utf-8")


def _short_host(value: str) -> str:
    substitutions = (
        ("NVIDIA-Nemotron-3-", "Nemotron "),
        ("mistralai/", ""),
        ("openai/", ""),
        ("Qwen3.6-", "Qwen "),
        ("-Instruct-v0.1", ""),
        ("-FP8", ""),
    )
    for before, after in substitutions:
        value = value.replace(before, after)
    return value


def _svg(analysis: dict[str, Any], points: list[dict[str, Any]]) -> bytes:
    width, height = 1400, 900
    left, top, panel_width, panel_height = 88, 130, 565, 360
    gap = 92
    right_left = left + panel_width + gap
    retention_top, retention_height = 590, 205
    plot_left_padding, plot_right_padding = 70, 40

    gain_extrema = [0.0]
    for point in points:
        gain_extrema.extend(point["gain_ci95_percentage_points"])
    gain_low = min(gain_extrema)
    gain_high = max(gain_extrema)
    gain_padding = max(2.0, 0.16 * max(1.0, gain_high - gain_low))
    gain_low -= gain_padding
    gain_high += gain_padding

    def y_gain(value: float) -> float:
        return (
            top
            + panel_height
            - (value - gain_low) / (gain_high - gain_low) * panel_height
        )

    def x_value(value: int, field: str, panel_left: float) -> float:
        logs = [math.log10(point[field]) for point in points]
        low, high = min(logs), max(logs)
        return (
            panel_left
            + plot_left_padding
            + (math.log10(value) - low)
            / (high - low)
            * (panel_width - plot_left_padding - plot_right_padding)
        )

    retention_low = min(
        90.0,
        min(point["retention_ci95_percentage"][0] for point in points) - 2.0,
    )
    retention_high = 101.0

    def y_retention(value: float) -> float:
        return (
            retention_top
            + retention_height
            - (value - retention_low)
            / (retention_high - retention_low)
            * retention_height
        )

    palette = ("#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626", "#0891b2")
    colors = tuple(palette[index % len(palette)] for index in range(len(points)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Shohin upward mixture-of-experts scaling</title>',
        '<desc id="desc">Matched source-disjoint trained-revision gains and retention across three distinct MoE model families and parameter scales.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="700" y="38" text-anchor="middle" font-family="sans-serif" font-size="25" font-weight="700" fill="#0f172a">Shohin upward MoE transfer</text>',
        '<text x="700" y="67" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#475569">Matched 256-row source-disjoint screens · trained revision versus unchanged</text>',
        f'<text x="700" y="92" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">Capability claim: {html.escape(str(analysis.get("capability_curve_claim")))}</text>',
    ]
    for panel_left, field, title in (
        (left, "total_parameters", "A. Gain vs total parameters"),
        (right_left, "active_parameters", "B. Gain vs active parameters"),
    ):
        parts.extend(
            [
                f'<rect x="{panel_left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" stroke="#cbd5e1"/>',
                f'<text x="{panel_left}" y="{top - 18}" font-family="sans-serif" font-size="17" font-weight="600" fill="#0f172a">{title}</text>',
            ]
        )
        for index in range(5):
            tick = gain_low + (gain_high - gain_low) * index / 4
            y = y_gain(tick)
            parts.extend(
                [
                    f'<line x1="{panel_left}" y1="{y:.2f}" x2="{panel_left + panel_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                    f'<text x="{panel_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{tick:+.1f}</text>',
                ]
            )
        zero_y = y_gain(0.0)
        parts.append(
            f'<line x1="{panel_left}" y1="{zero_y:.2f}" x2="{panel_left + panel_width}" y2="{zero_y:.2f}" stroke="#64748b" stroke-dasharray="5 4"/>'
        )
        ordered = sorted(points, key=lambda point: point[field])
        polyline = " ".join(
            f'{x_value(point[field], field, panel_left):.2f},{y_gain(point["gain_over_unchanged_percentage_points"]):.2f}'
            for point in ordered
        )
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4 4"/>'
        )
        for point, color in zip(points, colors, strict=True):
            x = x_value(point[field], field, panel_left)
            value = point["gain_over_unchanged_percentage_points"]
            lower, upper = point["gain_ci95_percentage_points"]
            y, y_lower, y_upper = y_gain(value), y_gain(lower), y_gain(upper)
            label = f"{point[field] / 1_000_000_000:g}B"
            parts.extend(
                [
                    f'<line x1="{x:.2f}" y1="{y_upper:.2f}" x2="{x:.2f}" y2="{y_lower:.2f}" stroke="{color}" stroke-width="3"/>',
                    f'<line x1="{x - 7:.2f}" y1="{y_upper:.2f}" x2="{x + 7:.2f}" y2="{y_upper:.2f}" stroke="{color}" stroke-width="3"/>',
                    f'<line x1="{x - 7:.2f}" y1="{y_lower:.2f}" x2="{x + 7:.2f}" y2="{y_lower:.2f}" stroke="{color}" stroke-width="3"/>',
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}"/>',
                    f'<text x="{x:.2f}" y="{top + panel_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">{label}</text>',
                    f'<text x="{x:.2f}" y="{y - 13:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0f172a">{value:+.1f} pp</text>',
                ]
            )
    parts.extend(
        [
            f'<text x="24" y="{top + panel_height / 2}" transform="rotate(-90 24 {top + panel_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#334155">Paired gain (percentage points, 95% CI)</text>',
            f'<rect x="{left}" y="{retention_top}" width="{width - 2 * left}" height="{retention_height}" fill="none" stroke="#cbd5e1"/>',
            f'<text x="{left}" y="{retention_top - 18}" font-family="sans-serif" font-size="17" font-weight="600" fill="#0f172a">C. Unchanged-correct retention vs active parameters</text>',
        ]
    )
    for tick in (90.0, 95.0, 100.0):
        if tick >= retention_low:
            y = y_retention(tick)
            stroke = "#dc2626" if tick == 95.0 else "#e2e8f0"
            dash = ' stroke-dasharray="5 4"' if tick == 95.0 else ""
            parts.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - left}" y2="{y:.2f}" stroke="{stroke}"{dash}/>',
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#475569">{tick:.0f}%</text>',
                ]
            )
    retention_width = width - 2 * left
    retention_xs = [
        left + 100 + index * (retention_width - 200) / (len(points) - 1)
        for index in range(len(points))
    ]
    polyline = " ".join(
        f'{x:.2f},{y_retention(point["retention_percentage"]):.2f}'
        for x, point in zip(retention_xs, points, strict=True)
    )
    parts.append(
        f'<polyline points="{polyline}" fill="none" stroke="#94a3b8" stroke-width="2"/>'
    )
    for x, point, color in zip(retention_xs, points, colors, strict=True):
        value = point["retention_percentage"]
        lower, upper = point["retention_ci95_percentage"]
        y, y_lower, y_upper = y_retention(value), y_retention(lower), y_retention(upper)
        label = html.escape(_short_host(point["host"]))
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{y_upper:.2f}" x2="{x:.2f}" y2="{y_lower:.2f}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}"/>',
                f'<text x="{x:.2f}" y="{y - 13:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#0f172a">{value:.1f}%</text>',
                f'<text x="{x:.2f}" y="{retention_top + retention_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#334155">{label}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="700" y="865" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">Error bars are paired Wald 95% intervals for capability gain and Wilson 95% intervals for retention. The three-point fit is a matched cross-family scaling screen.</text>',
            "</svg>\n",
        ]
    )
    return "".join(parts).encode("utf-8")


def render(analysis_path: Path, output_root: Path) -> dict[str, Any]:
    if (
        not output_root.is_absolute()
        or output_root.exists()
        or output_root.is_symlink()
        or output_root.parent.is_symlink()
        or not output_root.parent.is_dir()
    ):
        raise UpwardMoEFigureError("output root differs")
    analysis, points = _load(analysis_path)
    payloads = {SVG_NAME: _svg(analysis, points), CSV_NAME: _csv(points)}
    temporary = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
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
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "analysis_sha256": sha256_file(analysis_path),
            "analysis_claim": analysis.get("claim"),
            "capability_curve_claim": analysis.get("capability_curve_claim"),
            "conservative_retention_curve_claim": analysis.get(
                "conservative_retention_curve_claim"
            ),
            "point_source_sha256s": [point.get("source_sha256") for point in points],
            "records": records,
            "scientific_scores_changed": False,
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
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = render(arguments.analysis, arguments.output_root)
    print(json.dumps({"status": result["status"], "claim": result["analysis_claim"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
