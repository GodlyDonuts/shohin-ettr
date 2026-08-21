#!/usr/bin/env python3
"""Render the evidence overview figure for the Shohin publication report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#17212b"
MUTED = "#66737f"
GRID = "#d9e0e5"
QWEN = "#2f6fed"
OTHER_DENSE = "#7b61a8"
MIXTRAL = "#e17832"
COMMIT = "#1d9a6c"
CONTROL = "#aab4bd"
FROZEN = "#eef1f4"


def percentage(correct: int, total: int) -> float:
    return 100.0 * correct / total


def diagram_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 8.2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
    )


def diagram_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def dense_panel(ax: plt.Axes) -> None:
    labels = [
        "Qwen\n0.8B (H)",
        "Qwen\n4B (H)",
        "Qwen\n9B (H)",
        "SmolLM\n3B (D)",
        "OLMo\n7B (D)",
    ]
    gains = [
        percentage(328 - 242, 1279),
        percentage(554 - 380, 1279),
        percentage(625 - 495, 1279),
        percentage(469 - 358, 1289),
        percentage(259 - 231, 1289),
    ]
    colors = [QWEN, QWEN, QWEN, OTHER_DENSE, OTHER_DENSE]
    bars = ax.bar(np.arange(len(labels)), gains, color=colors, width=0.72)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Revision gain (percentage points)")
    ax.set_title(
        "a  Dense transfer repeats across sizes and families", loc="left", weight="bold"
    )
    ax.set_ylim(0, 16)
    for bar, value in zip(bars, gains, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"+{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )
    ax.text(
        0.99,
        0.96,
        "H = holdout; D = development",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=8,
    )


def moe_panel(ax: plt.Axes) -> None:
    experiments = [
        (
            "Qwen 35B\nscreen",
            256,
            [("unchanged", 111, CONTROL), ("causal gate", 143, QWEN)],
        ),
        (
            "Mixtral 141B\nscreen",
            256,
            [
                ("unchanged", 45, CONTROL),
                ("self-refine", 105, "#d3a44a"),
                ("revision", 114, MIXTRAL),
            ],
        ),
        (
            "Mixtral 141B\nvalidation",
            1023,
            [
                ("unchanged", 147, CONTROL),
                ("self-refine", 356, "#d3a44a"),
                ("revision", 448, MIXTRAL),
                ("commit", 287, COMMIT),
            ],
        ),
    ]
    group_centers = np.arange(len(experiments)) * 1.35
    width = 0.22
    for center, (_, total, arms) in zip(group_centers, experiments, strict=True):
        offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * width
        for offset, (arm, correct, color) in zip(offsets, arms, strict=True):
            value = percentage(correct, total)
            ax.bar(center + offset, value, width=width * 0.9, color=color, label=arm)
            ax.text(
                center + offset,
                value + 1.0,
                str(correct),
                ha="center",
                va="bottom",
                fontsize=7.7,
                weight="bold" if arm in {"causal gate", "revision"} else "normal",
            )
    ax.set_xticks(group_centers, [item[0] for item in experiments])
    ax.set_ylabel("Accuracy (%) — labels show correct count")
    ax.set_title("b  Matched MoE capability", loc="left", weight="bold")
    ax.set_ylim(0, 72)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper right",
    )


def retention_panel(ax: plt.Axes) -> None:
    points = [
        (
            "Qwen causal gate\n256 rows",
            percentage(105, 111),
            percentage(32, 256),
            QWEN,
            "o",
            (-1.0, -2.0),
            "right",
        ),
        (
            "Mixtral revision\n256 rows",
            percentage(34, 45),
            percentage(69, 256),
            MIXTRAL,
            "o",
            (1.0, 1.2),
            "left",
        ),
        (
            "Mixtral revision\n1,023 rows",
            percentage(95, 147),
            percentage(301, 1023),
            MIXTRAL,
            "s",
            (1.0, 1.2),
            "left",
        ),
        (
            "Mixtral commit\n1,023 rows",
            percentage(137, 147),
            percentage(140, 1023),
            COMMIT,
            "D",
            (-1.0, 1.2),
            "right",
        ),
    ]
    ax.axvspan(95, 100, color="#e3f4ed", alpha=0.9, zorder=0)
    ax.axvline(95, color=COMMIT, linewidth=1.1, linestyle="--")
    for label, retention, gain, color, marker, offset, alignment in points:
        ax.scatter(
            retention,
            gain,
            s=80,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{label}\n+{gain:.2f} pp / {retention:.1f}% retained",
            (retention, gain),
            xytext=(retention + offset[0], gain + offset[1]),
            textcoords="data",
            ha=alignment,
            va="bottom",
            fontsize=7.8,
            color=INK,
        )
    ax.set_xlim(58, 101)
    ax.set_ylim(8, 34)
    ax.set_xlabel("Unchanged-correct cases retained (%)")
    ax.set_ylabel("Gain over unchanged (percentage points)")
    ax.set_title(
        "c  Capability and conservative retention separate", loc="left", weight="bold"
    )
    ax.text(97.5, 9.0, "95% retention zone", ha="center", color=COMMIT, fontsize=8)


def lifecycle_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("a  Dense model-owned temporal lifecycle", loc="left", weight="bold")
    diagram_box(ax, 0.2, 2.45, 1.35, 0.9, "source\nx", facecolor=FROZEN)
    diagram_box(
        ax,
        2.05,
        2.35,
        1.9,
        1.1,
        "draft owner\nθ + δdraft",
        facecolor="#dfe9ff",
        edgecolor=QWEN,
        fontsize=7.2,
    )
    diagram_box(
        ax, 4.45, 2.45, 1.2, 0.9, "draft\nd", facecolor="#dfe9ff", edgecolor=QWEN
    )
    diagram_box(
        ax,
        6.35,
        3.65,
        2.0,
        1.0,
        "revision owner\nθ + δrevision",
        facecolor="#ffe7d6",
        edgecolor=MIXTRAL,
        fontsize=6.7,
    )
    diagram_box(
        ax, 8.8, 3.7, 1.1, 0.9, "revised\nr", facecolor="#ffe7d6", edgecolor=MIXTRAL
    )
    diagram_box(
        ax,
        6.35,
        1.25,
        2.0,
        1.0,
        "unchanged role\nmatched budget",
        facecolor=FROZEN,
        fontsize=6.7,
    )
    diagram_box(ax, 8.8, 1.3, 1.1, 0.9, "unchanged\nu", facecolor=FROZEN)
    diagram_box(
        ax,
        10.35,
        2.35,
        1.75,
        1.1,
        "commit owner\nchoose one whole\ntrajectory",
        facecolor="#def3e9",
        edgecolor=COMMIT,
        fontsize=6.4,
    )
    diagram_box(ax, 12.55, 2.45, 0.65, 0.9, "y", facecolor="#def3e9", edgecolor=COMMIT)
    diagram_arrow(ax, (1.55, 2.9), (2.05, 2.9))
    diagram_arrow(ax, (3.95, 2.9), (4.45, 2.9))
    diagram_arrow(ax, (5.65, 3.0), (6.35, 4.05), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (5.65, 2.8), (6.35, 1.85), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (8.35, 4.15), (8.8, 4.15))
    diagram_arrow(ax, (8.35, 1.75), (8.8, 1.75))
    diagram_arrow(ax, (9.9, 4.05), (10.35, 3.15), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (9.9, 1.75), (10.35, 2.65), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (12.1, 2.9), (12.55, 2.9), color=COMMIT)
    ax.text(
        6.9,
        5.25,
        "same source + draft\nsame output budget",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
    )
    ax.text(
        6.7,
        0.35,
        "Inference sees no correctness label, verifier, benchmark route, or tool result",
        ha="center",
        fontsize=7.7,
        color=MUTED,
    )


def residual_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("b  Sparse-host revision surface", loc="left", weight="bold")
    diagram_box(ax, 0.2, 2.45, 1.2, 0.9, "hidden\nh", facecolor=FROZEN)
    diagram_box(
        ax,
        2.15,
        3.55,
        2.45,
        1.05,
        "frozen MoE block\nmₗ(h)\nrouter + experts",
        facecolor=FROZEN,
    )
    diagram_box(
        ax,
        2.15,
        1.05,
        2.45,
        1.05,
        "trained low-rank path\n(α/q) BₗAₗh",
        facecolor="#ffe7d6",
        edgecolor=MIXTRAL,
    )
    diagram_box(ax, 5.55, 2.4, 1.1, 1.0, "+", facecolor="white")
    diagram_box(
        ax,
        7.55,
        2.45,
        1.9,
        0.9,
        "revised block\nm'ₗ(h)",
        facecolor="#ffe7d6",
        edgecolor=MIXTRAL,
    )
    diagram_arrow(ax, (1.4, 2.9), (2.15, 4.05), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (1.4, 2.8), (2.15, 1.55), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (4.6, 4.05), (5.55, 3.05), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (4.6, 1.55), (5.55, 2.75), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (6.65, 2.9), (7.55, 2.9), color=MIXTRAL)
    ax.text(
        5.0,
        5.15,
        "final 16 layers; native router/expert trainables = 0",
        ha="center",
        fontsize=7.8,
        color=MUTED,
    )
    ax.text(
        5.0,
        0.35,
        "Mixtral: rank 18, α=18, 3,538,944 trained parameters",
        ha="center",
        fontsize=7.8,
        color=MIXTRAL,
    )


def causal_gate_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("c  Tokenwise causal ownership", loc="left", weight="bold")
    diagram_box(ax, 0.2, 2.45, 1.2, 0.9, "hidden\nh", facecolor=FROZEN)
    diagram_box(
        ax,
        2.05,
        4.0,
        2.1,
        0.9,
        "frozen owner\nΔo,ₗ(h)",
        facecolor="#dfe9ff",
        edgecolor=QWEN,
    )
    diagram_box(
        ax,
        2.05,
        2.45,
        2.1,
        0.9,
        "learned gate\ngₗ(h)=σ(wₗh+bₗ)",
        facecolor="#def3e9",
        edgecolor=COMMIT,
        fontsize=7.7,
    )
    diagram_box(
        ax,
        2.05,
        0.9,
        2.1,
        0.9,
        "frozen revision\nΔr,ₗ(h)",
        facecolor="#ffe7d6",
        edgecolor=MIXTRAL,
    )
    diagram_box(
        ax,
        5.25,
        2.35,
        2.4,
        1.1,
        "Δo,ₗ + gₗ(Δr,ₗ−Δo,ₗ)",
        facecolor="#def3e9",
        edgecolor=COMMIT,
        fontsize=7.5,
    )
    diagram_box(ax, 8.45, 2.45, 1.25, 0.9, "add to\nmₗ(h)", facecolor=FROZEN)
    diagram_arrow(ax, (1.4, 3.0), (2.05, 4.4), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (1.4, 2.9), (2.05, 2.9))
    diagram_arrow(ax, (1.4, 2.8), (2.05, 1.35), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (4.15, 4.45), (5.25, 3.2), connectionstyle="arc3,rad=0.08")
    diagram_arrow(ax, (4.15, 2.9), (5.25, 2.9), color=COMMIT)
    diagram_arrow(ax, (4.15, 1.35), (5.25, 2.6), connectionstyle="arc3,rad=-0.08")
    diagram_arrow(ax, (7.65, 2.9), (8.45, 2.9), color=COMMIT)
    ax.text(
        5.0,
        5.25,
        "Only wₗ and bₗ train; both trajectory residuals stay frozen",
        ha="center",
        fontsize=7.8,
        color=MUTED,
    )
    ax.text(
        5.0,
        0.25,
        "Qwen3.6-35B-A3B: 32,784 gate parameters; response loss only",
        ha="center",
        fontsize=7.8,
        color=QWEN,
    )


def render_architecture(output_dir: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.4, 4.3),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.1]},
    )
    lifecycle_panel(axes[0])
    residual_panel(axes[1])
    causal_gate_panel(axes[2])
    fig.suptitle(
        "Shohin assigns learned owners across inference time while freezing the host backbone",
        x=0.01,
        ha="left",
        fontsize=15,
        weight="bold",
        color=INK,
    )
    svg = output_dir / "shohin_temporal_revision_architecture.svg"
    pdf = output_dir / "shohin_temporal_revision_architecture.pdf"
    fig.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Creator": "Shohin", "Date": "2026-08-18"},
    )
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fixed_time = datetime(2026, 8, 18, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "Shohin",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(fig)
    return svg, pdf


def render(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "shohin-temporal-revision-evidence-v1",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), constrained_layout=True)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    dense_panel(axes[0])
    moe_panel(axes[1])
    retention_panel(axes[2])
    fig.suptitle(
        "Shohin: temporal revision transfers; commitment determines whether capability is retained",
        x=0.01,
        ha="left",
        fontsize=15,
        weight="bold",
        color=INK,
    )
    svg = output_dir / "shohin_temporal_revision_evidence.svg"
    pdf = output_dir / "shohin_temporal_revision_evidence.pdf"
    fig.savefig(
        svg,
        bbox_inches="tight",
        metadata={"Creator": "Shohin", "Date": "2026-08-18"},
    )
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fixed_time = datetime(2026, 8, 18, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "Shohin",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    plt.close(fig)
    return svg, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = (*render(args.output_dir), *render_architecture(args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
