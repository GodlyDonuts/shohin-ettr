#!/usr/bin/env python3
"""Render the evidence overview figure for the Shohin publication report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#17212b"
MUTED = "#66737f"
GRID = "#d9e0e5"
QWEN = "#2f6fed"
OTHER_DENSE = "#7b61a8"
MIXTRAL = "#e17832"
COMMIT = "#1d9a6c"
CONTROL = "#aab4bd"


def percentage(correct: int, total: int) -> float:
    return 100.0 * correct / total


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
    for path in render(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
