#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cuda_dialect_per_dialect_score_and_token_efficiency import (
    DIALECT_COLORS,
    DIALECTS,
    OUTPUT_DIR,
    PROBLEMS,
    load_point,
    load_run_results,
    run_path,
)
from cuda_dialect_summary_normalized_score_and_token_efficiency import normalize_data
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-dialect scores and token efficiency normalized to CUDA C++."
        )
    )
    parser.add_argument(
        "runs",
        metavar="RUN",
        nargs="+",
        type=run_path,
        help="run directory containing a waves/ subdirectory",
    )
    return parser.parse_args()


def load_data(
    run_results: list[dict[tuple[str, int], tuple[Path, dict]]],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((len(run_results), len(DIALECTS), len(PROBLEMS)))
    tokens = np.zeros_like(scores)
    for run_index, results in enumerate(run_results):
        for problem_index, (problem, _) in enumerate(PROBLEMS):
            for dialect_index in range(len(DIALECTS)):
                scores[run_index, dialect_index, problem_index], tokens[
                    run_index, dialect_index, problem_index
                ] = load_point(*results[(problem, dialect_index)])
    return scores, tokens


def render_chart(
    normalized_scores: np.ndarray,
    normalized_mean_scores: np.ndarray,
    normalized_efficiency: np.ndarray,
    normalized_combined_efficiency: np.ndarray,
    theme: str,
) -> Path:
    if theme == "dark":
        plt.style.use("dark_background")
        figure_color = "#000000"
        axes_color = "#000000"
        grid_color = "#9aa4b2"
        circle_edge_color = "#000000"
        diamond_edge_color = "#f3f4f6"
        reference_color = "#f3f4f6"
    else:
        plt.style.use("default")
        figure_color = "white"
        axes_color = "white"
        grid_color = "#6b7280"
        circle_edge_color = "white"
        diamond_edge_color = "#111111"
        reference_color = "#111827"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(6, 2, figsize=(20, 27), constrained_layout=True)
    fig.patch.set_facecolor(figure_color)
    for ax in axes.flat:
        ax.set_facecolor(axes_color)

    x = np.arange(len(DIALECTS))
    run_count = normalized_scores.shape[0]
    run_offsets = (
        np.linspace(-0.12, 0.12, run_count) if run_count > 1 else np.zeros(1)
    )

    for row, (_, problem) in enumerate(PROBLEMS):
        for col, (metric, observations, centers, ylabel) in enumerate(
            [
                (
                    "Normalized score",
                    normalized_scores[:, :, row],
                    normalized_mean_scores[:, row],
                    "Relative score (CUDA C++ = 1.0)",
                ),
                (
                    "Normalized score per token",
                    normalized_efficiency[:, :, row],
                    normalized_combined_efficiency[:, row],
                    "Relative score / token (CUDA C++ = 1.0)",
                ),
            ]
        ):
            ax = axes[row, col]
            for index, dialect in enumerate(DIALECTS):
                color = DIALECT_COLORS[dialect]
                values = observations[:, index]
                center = centers[index]
                low, high = values.min(), values.max()

                ax.errorbar(
                    index,
                    center,
                    yerr=[[center - low], [high - center]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=3.0,
                    capsize=8,
                    capthick=2.4,
                    alpha=0.8,
                    zorder=1,
                )
                ax.scatter(
                    index + run_offsets,
                    values,
                    s=72,
                    marker="o",
                    color=color,
                    edgecolor=circle_edge_color,
                    linewidth=1.1,
                    alpha=0.7,
                    zorder=2,
                )
                ax.scatter(
                    index,
                    center,
                    s=125,
                    marker="D",
                    color=color,
                    edgecolor=diamond_edge_color,
                    linewidth=1.2,
                    zorder=3,
                )

            ax.axhline(
                1.0,
                color=reference_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=0,
            )
            ax.set_title(f"{problem} — {metric}", fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, DIALECTS, rotation=22, ha="right")
            ax.grid(axis="y", color=grid_color, alpha=0.28, linewidth=0.8)
            ax.set_axisbelow(True)

            ymin, ymax = ax.get_ylim()
            padding = max((ymax - ymin) * 0.08, 1e-6)
            ax.set_ylim(ymin - padding, ymax + padding)

    fig.suptitle(
        f"KernelBench-Hard: {run_count}-run score and token efficiency "
        "relative to CUDA C++",
        fontsize=19,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            linewidth=3,
            markersize=9,
            color=DIALECT_COLORS[dialect],
            markeredgecolor=diamond_edge_color,
            label=dialect,
        )
        for dialect in DIALECTS
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=len(DIALECTS),
        frameon=False,
        title=(
            "Circles: individual runs   ·   Diamond: normalized mean score / "
            "combined token efficiency   ·   Bar: observed range   ·   "
            "Dashed line: CUDA C++ baseline"
        ),
        title_fontsize=11,
        fontsize=11,
        handlelength=2.4,
        columnspacing=2.0,
    )

    output = OUTPUT_DIR / (
        "cuda_dialect__kernelbench_hard__per_dialect_normalized_score_and_token_efficiency_"
        f"{theme}__{datetime.now(timezone.utc):%Y_%m_%d}.png"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure_color)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    run_results = [load_run_results(root) for root in args.runs]
    models = {
        result.get("model") or "unknown"
        for results in run_results
        for _, result in results.values()
    }
    harnesses = {
        result.get("harness") or "unknown"
        for results in run_results
        for _, result in results.values()
    }
    if len(models) != 1 or len(harnesses) != 1:
        raise ValueError(f"mixed model/harness data: models={models}, harnesses={harnesses}")

    scores, tokens = load_data(run_results)
    normalized = normalize_data(scores, tokens)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        output = render_chart(*normalized, theme)
        print(output)


if __name__ == "__main__":
    main()
