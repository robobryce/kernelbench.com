#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cuda_dialect_summary_score_and_token_efficiency import (
    DIALECTS,
    MODEL_DISPLAY_NAMES,
    OUTPUT_DIR,
    draw_bars,
    load_data,
    run_path,
)
from matplotlib.ticker import LogLocator, StrMethodFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot scores and token efficiency normalized to CUDA C++ for each problem."
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


def normalize_data(
    scores: np.ndarray, tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(tokens)) or np.any(tokens <= 0):
        raise ValueError("token counts must be finite and positive")

    score_per_token = (scores / 100.0) / tokens
    mean_scores = scores.mean(axis=0)
    combined_efficiency = (scores.sum(axis=0) / 100.0) / tokens.sum(axis=0)

    cuda_score_baseline = mean_scores[0]
    cuda_efficiency_baseline = combined_efficiency[0]
    if np.any(cuda_score_baseline <= 0):
        raise ValueError("CUDA C++ mean score must be positive for every problem")
    if np.any(cuda_efficiency_baseline <= 0):
        raise ValueError("CUDA C++ token efficiency must be positive for every problem")

    # Normalize observations to the aggregate CUDA C++ baseline. This keeps
    # individual-run variation visible while fixing every CUDA C++ center at 1.0.
    normalized_scores = scores / cuda_score_baseline[np.newaxis, np.newaxis, :]
    normalized_mean_scores = mean_scores / cuda_score_baseline[np.newaxis, :]
    normalized_efficiency = score_per_token / cuda_efficiency_baseline[
        np.newaxis, np.newaxis, :
    ]
    normalized_combined_efficiency = (
        combined_efficiency / cuda_efficiency_baseline[np.newaxis, :]
    )
    return (
        normalized_scores,
        normalized_mean_scores,
        normalized_efficiency,
        normalized_combined_efficiency,
    )


def render_chart(
    normalized_scores: np.ndarray,
    normalized_mean_scores: np.ndarray,
    normalized_efficiency: np.ndarray,
    normalized_combined_efficiency: np.ndarray,
    model: str,
    harness: str,
    run_count: int,
    theme: str,
) -> Path:
    if theme == "dark":
        plt.style.use("dark_background")
        figure_color = "#000000"
        axes_color = "#000000"
        grid_color = "#9aa4b2"
        edge_color = "#000000"
        error_color = "#d1d5db"
        reference_color = "#f3f4f6"
    else:
        plt.style.use("default")
        figure_color = "white"
        axes_color = "white"
        grid_color = "#6b7280"
        edge_color = "white"
        error_color = "#222222"
        reference_color = "#111827"

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    fig.patch.set_facecolor(figure_color)
    for ax in axes:
        ax.set_facecolor(axes_color)

    draw_bars(
        axes[0],
        normalized_scores,
        normalized_mean_scores,
        "Relative score (CUDA C++ = 1.0)",
        "KernelBench-Hard normalized score by problem",
        error_color,
    )
    draw_bars(
        axes[1],
        normalized_efficiency,
        normalized_combined_efficiency,
        "Relative score / token (CUDA C++ = 1.0)",
        "KernelBench-Hard normalized token efficiency by problem",
        error_color,
    )
    for ax in axes:
        ax.set_yscale("log")
        ax.autoscale(enable=True, axis="y")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 6)))
        ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:g}"))
        ax.axhline(
            1.0,
            color=reference_color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            zorder=0,
        )
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_xlabel("Problem", labelpad=5)
    axes[1].set_xlabel("Problem")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=len(DIALECTS),
        frameon=False,
        title=(
            "Bar: normalized mean score / combined token efficiency · "
            "Error bar: observed range · Dashed line: CUDA C++ baseline"
        ),
    )
    fig.suptitle(
        "KernelBench-Hard: score and token efficiency relative to CUDA C++\n"
        f"Runs: {run_count}  ·  Model: {MODEL_DISPLAY_NAMES.get(model, model)}"
        f"  ·  Harness: {harness}",
        fontsize=18,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.48)

    for ax in axes:
        for patch in ax.patches:
            patch.set_edgecolor(edge_color)
        for line in ax.get_ygridlines():
            line.set_color(grid_color)

    output = OUTPUT_DIR / (
        "cuda_dialect__kernelbench_hard__summary_normalized_score_and_token_efficiency_"
        f"{theme}__{datetime.now(timezone.utc):%Y_%m_%d}.png"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure_color)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    scores, tokens, model, harness = load_data(args.runs)
    normalized = normalize_data(scores, tokens)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        output = render_chart(
            *normalized,
            model,
            harness,
            len(args.runs),
            theme,
        )
        print(output)


if __name__ == "__main__":
    main()
