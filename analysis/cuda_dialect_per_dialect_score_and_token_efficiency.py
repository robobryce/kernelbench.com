#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cuda_dialect_summary_score_and_token_efficiency import (
    aggregate_data,
    load_result,
)
from matplotlib.lines import Line2D

DIALECTS = [
    "CUDA C++",
    "CUDA Oxide",
    "CUTE DSL",
    "Triton",
    "cuTile Python",
    "cuTile Rust",
]

PROBLEMS = [
    ("01_fp8_gemm", "FP8 GEMM"),
    ("02_kda_cutlass", "KDA CUTLASS"),
    ("03_paged_attention", "Paged Attention"),
    ("05_topk_bitonic", "TopK Bitonic"),
    ("06_sonic_moe_swiglu", "Sonic MoE SwiGLU"),
    ("07_w4a16_gemm", "W4A16 GEMM"),
]

DIALECT_COLORS = {
    "CUDA C++": "#0072B2",
    "CUDA Oxide": "#D55E00",
    "CUTE DSL": "#009E73",
    "Triton": "#CC79A7",
    "cuTile Python": "#E69F00",
    "cuTile Rust": "#6F4E7C",
}

VARIANT_TO_DIALECT_INDEX = {
    "kernelbench-hard-cuda-cpp": 0,
    "kernelbench-hard-cuda-oxide": 1,
    "kernelbench-hard-cute-dsl": 2,
    "kernelbench-hard-triton": 3,
    "kernelbench-hard-cutile-python": 4,
    "kernelbench-hard-cutile-rust": 5,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def run_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path / "waves").is_dir():
        raise argparse.ArgumentTypeError(f"not a run directory: {value}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-dialect scores and token efficiency for one or more runs."
    )
    parser.add_argument(
        "runs",
        metavar="RUN",
        nargs="+",
        type=run_path,
        help="run directory containing a waves/ subdirectory",
    )
    return parser.parse_args()


def load_run_results(root: Path) -> dict[tuple[str, int], tuple[Path, dict]]:
    results = {}
    for wave_dir in sorted((root / "waves").iterdir()):
        if not wave_dir.is_dir():
            continue

        nodes_path = wave_dir / "nodes.tsv"
        with nodes_path.open(newline="") as stream:
            nodes = {
                int(row["idx"]): row["variant"]
                for row in csv.DictReader(stream, delimiter="\t")
            }

        pattern = "exports/*/kernelbench-result/result.json"
        for path in sorted(wave_dir.glob(pattern)):
            try:
                export_index = int(path.parents[1].name)
            except ValueError as error:
                raise ValueError(f"non-numeric export directory: {path}") from error
            variant = nodes.get(export_index)
            if variant not in VARIANT_TO_DIALECT_INDEX:
                raise ValueError(f"unknown variant for export {export_index}: {nodes_path}")

            with path.open() as stream:
                result = json.load(stream)
            problem = result.get("problem")
            if not isinstance(problem, str):
                raise TypeError(f"missing problem name: {path}")
            dialect_index = VARIANT_TO_DIALECT_INDEX[variant]
            key = (problem, dialect_index)
            if key in results:
                raise ValueError(f"duplicate result for {problem}/{variant}: {path}")
            results[key] = (path, result)

    expected = {
        (problem, dialect_index)
        for problem, _ in PROBLEMS
        for dialect_index in range(len(DIALECTS))
    }
    missing = sorted(expected - results.keys())
    if missing:
        details = ", ".join(f"{problem}/{DIALECTS[index]}" for problem, index in missing)
        raise ValueError(f"missing results in {root}: {details}")
    return results


def load_point(path: Path, result: dict) -> tuple[float, float]:
    return load_result(path, result)


def render_chart(
    run_results: list[dict[tuple[str, int], tuple[Path, dict]]], theme: str
) -> Path:
    if theme == "dark":
        plt.style.use("dark_background")
        figure_color = "#000000"
        axes_color = "#000000"
        grid_color = "#9aa4b2"
        circle_edge_color = "#000000"
        diamond_edge_color = "#f3f4f6"
        missing_color = "#9aa4b2"
    else:
        plt.style.use("default")
        figure_color = "white"
        axes_color = "white"
        grid_color = "#6b7280"
        circle_edge_color = "white"
        diamond_edge_color = "#111111"
        missing_color = "#6b7280"

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
    run_offsets = (
        np.linspace(-0.12, 0.12, len(run_results))
        if len(run_results) > 1
        else np.zeros(1)
    )
    excluded_count = 0

    for row, (problem_id, problem) in enumerate(PROBLEMS):
        points_by_run = [
            [
                load_point(*results[(problem_id, index)])
                for index in range(len(DIALECTS))
            ]
            for results in run_results
        ]
        scores = np.array(
            [[point[0] for point in points] for points in points_by_run]
        )
        tokens = np.array(
            [[point[1] for point in points] for points in points_by_run]
        )
        excluded_count += int(np.isnan(scores).sum())
        efficiency, mean_scores, combined_efficiency = aggregate_data(scores, tokens)

        for col, (metric, observations, centers, ylabel) in enumerate(
            [
                ("Score", scores, mean_scores, "Score (%)"),
                (
                    "Score per token",
                    efficiency,
                    combined_efficiency,
                    "Score per 100 million tokens",
                ),
            ]
        ):
            ax = axes[row, col]
            for index, dialect in enumerate(DIALECTS):
                color = DIALECT_COLORS[dialect]
                values = observations[:, index]
                center = centers[index]
                included = np.isfinite(values)
                if not included.any():
                    ax.text(
                        index,
                        0.03,
                        "N/A",
                        transform=ax.get_xaxis_transform(),
                        color=missing_color,
                        fontsize=9,
                        ha="center",
                        va="bottom",
                    )
                    continue
                included_values = values[included]
                low, high = included_values.min(), included_values.max()

                # The bar spans all observed runs. The large diamond is the mean
                # score or combined efficiency; the circles retain each run.
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
                    index + run_offsets[included],
                    included_values,
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

            ax.set_title(f"{problem} — {metric}", fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, DIALECTS, rotation=22, ha="right")
            ax.grid(axis="y", color=grid_color, alpha=0.28, linewidth=0.8)
            ax.set_axisbelow(True)

            # Add breathing room so endpoint caps do not touch the axes.
            ymin, ymax = ax.get_ylim()
            padding = max((ymax - ymin) * 0.08, 1e-6)
            ax.set_ylim(ymin - padding, ymax + padding)

    fig.suptitle(
        f"KernelBench-Hard: {len(run_results)}-run score and token efficiency "
        f"by problem and dialect · {excluded_count} reward-hacked results excluded",
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
        ncols=6,
        frameon=False,
        title=(
            "Circles: unflagged runs   ·   Diamond: unflagged mean score / "
            "combined token efficiency   ·   Bar: unflagged range   ·   "
            "N/A: all runs reward-hacked"
        ),
        title_fontsize=11,
        fontsize=11,
        handlelength=2.4,
        columnspacing=2.0,
    )
    output = OUTPUT_DIR / (
        "cuda_dialect__kernelbench_hard__per_dialect_score_and_token_efficiency_"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        output = render_chart(run_results, theme)
        print(output)


if __name__ == "__main__":
    main()
