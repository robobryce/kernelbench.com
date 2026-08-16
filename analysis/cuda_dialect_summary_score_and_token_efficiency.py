#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

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
    ("06_sonic_moe_swiglu", "Sonic MoE\nSwiGLU"),
    ("07_w4a16_gemm", "W4A16 GEMM"),
]

COLORS = {
    "CUDA C++": "#0072B2",
    "CUDA Oxide": "#D55E00",
    "CUTE DSL": "#009E73",
    "Triton": "#CC79A7",
    "cuTile Python": "#E69F00",
    "cuTile Rust": "#6F4E7C",
}

MODEL_DISPLAY_NAMES = {
    "azure/openai/gpt-5.6-sol": "GPT 5.6 Sol",
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


def lighter_color(color: str, amount: float = 0.22) -> tuple[float, float, float]:
    return tuple(channel + (1.0 - channel) * amount for channel in to_rgb(color))


def run_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path / "waves").is_dir():
        raise argparse.ArgumentTypeError(f"not a run directory: {value}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot summary scores and token efficiency for one or more runs."
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


def load_result(path: Path, result: dict) -> tuple[float, float]:
    reward_hacked = result.get("reward_hacked", False)
    if not isinstance(reward_hacked, bool):
        raise TypeError(f"invalid reward_hacked flag: {path}")
    if reward_hacked:
        return np.nan, np.nan

    if result.get("correct") is False:
        peak_fraction = 0.0
    elif result.get("correct") is True:
        peak_fraction = result.get("peak_fraction")
        if (
            not isinstance(peak_fraction, (int, float))
            or not np.isfinite(peak_fraction)
            or peak_fraction <= 0
        ):
            raise ValueError(f"missing corrected score: {path}")
    else:
        raise ValueError(f"missing correctness grade: {path}")

    usage = result.get("usage") or {}
    tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    if not np.isfinite(tokens) or tokens <= 0:
        raise ValueError(f"missing token usage: {path}")
    return 100.0 * peak_fraction, tokens / 1e8


def aggregate_data(
    scores: np.ndarray, tokens: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.shape != tokens.shape:
        raise ValueError("score and token arrays must have the same shape")

    excluded = np.isnan(scores) & np.isnan(tokens)
    included = np.isfinite(scores) & np.isfinite(tokens)
    if not np.all(excluded | included):
        raise ValueError("scores and token counts must be finite or excluded together")
    if np.any(scores[included] < 0):
        raise ValueError("scores must be non-negative")
    if np.any(tokens[included] <= 0):
        raise ValueError("included token counts must be positive")

    score_per_token = np.full_like(scores, np.nan, dtype=float)
    np.divide(scores / 100.0, tokens, out=score_per_token, where=included)

    included_counts = included.sum(axis=0)
    score_sums = np.nansum(scores, axis=0)
    mean_scores = np.full(scores.shape[1:], np.nan, dtype=float)
    np.divide(
        score_sums,
        included_counts,
        out=mean_scores,
        where=included_counts > 0,
    )

    token_sums = np.nansum(tokens, axis=0)
    combined_score_per_token = np.full(scores.shape[1:], np.nan, dtype=float)
    np.divide(
        score_sums / 100.0,
        token_sums,
        out=combined_score_per_token,
        where=token_sums > 0,
    )
    return score_per_token, mean_scores, combined_score_per_token


def load_data(runs: list[Path]) -> tuple[np.ndarray, np.ndarray, str, str]:
    scores = np.zeros((len(runs), len(DIALECTS), len(PROBLEMS)))
    tokens = np.zeros_like(scores)
    models: set[str] = set()
    harnesses: set[str] = set()
    for run_index, root in enumerate(runs):
        results = load_run_results(root)
        for problem_index, (problem, _) in enumerate(PROBLEMS):
            for dialect_index in range(len(DIALECTS)):
                path, metadata = results[(problem, dialect_index)]
                models.add(metadata.get("model") or "unknown")
                harnesses.add(metadata.get("harness") or "unknown")
                scores[run_index, dialect_index, problem_index], tokens[
                    run_index, dialect_index, problem_index
                ] = load_result(path, metadata)
    if len(models) != 1 or len(harnesses) != 1:
        raise ValueError(f"mixed model/harness data: models={models}, harnesses={harnesses}")
    return scores, tokens, models.pop(), harnesses.pop()


def draw_bars(
    ax: plt.Axes,
    values: np.ndarray,
    centers: np.ndarray,
    ylabel: str,
    title: str,
    missing_color: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(PROBLEMS))
    group_width = 0.84
    bar_width = group_width / len(DIALECTS)
    for dialect_index, dialect in enumerate(DIALECTS):
        observations = values[:, dialect_index, :]
        average = centers[dialect_index]
        low = np.full(len(PROBLEMS), np.nan)
        high = np.full(len(PROBLEMS), np.nan)
        for problem_index in range(len(PROBLEMS)):
            included = observations[:, problem_index]
            included = included[np.isfinite(included)]
            if included.size:
                low[problem_index] = included.min()
                high[problem_index] = included.max()
        positions = x - group_width / 2 + bar_width * (dialect_index + 0.5)
        included = np.isfinite(average) & np.isfinite(low) & np.isfinite(high)
        for position in positions[~included]:
            ax.text(
                position,
                0.02,
                "N/A",
                transform=ax.get_xaxis_transform(),
                color=missing_color,
                fontsize=7,
                rotation=90,
                ha="center",
                va="bottom",
            )
        ax.bar(
            positions[included],
            low[included],
            width=bar_width * 0.92,
            color=COLORS[dialect],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            label=dialect,
        )
        ax.bar(
            positions[included],
            np.maximum(high[included] - low[included], 0),
            bottom=low[included],
            width=bar_width * 0.92,
            color=lighter_color(COLORS[dialect]),
            edgecolor="none",
            linewidth=0,
            antialiased=False,
        )
        average_half_width = bar_width * 0.34
        ax.hlines(
            average[included],
            positions[included] - average_half_width,
            positions[included] + average_half_width,
            color="#000000",
            linewidth=2.2,
            zorder=4,
        )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [label for _, label in PROBLEMS])
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(bottom=0)


def render_chart(
    scores: np.ndarray,
    tokens: np.ndarray,
    model: str,
    harness: str,
    run_count: int,
    theme: str,
) -> Path:
    score_per_token, average_scores, combined_score_per_token = aggregate_data(
        scores, tokens
    )
    excluded_count = int(np.isnan(scores).sum())

    if theme == "dark":
        plt.style.use("dark_background")
        figure_color = "#000000"
        axes_color = "#000000"
        grid_color = "#9aa4b2"
        missing_color = "#d1d5db"
    else:
        plt.style.use("default")
        figure_color = "white"
        axes_color = "white"
        grid_color = "#6b7280"
        missing_color = "#222222"

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
        scores,
        average_scores,
        "Score (%)",
        "KernelBench-Hard score by problem",
        missing_color,
        (0, 50),
    )
    draw_bars(
        axes[1],
        score_per_token,
        combined_score_per_token,
        "Score / tokens (score / 100M tokens)",
        "KernelBench-Hard token efficiency by problem",
        missing_color,
    )
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].set_xlabel("Problem", labelpad=5)
    axes[1].set_xlabel("Problem")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=6,
        frameon=False,
        title=(
            "Base stack: unflagged minimum · Lighter stack: range to maximum · "
            "Line: mean score / combined token efficiency · "
            "N/A: all runs reward-hacked"
        ),
    )
    fig.suptitle(
        "KernelBench-Hard: score and token efficiency by dialect\n"
        f"Runs: {run_count}  ·  Reward-hacked results excluded: {excluded_count}  ·  "
        f"Model: {MODEL_DISPLAY_NAMES.get(model, model)}"
        f"  ·  Harness: {harness}",
        fontsize=18,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.48)

    for ax in axes:
        for line in ax.get_ygridlines():
            line.set_color(grid_color)

    output = OUTPUT_DIR / (
        "cuda_dialect__kernelbench_hard__summary_score_and_token_efficiency_"
        f"{theme}__{datetime.now(timezone.utc):%Y_%m_%d}.png"
    )
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure_color)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    scores, tokens, model, harness = load_data(args.runs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        output = render_chart(scores, tokens, model, harness, len(args.runs), theme)
        print(output)


if __name__ == "__main__":
    main()
