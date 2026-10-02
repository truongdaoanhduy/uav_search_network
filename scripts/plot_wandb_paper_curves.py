#!/usr/bin/env python3
"""Export W&B episode metrics as faint-raw + bold-EMA publication figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

DEFAULT_METRICS = {
    "reward": (
        "episode/return_mean",
        "Reward Value",
        None,
    ),
    "coverage": (
        "episode/coverage_percent_mean",
        "Coverage (%)",
        (0.0, 100.0),
    ),
    "search": (
        "episode/target_search_rate_percent",
        "Target Search Rate (%)",
        (0.0, 100.0),
    ),
    "delivery": (
        "episode/target_delivery_rate_percent",
        "Target Delivery Rate (%)",
        (0.0, 100.0),
    ),
}


def ema(values, beta=0.95):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    if values.size == 0:
        return values.copy()
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = (
            beta * output[index - 1]
            + (1.0 - beta) * values[index]
        )
    return output


def fetch(run_path, metric, x_key):
    run = wandb.Api().run(run_path)
    rows = list(
        run.scan_history(
            keys=[
                x_key,
                metric,
            ]
        )
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame[
            [
                x_key,
                metric,
            ]
        ]
        .dropna()
        .sort_values(x_key)
        .drop_duplicates(
            subset=[x_key],
            keep="last",
        )
        .reset_index(drop=True)
    )


def plot_grouped(
    groups,
    metric,
    ylabel,
    output_path,
    *,
    x_key="episodes_completed",
    beta=0.95,
    y_limits=None,
    raw_alpha=0.12,
    raw_linewidth=0.9,
    main_linewidth=2.6,
    band_alpha=0.16,
):
    fig, ax = plt.subplots(
        figsize=(6.4, 4.4)
    )
    colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    for group_index, (
        label,
        run_paths,
    ) in enumerate(groups.items()):
        color = colors[
            group_index % len(colors)
        ]
        clean_runs = []

        for run_path in run_paths:
            frame = fetch(
                run_path,
                metric,
                x_key,
            )
            if frame.empty:
                continue
            x = frame[x_key].to_numpy(
                dtype=np.float64
            )
            y = frame[metric].to_numpy(
                dtype=np.float64
            )
            ax.plot(
                x,
                y,
                color=color,
                alpha=raw_alpha,
                linewidth=raw_linewidth,
                zorder=1,
            )
            clean_runs.append((x, y))

        if not clean_runs:
            continue

        if len(clean_runs) == 1:
            x_main, y_raw = clean_runs[0]
            y_main = ema(
                y_raw,
                beta=beta,
            )
        else:
            x_min = max(
                x[0]
                for x, _ in clean_runs
            )
            x_max = min(
                x[-1]
                for x, _ in clean_runs
            )
            points = max(
                64,
                min(
                    1200,
                    max(
                        len(x)
                        for x, _ in clean_runs
                    ),
                ),
            )
            x_main = np.linspace(
                x_min,
                x_max,
                points,
            )
            values = np.stack(
                [
                    np.interp(
                        x_main,
                        x,
                        y,
                    )
                    for x, y in clean_runs
                ]
            )
            mean = np.mean(
                values,
                axis=0,
            )
            sem = np.std(
                values,
                axis=0,
                ddof=1,
            ) / math.sqrt(
                len(clean_runs)
            )
            y_main = ema(
                mean,
                beta=beta,
            )
            ax.fill_between(
                x_main,
                ema(
                    mean - sem,
                    beta=beta,
                ),
                ema(
                    mean + sem,
                    beta=beta,
                ),
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=2,
            )

        ax.plot(
            x_main,
            y_main,
            color=color,
            linewidth=main_linewidth,
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.7,
    )
    ax.legend(frameon=True)
    ax.ticklabel_format(
        style="sci",
        axis="x",
        scilimits=(4, 4),
        useMathText=True,
    )
    fig.tight_layout()

    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_runs(items):
    groups = {}
    for item in items:
        label, sep, run_path = item.partition("=")
        if not sep:
            raise ValueError(
                "--run must be LABEL=entity/project/run_id"
            )
        groups.setdefault(
            label,
            [],
        ).append(run_path)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "LABEL=entity/project/run_id; repeat for algorithms "
            "or multiple independent runs"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=sorted(DEFAULT_METRICS),
        default="reward",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures",
    )
    args = parser.parse_args()

    groups = parse_runs(args.run)
    metric, ylabel, y_limits = (
        DEFAULT_METRICS[
            args.metric
        ]
    )
    output = (
        Path(args.output_dir)
        / f"{args.metric}_learning_curve.png"
    )
    plot_grouped(
        groups,
        metric,
        ylabel,
        output,
        beta=args.beta,
        y_limits=y_limits,
    )
    print(output)


if __name__ == "__main__":
    main()
