#!/usr/bin/env python3
"""Format the main W&B comparison section like a publication RL learning curve."""

from __future__ import annotations

import argparse

from wandb_workspaces import workspaces as ws
from wandb_workspaces.reports import v2 as wr

DEFAULT_VIEW = (
    "https://wandb.ai/uav_search_paper/uav_search_target"
    "?nw=wpu3r5w2yti"
)


def _line(
    title,
    metric,
    ylabel,
    *,
    y_range=(None, None),
):
    return wr.LinePlot(
        title=title,
        x=wr.Metric(
            "episodes_completed"
        ),
        y=[
            wr.Metric(metric)
        ],
        title_x="Episode",
        title_y=ylabel,
        range_y=y_range,
        smoothing_type="exponential",
        smoothing_factor=0.95,
        smoothing_show_original=True,
        legend_position="south",
    )


def format_workspace(
    view_url=DEFAULT_VIEW,
):
    workspace = ws.Workspace.from_url(
        view_url
    )

    comparison = ws.Section(
        name="01 — Algorithm Comparison",
        is_open=True,
        pinned=True,
        panels=[
            _line(
                "Total Reward (Episode Return)",
                "episode/return_mean",
                "Episode return",
            ),
            _line(
                "Mission Success Rate",
                "episode/success_rate",
                "Success rate",
                y_range=(0, 1),
            ),
            _line(
                "Target Search Rate",
                "episode/target_search_rate_percent",
                "Search rate (%)",
                y_range=(0, 100),
            ),
            _line(
                "Target Delivery Rate",
                "episode/target_delivery_rate_percent",
                "Delivery rate (%)",
                y_range=(0, 100),
            ),
            _line(
                "Area Coverage",
                "episode/coverage_percent_mean",
                "Coverage (%)",
                y_range=(0, 100),
            ),
            _line(
                "Report Delivery Efficiency",
                "episode/report_delivery_given_confirmation_rate_percent",
                "Delivered / confirmed (%)",
                y_range=(0, 100),
            ),
            _line(
                "Episode End Step",
                "episode/end_step",
                "Step",
            ),
            _line(
                "Total Energy",
                "episode/total_energy_j_mean",
                "Energy (J)",
            ),
            _line(
                "Blocked Motion Rate",
                "episode/blocked_motion_rate",
                "Rate",
                y_range=(0, 1),
            ),
        ],
    )

    remaining = [
        section
        for section in workspace.sections
        if section.name
        not in {
            "01 — Paper Overview",
            "01 — Algorithm Comparison",
        }
    ]
    for section in remaining:
        section.is_open = False
        section.pinned = False

    workspace.sections = [
        comparison,
        *remaining,
    ]
    return workspace.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view-url",
        default=DEFAULT_VIEW,
    )
    args = parser.parse_args()
    saved = format_workspace(
        args.view_url
    )
    print(saved.url)


if __name__ == "__main__":
    main()
