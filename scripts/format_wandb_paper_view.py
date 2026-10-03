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

    current = workspace.runset_settings
    workspace.runset_settings = ws.RunsetSettings(
        query=current.query,
        regex_query=current.regex_query,
        filters=current.filters,
        groupby=list(current.groupby),
        order=list(current.order),
        run_settings=dict(current.run_settings),
        group_colors=dict(current.group_colors),
        pinned_columns=[
            "run:displayName",
            "config:algorithm",
            "config:seed",
            "config:kaggle_account",
            "config:num_envs",
            "config:target_episodes",
            "config:network_backend",
            "summary:episodes_completed",
        ],
        baseline_run=current.baseline_run,
        pinned_runs=list(current.pinned_runs),
    )

    comparison = ws.Section(
        name="01 — Algorithm Comparison",
        is_open=True,
        pinned=True,
        panels=[
            _line(
                "Total Reward (Episode Return)",
                "episode/return",
                "Episode return",
            ),
            _line(
                "Mission Success Rate",
                "episode/success",
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
                "episode/coverage_percent",
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
                "episode/total_energy_j",
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

    safety_errors = ws.Section(
        name="02 — Safety + Errors",
        is_open=True,
        pinned=False,
        panels=[
            _line("Blocked Motion Rate", "episode/blocked_motion_rate", "Rate", y_range=(0, 1)),
            _line("Boundary Clip Rate", "episode/boundary_clip_rate", "Rate", y_range=(0, 1)),
            _line("Peer Safety Block Rate", "episode/peer_safety_block_rate", "Rate", y_range=(0, 1)),
            _line("Obstacle Block Rate", "episode/obstacle_block_rate", "Rate", y_range=(0, 1)),
            _line("False Confirmations", "episode/false_confirmations", "Count"),
            _line("Expired Reports", "episode/expired_reports", "Count"),
            _line("Dropped Reports", "episode/dropped_reports", "Count"),
        ],
    )

    energy = ws.Section(
        name="03 — Energy",
        is_open=True,
        pinned=False,
        panels=[
            _line("Total Energy", "episode/total_energy_j", "Energy (J)"),
            _line("Communication Energy", "episode/communication_energy_j", "Energy (J)"),
            _line("Battery Remaining", "episode/battery_remaining_mean_percent", "Battery (%)", y_range=(0, 100)),
            _line("Energy / Confirmed Target", "episode/energy_per_confirmed_target_j", "J / target"),
            _line("Energy / Delivered Target", "episode/energy_per_delivered_target_j", "J / target"),
        ],
    )

    network = ws.Section(
        name="04 — Network",
        is_open=True,
        pinned=False,
        panels=[
            _line("PHY Success", "episode/network_phy_success_percent", "PHY success (%)", y_range=(0, 100)),
            _line("Network Throughput", "episode/network_throughput_kbps", "Throughput (kbps)"),
            _line("TX Payload Success Ratio", "episode/network_tx_payload_success_ratio", "Ratio", y_range=(0, 1)),
            _line("TX Attempts", "episode/network_tx_attempts", "Attempts"),
            _line("NLOS Attempt Rate", "episode/network_nlos_attempt_rate_percent", "NLOS (%)", y_range=(0, 100)),
            _line("GCS In-range UAV Fraction", "episode/gcs_in_range_uav_fraction", "Fraction", y_range=(0, 1)),
            _line("Report Delivery Latency", "episode/report_delivery_latency_s", "Latency (s)"),
        ],
    )

    mission_timing = ws.Section(
        name="05 — Episode + Mission Timing",
        is_open=True,
        pinned=False,
        panels=[
            _line("Episode Length", "episode/length", "Steps"),
            _line("Episode End Step", "episode/end_step", "Step"),
            _line("Distance Flown", "episode/distance_total_m", "Distance (m)"),
            _line("Information Gain", "episode/information_gain_bits", "Information gain (bits)"),
            _line("Time to First Confirm", "episode/time_to_first_confirm_s", "Time (s)"),
            _line("Time to All Confirm", "episode/time_to_all_confirm_s", "Time (s)"),
            _line("Time to First Delivery", "episode/time_to_first_delivery_s", "Time (s)"),
        ],
    )

    remaining = [
        section
        for section in workspace.sections
        if section.name
        not in {
            "01 — Paper Overview",
            "01 — Algorithm Comparison",
            "02 — Policy Diagnostics",
            "02 — Safety + Errors",
            "03 — Energy",
            "04 — Network",
            "05 — Episode + Mission Timing",
        }
    ]
    for section in remaining:
        section.is_open = False
        section.pinned = False

    workspace.sections = [
        comparison,
        safety_errors,
        energy,
        network,
        mission_timing,
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
