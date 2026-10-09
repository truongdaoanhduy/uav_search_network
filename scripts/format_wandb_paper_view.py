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


def _train_line(
    title,
    metric,
    ylabel,
    *,
    y_range=(None, None),
):
    return wr.LinePlot(
        title=title,
        x=wr.Metric("train/vector_steps"),
        y=[wr.Metric(metric)],
        title_x="Vector step",
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

    mission_learning = ws.Section(
        name="01 — Mission / Learning",
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
        ],
    )

    safety_errors = ws.Section(
        name="03 — Safety + Errors",
        is_open=True,
        pinned=False,
        panels=[
            _line("APF Intervention Rate", "episode/apf_intervention_rate", "Rate", y_range=(0, 1)),
            _line("APF Peer Intervention Rate", "episode/apf_peer_intervention_rate", "Rate", y_range=(0, 1)),
            _line("APF Obstacle Intervention Rate", "episode/apf_obstacle_intervention_rate", "Rate", y_range=(0, 1)),
            _line("APF Boundary Intervention Rate", "episode/apf_boundary_intervention_rate", "Rate", y_range=(0, 1)),
            _line("APF Emergency Rate", "episode/apf_emergency_rate", "Rate", y_range=(0, 1)),
            _line("APF Correction Magnitude", "episode/apf_correction_norm_mean_mps2", "Acceleration (m/s²)"),
            _line("Boundary Clip Rate", "episode/boundary_clip_rate", "Rate", y_range=(0, 1)),
            _line("False Confirmations", "episode/false_confirmations", "Count"),
            _line("Expired Reports", "episode/expired_reports", "Count"),
            _line("Dropped Reports", "episode/dropped_reports", "Count"),
        ],
    )

    energy = ws.Section(
        name="04 — Energy",
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
        name="05 — Communication / Network",
        is_open=True,
        pinned=False,
        panels=[
            _line("PHY Success", "episode/network_phy_success_percent", "PHY success (%)", y_range=(0, 100)),
            _line("Network Throughput", "episode/network_throughput_kbps", "Throughput (kbps)"),
            _line("TX Payload Success Ratio", "episode/network_tx_payload_success_ratio", "Ratio", y_range=(0, 1)),
            _line("TX Attempts", "episode/network_tx_attempts", "Attempts"),
            _line("NLOS Attempt Rate", "episode/network_nlos_attempt_rate_percent", "NLOS (%)", y_range=(0, 100)),
            _line("GCS In-range UAV Fraction", "episode/gcs_in_range_uav_fraction", "Fraction", y_range=(0, 1)),
            _line("GCS Contact-Graph Reachability", "episode/gcs_contact_graph_reachability_percent", "Reachable UAVs (%)", y_range=(0, 100)),
            _line("Report Delivery Latency", "episode/report_delivery_latency_s", "Latency (s)"),
        ],
    )

    mission_efficiency = ws.Section(
        name="02 — Mission Efficiency",
        is_open=True,
        pinned=False,
        panels=[
            _line("Episode Length", "episode/length", "Steps"),
            _line("Episode End Step", "episode/end_step", "Step"),
            _line("Distance Flown", "episode/distance_total_m", "Distance (m)"),
            _line("Information Gain", "episode/information_gain_bits", "Information gain (bits)"),
            _line("Target Encounter Rate", "episode/target_encounter_rate_percent", "Encounter rate (%)", y_range=(0, 100)),
            _line("Report Delivery Efficiency", "episode/report_delivery_given_confirmation_rate_percent", "Delivered / confirmed (%)", y_range=(0, 100)),
            _line("Coverage Redundancy", "episode/coverage_redundancy_percent", "Redundant sensing (%)", y_range=(0, 100)),
            _line("Time to First Confirm", "episode/time_to_first_confirm_s", "Time (s)"),
            _line("Time to All Confirm", "episode/time_to_all_confirm_s", "Time (s)"),
            _line("Time to First Delivery", "episode/time_to_first_delivery_s", "Time (s)"),
        ],
    )

    training_diagnostics = ws.Section(
        name="06 — Training Diagnostics",
        is_open=False,
        pinned=False,
        panels=[
            _train_line("Actor Loss", "train/actor_loss", "Loss"),
            _train_line("Critic Loss", "train/critic_loss", "Loss"),
            _train_line("Continuous Entropy", "train/continuous_entropy", "Entropy"),
            _train_line("Discrete Entropy", "train/discrete_entropy", "Entropy"),
            _train_line("Target Q Mean", "train/target_q_mean", "Q value"),
            _train_line("Investigation Shaping", "train/reward_investigation_shaping_mean", "Reward / step"),
            _train_line("APF Intervention Penalty", "train/reward_apf_intervention_mean", "Reward / step"),
            _train_line("Inactive Horizon Makeup", "train/reward_inactive_horizon_makeup_mean", "Reward / step"),
            _train_line("Optimizer Updates / 1000 Transitions", "train/optimizer_updates_per_1000_transitions", "Updates"),
            _train_line("Replay Samples / New Transition", "train/replay_samples_per_new_transition_effective", "Ratio"),
            _train_line("Transitions / Second", "train/transitions_per_second", "Transitions/s"),
        ],
    )

    remaining = [
        section
        for section in workspace.sections
        if section.name
        not in {
            "01 — Paper Overview",
            "01 — Algorithm Comparison",
            "01 — Mission / Learning",
            "02 — Policy Diagnostics",
            "02 — Safety + Errors",
            "02 — Mission Efficiency",
            "03 — Energy",
            "03 — Safety + Errors",
            "04 — Network",
            "04 — Energy",
            "05 — Episode + Mission Timing",
            "05 — Communication / Network",
            "06 — Training Diagnostics",
        }
    ]
    for section in remaining:
        section.is_open = False
        section.pinned = False

    workspace.sections = [
        mission_learning,
        mission_efficiency,
        safety_errors,
        energy,
        network,
        training_diagnostics,
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
