from __future__ import annotations

import os
from pathlib import Path

import pytest

from uav_marl.network.backends import uavnetsim_available
from uav_marl.training.gpu import (
    benchmark_uavnetsim_full_flow,
    validate_uavnetsim_gpu_direct_link,
    validate_uavnetsim_network_parity,
)

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_UAVNETSIM_INTEGRATION") != "1"
    or not uavnetsim_available(None),
    reason=(
        "set RUN_UAVNETSIM_INTEGRATION=1 with UavNetSim installed "
        "to run the authoritative network regressions"
    ),
)


def test_uavnetsim_direct_link_and_bridge_parity():
    direct = validate_uavnetsim_gpu_direct_link(
        ROOT,
        repo_path=None,
        seed=44,
        steps=6,
        device="cpu",
        strict_cuda=False,
    )
    assert direct["passed"] is True
    assert direct["byte_parity"] is True
    assert direct["delivery_parity"] is True
    assert direct["reward_max_abs_diff"] < 1e-4

    network = validate_uavnetsim_network_parity(
        ROOT,
        repo_path=None,
        seed=44,
        device="cpu",
        strict_cuda=False,
        steps=4,
    )
    assert network["passed"] is True


def test_uavnetsim_full_flow_cpu_tensor_parity():
    result = benchmark_uavnetsim_full_flow(
        ROOT,
        repo_path=None,
        steps=6,
        seed=44,
        device="cpu",
        strict_cuda=False,
        communication_loaded=True,
    )
    parity = result["parity"]
    assert parity["network_core_equal"] is True
    assert parity["confirmed_equal"] is True
    assert parity["delivered_equal"] is True

    # CPU/reference code uses NumPy float64 in several places while the
    # tensor environment is float32. Sub-millimeter / ~1e-5 differences are
    # numerical precision, not mission-semantic divergence.
    assert parity["final_position_max_abs_diff_m"] < 1e-3
    assert parity["final_belief_max_abs_diff"] < 1e-5
    assert parity["reward_max_abs_diff"] < 1e-4
