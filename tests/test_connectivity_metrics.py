from __future__ import annotations

import numpy as np
import torch

from uav_marl.common import (
    contact_graph_reachability_fraction_numpy,
    contact_graph_reachability_fraction_torch,
)


def test_numpy_contact_graph_reachability_propagates_over_peer_chain():
    positions = np.asarray(
        [
            [900.0, 0.0, 0.0],
            [1800.0, 0.0, 0.0],
            [2700.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    active = np.asarray([True, True, True])
    fraction = contact_graph_reachability_fraction_numpy(
        positions,
        active,
        np.asarray([0.0, 0.0, 0.0]),
        peer_range_m=1000.0,
        gcs_range_m=1000.0,
    )
    assert fraction == 1.0


def test_torch_contact_graph_reachability_is_batched_and_ignores_inactive_uavs():
    positions = torch.tensor(
        [
            [
                [900.0, 0.0, 0.0],
                [1800.0, 0.0, 0.0],
                [2700.0, 0.0, 0.0],
            ],
            [
                [900.0, 0.0, 0.0],
                [1900.0, 0.0, 0.0],
                [4000.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    active = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    fraction = contact_graph_reachability_fraction_torch(
        positions,
        active,
        torch.tensor([0.0, 0.0, 0.0]),
        peer_range_m=1000.0,
        gcs_range_m=1000.0,
    )
    assert torch.allclose(fraction, torch.tensor([1.0, 1.0]))
