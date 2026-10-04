"""Production source migrated from frozen fix_test_gpu.ipynb.

Do not import the notebook at runtime. The notebook is retained only as a
reference/regression artifact. Production changes belong in Python modules.
Migrated from notebook cells 3..6.
"""

# --- frozen notebook cell 3 ---
from __future__ import annotations

import ast
import random
import sys
import types
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Hydra configs/ is the user-facing source of truth. config.py is a
# compatibility mapping imported through the Python module path; production
# code must not depend on the process current working directory.
from config import CONFIG


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-GPU mode requires an available CUDA device")


def contact_graph_reachability_fraction_numpy(
    positions,
    active,
    gcs_position,
    *,
    peer_range_m,
    gcs_range_m,
):
    """Fraction of active UAVs with a range-graph path to the GCS."""
    positions = np.asarray(positions, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    gcs_position = np.asarray(gcs_position, dtype=np.float64).reshape(3)
    active_count = int(np.count_nonzero(active))
    if active_count == 0:
        return 0.0

    peer_distance = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )
    peer_edges = (
        peer_distance <= float(peer_range_m)
    ) & active[:, None] & active[None, :]
    gcs_distance = np.linalg.norm(
        positions - gcs_position[None, :],
        axis=-1,
    )
    reachable = active & (gcs_distance <= float(gcs_range_m))
    for _ in range(max(0, int(active.shape[0]) - 1)):
        expanded = reachable | np.any(
            peer_edges & reachable[None, :],
            axis=1,
        )
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    return float(np.count_nonzero(reachable) / active_count)


def contact_graph_reachability_fraction_torch(
    positions,
    active,
    gcs_position,
    *,
    peer_range_m,
    gcs_range_m,
):
    """Batched torch equivalent of contact_graph_reachability_fraction_numpy."""
    active = active.to(dtype=torch.bool)
    gcs_position = torch.as_tensor(
        gcs_position,
        device=positions.device,
        dtype=positions.dtype,
    ).reshape(1, 1, 3)
    peer_distance = torch.linalg.vector_norm(
        positions.unsqueeze(2) - positions.unsqueeze(1),
        dim=-1,
    )
    peer_edges = (
        peer_distance <= float(peer_range_m)
    ) & active.unsqueeze(2) & active.unsqueeze(1)
    gcs_distance = torch.linalg.vector_norm(
        positions - gcs_position,
        dim=-1,
    )
    reachable = active & (gcs_distance <= float(gcs_range_m))
    for _ in range(max(0, int(positions.shape[1]) - 1)):
        reachable = reachable | (
            peer_edges & reachable.unsqueeze(1)
        ).any(dim=-1)
    return (
        reachable.float().sum(dim=-1)
        / active.float().sum(dim=-1).clamp_min(1.0)
    )


# --- frozen notebook cell 4 ---
import os as _bootstrap_os
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
import copy
import csv
import importlib
import json
import os
import random
import secrets
import subprocess
import tempfile
import sys
import time
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import ClassVar
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from config import CONFIG


# --- frozen notebook cell 5 ---
def set_seed(seed=44):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if hasattr(
            torch.backends.cudnn,
            "allow_tf32",
        ):
            torch.backends.cudnn.allow_tf32 = False

    if (
        hasattr(
            torch.backends,
            "cuda",
        )
        and hasattr(
            torch.backends.cuda,
            "matmul",
        )
    ):
        torch.backends.cuda.matmul.allow_tf32 = False


# --- frozen notebook cell 6 ---
from dataclasses import asdict, dataclass, is_dataclass


# Export public and internal helpers required by downstream production layers.
__all__ = [name for name in globals() if not name.startswith("__")]
