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

# Hydra configs/ is the user-facing source of truth. config.py is a compatibility mapping.
if not Path("config.py").is_file() and Path("/kaggle/input").is_dir():
    _gpu_config_candidates = sorted(
        Path("/kaggle/input").glob("**/config.py")
    )
    if len(_gpu_config_candidates) != 1:
        raise FileNotFoundError(
            "config.py is required. Upload it alongside test_gpu.ipynb or "
            "attach exactly one Kaggle input containing config.py. Found "
            f"{len(_gpu_config_candidates)} candidates."
        )
    _gpu_config_dir = str(_gpu_config_candidates[0].parent)
    if _gpu_config_dir not in sys.path:
        sys.path.insert(0, _gpu_config_dir)

from config import CONFIG


def _require_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full-GPU mode requires an available CUDA device")


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
