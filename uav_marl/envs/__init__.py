"""UAV-search environments.

Keep package initialization lightweight: CPU/reference and GPU batched
implementations are imported lazily to avoid algorithm/environment cycles.
"""

from .uav_search import UAVSearchEnv


def __getattr__(name):
    if name == "FullGpuUAVBatchEnv":
        from .gpu import FullGpuUAVBatchEnv

        return FullGpuUAVBatchEnv
    raise AttributeError(name)


__all__ = ["FullGpuUAVBatchEnv", "UAVSearchEnv"]
