import subprocess
import sys
from pathlib import Path

from uav_marl.training.gpu import (
    _train_full_gpu_ddp_worker,
    _write_ddp_launcher_script,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ddp_worker_is_importable_in_fresh_python_process():
    code = (
        "from uav_marl.training.gpu import _train_full_gpu_ddp_worker as f; "
        "print(f.__name__); print(f.__module__)"
    )
    cp = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert "_train_full_gpu_ddp_worker" in cp.stdout
    assert "uav_marl.training.gpu" in cp.stdout


def test_torchrun_launcher_imports_python_worker_directly(tmp_path):
    launcher = _write_ddp_launcher_script(tmp_path / "runner.py")
    source = launcher.read_text()
    compile(source, str(launcher), "exec")

    assert "from uav_marl.training.gpu import _train_full_gpu_ddp_worker" in source
    assert "cloudpickle" not in source
    assert "load_notebook_engine" not in source
    assert "fix_test_gpu.ipynb" not in source
    assert "--payload" not in source


def test_worker_object_comes_from_production_module():
    assert _train_full_gpu_ddp_worker.__module__ == "uav_marl.training.gpu"
