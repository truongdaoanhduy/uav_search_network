import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_package_imports_from_non_repo_working_directory(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    code = (
        "from uav_marl.training.gpu import _train_full_gpu_ddp_worker; "
        "print(_train_full_gpu_ddp_worker.__module__)"
    )
    cp = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert cp.returncode == 0, cp.stderr
    assert "uav_marl.training.gpu" in cp.stdout
