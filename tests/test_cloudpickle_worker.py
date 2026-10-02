import subprocess
import sys
from pathlib import Path

import cloudpickle

from uav_marl.notebook_engine import load_notebook_engine

ROOT = Path(__file__).resolve().parents[1]


def test_ddp_worker_cloudpickle_roundtrip_in_fresh_process(tmp_path):
    ns = load_notebook_engine(ROOT)
    worker = ns["_train_full_gpu_ddp_worker"]

    payload = tmp_path / "worker.pkl"
    with payload.open("wb") as handle:
        cloudpickle.dump(worker, handle)

    code = (
        "import cloudpickle; "
        f"f=cloudpickle.load(open({str(payload)!r}, 'rb')); "
        "print(f.__name__); "
        "print(f.__module__)"
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
