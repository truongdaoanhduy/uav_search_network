import subprocess
import sys
from pathlib import Path

import cloudpickle

from uav_marl.notebook_engine import load_notebook_engine

ROOT = Path(__file__).resolve().parents[1]


def test_ddp_worker_unpickles_in_fresh_interpreter(tmp_path):
    namespace = load_notebook_engine(ROOT)
    worker = namespace["_train_full_gpu_ddp_worker"]
    payload = tmp_path / "worker.cloudpickle"
    with payload.open("wb") as handle:
        cloudpickle.dump(worker, handle)

    code = (
        "import cloudpickle, sys\n"
        "from uav_marl.notebook_engine import load_notebook_engine\n"
        "repo=sys.argv[1]\n"
        "load_notebook_engine(repo)\n"
        "with open(sys.argv[2], 'rb') as h:\n"
        "    worker=cloudpickle.load(h)\n"
        "print(worker.__name__, worker.__module__)\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT),
            str(payload),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "_train_full_gpu_ddp_worker" in proc.stdout
    assert "uav_marl_notebook_engine" in proc.stdout
