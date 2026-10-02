import ast
import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
FROZEN_NOTEBOOK_SHA256 = (
    "47132f07f1fb0ffc84f90171fc422d72a36679b51f01a87ba3dc969e8dbe1a62"
)


def test_fix_test_gpu_notebook_is_frozen_reference():
    notebook = ROOT / "fix_test_gpu.ipynb"
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert digest == FROZEN_NOTEBOOK_SHA256


def test_frozen_notebook_keeps_reference_symbols():
    nb = nbformat.read(ROOT / "fix_test_gpu.ipynb", as_version=4)
    symbols = set()
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols.add(node.name)

    expected = {
        "UAVSearchEnv",
        "FullGpuUAVBatchEnv",
        "HybridMASAC",
        "HybridMATD3",
        "train_full_gpu_auto",
        "postprocess_checkpoint_cpu",
    }
    assert expected <= symbols


def test_production_runtime_does_not_reference_notebook_loader():
    production_roots = [
        ROOT / "uav_marl",
        ROOT / "train.py",
        ROOT / "visualize.py",
    ]
    for root in production_roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            text = path.read_text()
            assert "load_notebook_engine" not in text
