"""Temporary compatibility loader for the notebook-first training engine.

The long-term target is to move trainer/environment classes into normal Python
modules. Until then, this loader executes only declaration/configuration cells
from fix_test_gpu.ipynb and rejects unexpected top-level executable statements.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat

_ALLOWED_TOP_LEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.If,
)


def _validate_cell_safety(source: str, cell_index: int) -> None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, _ALLOWED_TOP_LEVEL):
            continue
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            ):
                continue
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_get_torch_sensing_tls"
            ):
                continue
        raise RuntimeError(
            "refusing to execute unexpected top-level notebook statement "
            f"in cell {cell_index}: {type(node).__name__}"
        )


def load_notebook_engine(
    repo: str | Path,
    notebook_name: str = "fix_test_gpu.ipynb",
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    notebook_path = repo / notebook_name
    if not notebook_path.is_file():
        raise FileNotFoundError(notebook_path)

    notebook = nbformat.read(notebook_path, as_version=4)
    module_name = "uav_marl_notebook_engine"
    module = ModuleType(module_name)
    module.__file__ = str(notebook_path)
    module.__package__ = None
    sys.modules[module_name] = module

    # This module is created dynamically from notebook declarations and is not
    # importable by name in fresh torchrun worker processes. Without explicit
    # by-value registration, cloudpickle records worker functions by reference
    # and child processes fail with ModuleNotFoundError.
    import cloudpickle

    cloudpickle.register_pickle_by_value(module)
    namespace: dict[str, Any] = module.__dict__

    old_cwd = Path.cwd()
    added_path = False
    try:
        os.chdir(repo)
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
            added_path = True

        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            _validate_cell_safety(cell.source, index)
            code = compile(
                cell.source,
                f"{notebook_path}#cell-{index}",
                "exec",
            )
            exec(code, namespace, namespace)  # noqa: S102 - AST-validated declarations only.
    finally:
        os.chdir(old_cwd)
        if added_path:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass

    # The distributed trainer serializes notebook-defined workers with
    # cloudpickle before launching torchrun. This module exists only in-memory,
    # so serializing it by reference would make child interpreters fail with
    # ModuleNotFoundError. Register the dynamic module for by-value pickling.
    import cloudpickle

    cloudpickle.register_pickle_by_value(module)

    required = {
        "train_full_gpu_auto",
        "postprocess_checkpoint_cpu",
        "finalize_training_visualization",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(
            "notebook engine is missing required symbols: "
            + ", ".join(missing)
        )
    return namespace
