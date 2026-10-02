"""Training-to-CPU-visualization handoff utilities.

GPU production jobs write a compact handoff package into /kaggle/working
(or a local fallback). A later CPU-only job consumes that package, verifies the
checkpoint digest, restores the resolved experiment config, and runs the
authoritative CPU post-process path.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

HANDOFF_FILENAME = "uav_training_handoff.json"
CHECKPOINT_FILENAME = "checkpoint.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def default_handoff_root(repo: str | Path) -> Path:
    repo = Path(repo).resolve()
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "uav_training_handoff"
    return repo / "uav_training_handoff"


def create_training_handoff(
    training_result: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    repo: str | Path,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    result = dict(training_result)
    checkpoint_value = result.get("final_checkpoint_path")
    if not checkpoint_value:
        paths = list(result.get("checkpoint_paths") or [])
        if paths:
            checkpoint_value = paths[-1]
    if not checkpoint_value:
        raise RuntimeError(
            "training completed without a final checkpoint; CPU handoff cannot be created"
        )

    checkpoint_path = Path(str(checkpoint_value)).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    if checkpoint_path.stat().st_size <= 0:
        raise RuntimeError(f"final checkpoint is empty: {checkpoint_path}")

    algorithm = str(resolved_config["algorithm"]["name"]).strip().lower()
    seed = int(resolved_config["seed"])
    run_token = str(result.get("wandb_run_id") or "no-wandb-id")
    run_dir = (
        default_handoff_root(repo)
        / f"{algorithm}-seed{seed}-{run_token}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_checkpoint = run_dir / CHECKPOINT_FILENAME
    if checkpoint_path != copied_checkpoint:
        shutil.copy2(checkpoint_path, copied_checkpoint)

    checkpoint_sha256 = _sha256(copied_checkpoint)
    artifact_ref = (
        result.get("final_checkpoint_artifact")
        or result.get("checkpoint_artifact")
        or result.get("wandb_checkpoint_artifact")
    )

    payload = {
        "schema_version": 1,
        "algorithm": algorithm,
        "seed": seed,
        "network_backend": str(
            resolved_config["runtime"]["network_backend"]
        ),
        "experiment": str(resolved_config["experiment"]["name"]),
        "source_provider": str(resolved_config["runtime"]["provider"]),
        "source_execution_mode": str(
            resolved_config["runtime"]["execution_mode"]
        ),
        "source_wandb_run_id": result.get("wandb_run_id"),
        "source_wandb_run_url": result.get("wandb_run_url"),
        "source_wandb_artifact": artifact_ref,
        "source_git_commit": _git_commit(repo),
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": copied_checkpoint.stat().st_size,
        "episodes_completed": result.get("episodes_completed"),
        "global_step": result.get("global_step"),
        "resolved_config_sha256": result.get("resolved_config_sha256"),
        "resolved_config": deepcopy(dict(resolved_config)),
        "postprocess": {
            "eval_episodes": int(
                resolved_config["experiment"]["postprocess"]["eval_episodes"]
            ),
            "video_fps": int(
                resolved_config["experiment"]["postprocess"]["video_fps"]
            ),
            "render_video": bool(
                resolved_config["experiment"]["visualization"]["render_video"]
            ),
            "upload_media_to_wandb": bool(
                resolved_config["experiment"]["visualization"][
                    "upload_media_to_wandb"
                ]
            ),
            "upload_huggingface": bool(
                resolved_config["experiment"]["postprocess"][
                    "upload_huggingface"
                ]
            ),
        },
    }

    handoff_path = run_dir / HANDOFF_FILENAME
    handoff_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )
    result.update(
        {
            "visualization_complete": False,
            "visualization_pending": True,
            "postprocess_mode": "separate_cpu",
            "handoff_dir": str(run_dir),
            "handoff_path": str(handoff_path),
            "handoff_checkpoint_path": str(copied_checkpoint),
            "handoff_checkpoint_sha256": checkpoint_sha256,
        }
    )
    return result


def find_handoff(root: str | Path) -> Path:
    root = Path(root)
    if root.is_file():
        if root.name != HANDOFF_FILENAME:
            raise ValueError(
                f"handoff file must be named {HANDOFF_FILENAME}"
            )
        return root.resolve()

    matches = sorted(root.rglob(HANDOFF_FILENAME))
    if not matches:
        raise FileNotFoundError(
            f"no {HANDOFF_FILENAME} found under {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            "multiple training handoffs found; pass an explicit handoff path: "
            + ", ".join(str(path) for path in matches[:10])
        )
    return matches[0].resolve()


def load_training_handoff(
    handoff: str | Path,
    *,
    verify_checkpoint: bool = True,
) -> tuple[dict[str, Any], Path]:
    handoff_path = find_handoff(handoff)
    payload = json.loads(handoff_path.read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError(
            f"unsupported handoff schema: {payload.get('schema_version')}"
        )

    checkpoint_path = (
        handoff_path.parent / str(payload["checkpoint_filename"])
    ).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    if checkpoint_path.stat().st_size != int(payload["checkpoint_bytes"]):
        raise RuntimeError(
            "checkpoint size mismatch in CPU handoff package"
        )
    if verify_checkpoint:
        actual = _sha256(checkpoint_path)
        expected = str(payload["checkpoint_sha256"])
        if actual != expected:
            raise RuntimeError(
                "checkpoint SHA256 mismatch in CPU handoff package"
            )
    return payload, checkpoint_path


def restore_legacy_config_from_handoff(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    from config import CONFIG as LEGACY_CONFIG

    from .configuration import apply_to_legacy_config

    resolved_config = deepcopy(dict(payload["resolved_config"]))
    apply_to_legacy_config(LEGACY_CONFIG, resolved_config)
    return resolved_config


def run_cpu_postprocess_from_handoff(
    handoff: str | Path,
    *,
    repo: str | Path,
    output_dir: str | Path,
    verify_checkpoint: bool = True,
    log_wandb: bool | None = None,
    upload_huggingface: bool | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    payload, checkpoint_path = load_training_handoff(
        handoff,
        verify_checkpoint=verify_checkpoint,
    )
    restore_legacy_config_from_handoff(payload)

    from .evaluation import postprocess_checkpoint_cpu

    postprocess_fn = postprocess_checkpoint_cpu

    if log_wandb is None:
        log_wandb = bool(
            payload["postprocess"].get(
                "upload_media_to_wandb",
                False,
            )
        )
    if upload_huggingface is None:
        upload_huggingface = bool(
            payload["postprocess"].get(
                "upload_huggingface",
                False,
            )
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = postprocess_fn(
        checkpoint_path=checkpoint_path,
        artifact_ref=None,
        algorithm=str(payload["algorithm"]),
        network_backend=str(payload["network_backend"]),
        seed=int(payload["seed"]),
        episodes=int(payload["postprocess"]["eval_episodes"]),
        output_dir=output_dir,
        source_run_id=payload.get("source_wandb_run_id"),
        log_wandb=bool(log_wandb),
        upload_huggingface=bool(upload_huggingface),
        render_video=bool(
            payload["postprocess"].get("render_video", True)
        ),
    )

    image_path = Path(result["image_path"])
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        raise RuntimeError("CPU visualization PNG was not created")

    video_value = result.get("video_path")
    if bool(payload["postprocess"].get("render_video", True)):
        if not video_value:
            raise RuntimeError("CPU visualization MP4 path is missing")
        video_path = Path(video_value)
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError("CPU visualization MP4 was not created")

    completion = {
        "visualization_complete": True,
        "source_handoff": str(find_handoff(handoff)),
        "source_checkpoint_sha256": str(payload["checkpoint_sha256"]),
        **result,
    }
    completion_path = output_dir / "cpu_visualization_complete.json"
    completion_path.write_text(
        json.dumps(
            completion,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return completion
