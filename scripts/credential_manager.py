#!/usr/bin/env python3
"""Local credential/quota manager for UAV-search Kaggle workflows.

Reads key.txt (JSON), never prints secret values, checks Kaggle accelerator
quota, and launches child commands with the selected account plus W&B/HF keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_KEY_FILE = Path(__file__).resolve().parents[1] / "key.txt"


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    accounts = data.get("kaggle", {}).get("accounts", [])
    if not isinstance(accounts, list) or not accounts:
        raise RuntimeError("key.txt has no kaggle.accounts list")
    for index, item in enumerate(accounts):
        if not isinstance(item, dict) or not item.get("token"):
            raise RuntimeError(f"kaggle.accounts[{index}] is invalid")
    if not data.get("wandb", {}).get("api_key"):
        raise RuntimeError("key.txt is missing wandb.api_key")
    if not data.get("huggingface", {}).get("token"):
        raise RuntimeError("key.txt is missing huggingface.token")
    return data


def _hours(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*h", text)
    if match:
        return float(match.group(1))
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _find_resource_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "resource" in value and any(k in value for k in ("remaining", "used", "total")):
            rows.append(value)
        for child in value.values():
            rows.extend(_find_resource_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_find_resource_rows(child))
    return rows


def kaggle_quota(token: str, timeout: int = 45) -> dict[str, Any]:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    # A subprocess is deliberate: modern Kaggle auth can consume the token
    # from its process environment during import/authentication.
    proc = subprocess.run(
        ["kaggle", "quota", "--format", "json"],
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "unknown Kaggle CLI error").strip()
        # Never allow a credential to appear in diagnostics.
        message = message.replace(token, "<redacted>")
        return {"ok": False, "error": message[:800]}

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "kaggle quota did not return valid JSON",
        }

    resources: dict[str, dict[str, Any]] = {}
    for row in _find_resource_rows(payload):
        name = str(row.get("resource", "")).upper()
        if not name:
            continue
        resources[name] = {
            "used_hours": _hours(row.get("used")),
            "remaining_hours": _hours(row.get("remaining")),
            "total_hours": _hours(row.get("total")),
            "refresh_at": row.get("refreshAt") or row.get("refresh_at"),
        }
    if "GPU" not in resources:
        return {
            "ok": False,
            "error": "Kaggle quota response contained no GPU row",
            "resources": resources,
        }
    return {"ok": True, "resources": resources}


def check_accounts(data: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, account in enumerate(data["kaggle"]["accounts"]):
        name = str(account.get("name") or f"account_{index + 1:02d}")
        enabled = bool(account.get("enabled", True))
        if not enabled:
            results.append(
                {
                    "index": index,
                    "name": name,
                    "enabled": False,
                    "ok": False,
                    "error": "disabled",
                }
            )
            continue
        quota = kaggle_quota(str(account["token"]))
        result = {
            "index": index,
            "name": name,
            "enabled": True,
            **quota,
        }
        results.append(result)
    return results


def pick_account(
    data: dict[str, Any],
    *,
    purpose: str,
    required_gpu_hours: float,
    requested_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if requested_name is not None:
        accounts = data["kaggle"]["accounts"]
        for index, account in enumerate(accounts):
            name = str(account.get("name") or f"account_{index + 1:02d}")
            if name != requested_name:
                continue
            if not bool(account.get("enabled", True)):
                raise RuntimeError(
                    f"Requested Kaggle account {requested_name!r} is disabled"
                )
            quota = kaggle_quota(str(account["token"]))
            if not quota.get("ok"):
                raise RuntimeError(
                    f"Requested Kaggle account {requested_name!r} is unavailable: "
                    + str(quota.get("error"))
                )
            chosen = {
                "index": index,
                "name": name,
                "enabled": True,
                **quota,
            }
            if purpose == "train":
                remaining = (
                    chosen.get("resources", {})
                    .get("GPU", {})
                    .get("remaining_hours")
                )
                if remaining is None or float(remaining) < required_gpu_hours:
                    raise RuntimeError(
                        f"Requested Kaggle account {requested_name!r} does not have "
                        f"enough GPU quota: remaining={remaining}h, "
                        f"required={required_gpu_hours:.2f}h"
                    )
            return account, chosen
        raise RuntimeError(f"Unknown Kaggle account {requested_name!r}")

    results = check_accounts(data)
    valid = [r for r in results if r.get("ok")]
    if not valid:
        raise RuntimeError("No valid enabled Kaggle account was found")

    if purpose == "visualize":
        # CPU-only visualization must not consume GPU quota.
        chosen = valid[0]
        return data["kaggle"]["accounts"][chosen["index"]], chosen

    candidates = []
    for result in valid:
        remaining = (
            result.get("resources", {})
            .get("GPU", {})
            .get("remaining_hours")
        )
        if remaining is not None and float(remaining) >= required_gpu_hours:
            candidates.append(result)
    if not candidates:
        details = ", ".join(
            f"{r['name']}={r.get('resources', {}).get('GPU', {}).get('remaining_hours')}h"
            for r in valid
        )
        raise RuntimeError(
            "No Kaggle account has enough GPU quota. "
            f"Required={required_gpu_hours:.2f}h; available: {details}"
        )
    chosen = max(
        candidates,
        key=lambda r: float(r["resources"]["GPU"]["remaining_hours"]),
    )
    return data["kaggle"]["accounts"][chosen["index"]], chosen


def child_env(data: dict[str, Any], kaggle_token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = kaggle_token

    wandb = data["wandb"]
    env["WANDB_API_KEY"] = str(wandb["api_key"])
    if wandb.get("entity"):
        env["WANDB_ENTITY"] = str(wandb["entity"])
    if wandb.get("project"):
        env["WANDB_PROJECT"] = str(wandb["project"])

    hf = data["huggingface"]
    env["HF_TOKEN"] = str(hf["token"])
    if hf.get("repo_id"):
        env["HF_REPO_ID"] = str(hf["repo_id"])
    return env


def validate_services(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    try:
        import wandb

        api = wandb.Api(api_key=str(data["wandb"]["api_key"]), timeout=20)
        runs = api.runs(
            path=f"{data['wandb']['entity']}/{data['wandb']['project']}",
            per_page=1,
        )
        next(iter(runs), None)
        result["wandb"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        result["wandb"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }

    try:
        from huggingface_hub import HfApi

        info = HfApi(token=str(data["huggingface"]["token"])).whoami()
        result["huggingface"] = {
            "ok": True,
            "name": info.get("name") if isinstance(info, dict) else None,
        }
    except Exception as exc:  # noqa: BLE001
        result["huggingface"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }

    return result


def print_status(data: dict[str, Any], validate_external: bool) -> int:
    results = check_accounts(data)
    print("Kaggle accounts:")
    for r in results:
        if not r.get("ok"):
            print(f"  {r['name']}: unavailable ({r.get('error', 'unknown error')})")
            continue
        gpu = r["resources"].get("GPU", {})
        tpu = r["resources"].get("TPU", {})
        print(
            f"  {r['name']}: GPU remaining={gpu.get('remaining_hours')}h "
            f"(used={gpu.get('used_hours')}h / total={gpu.get('total_hours')}h), "
            f"TPU remaining={tpu.get('remaining_hours')}h"
        )

    if validate_external:
        services = validate_services(data)
        w = services["wandb"]
        h = services["huggingface"]
        print("W&B:", "OK" if w.get("ok") else f"FAILED ({w.get('error')})")
        if h.get("ok"):
            suffix = f" [{h.get('name')}]" if h.get("name") else ""
            print("Hugging Face: OK" + suffix)
        else:
            print(f"Hugging Face: FAILED ({h.get('error')})")
        if not w.get("ok") or not h.get("ok"):
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.add_argument("--validate-services", action="store_true")

    select = sub.add_parser("select")
    select.add_argument("--purpose", choices=("train", "visualize"), required=True)
    select.add_argument("--required-gpu-hours", type=float, default=0.25)
    select.add_argument("--account", type=str, default=None)

    execute = sub.add_parser("exec")
    execute.add_argument("--purpose", choices=("train", "visualize"), required=True)
    execute.add_argument("--required-gpu-hours", type=float, default=0.25)
    execute.add_argument("--account", type=str, default=None)
    execute.add_argument("child", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    data = load_config(args.key_file)

    if args.command == "status":
        return print_status(data, args.validate_services)

    account, metadata = pick_account(
        data,
        purpose=args.purpose,
        required_gpu_hours=float(args.required_gpu_hours),
        requested_name=getattr(args, "account", None),
    )
    if args.purpose == "train":
        gpu = metadata["resources"]["GPU"]
        print(
            f"Selected {metadata['name']} for GPU training: "
            f"{gpu.get('remaining_hours')}h remaining."
        )
    else:
        print(
            f"Selected {metadata['name']} for CPU visualization "
            "(GPU quota is not required)."
        )

    if args.command == "select":
        return 0

    child = list(args.child)
    if child and child[0] == "--":
        child = child[1:]
    if not child:
        raise RuntimeError("exec requires a command after --")
    proc = subprocess.run(
        child,
        env=child_env(data, str(account["token"])),
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
