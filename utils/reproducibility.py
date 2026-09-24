"""Runtime and repository metadata for reproducible experiment artifacts."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_state() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[1]
    unknown = {"commit": None, "branch": None, "dirty": None}
    # Installed wheels must not discover an unrelated enclosing repository.
    if not (repository / ".git").exists():
        return unknown

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=repository, check=False,
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def runtime_metadata(device: torch.device | None = None) -> dict:
    accelerator = {
        "type": getattr(device, "type", None),
        "name": None,
        "device_count": 0,
        "total_memory_bytes": None,
        "compute_capability": None,
    }
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        accelerator.update(
            {
                "name": properties.name,
                "device_count": int(torch.cuda.device_count()),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    elif getattr(device, "type", None) == "mps":
        accelerator.update(
            {
                "name": platform.processor() or platform.machine(),
                "device_count": 1,
            }
        )
    elif getattr(device, "type", None) == "cpu":
        accelerator.update(
            {
                "name": platform.processor() or platform.machine(),
                "device_count": 1,
            }
        )
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
        },
        "torch_build": {
            "git_version": torch.version.git_version,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "accelerator": accelerator,
    }
