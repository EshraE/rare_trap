"""Shared ThinkTrap experiment defaults."""

from __future__ import annotations

import os


def _device_order_from_env() -> tuple[str, ...]:
    value = os.environ.get("THINKTRAP_DEVICE_ORDER")
    if not value:
        return ("cuda", "mps", "cpu")
    order = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    return order or ("cuda", "mps", "cpu")


DEVICE_ORDER = _device_order_from_env()
COND_PROB = 0.2
PROJECTION_SEED = 1010
LEVEL_ZERO_SCREEN_SAMPLES = 200
LEVEL_ZERO_SCREEN_PROBABILITY = 0.1
