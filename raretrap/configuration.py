"""Public experiment settings, independent of model loading."""
from __future__ import annotations

import copy
import json
import re
from importlib.resources import files
from raretrap.terminology import GEOMETRY_AWARE, IMPLEMENTATION_ESTIMATOR_MODE
from raretrap.terminology import PAPER_PROJECTION_ARMS as PAPER_PROJECTION_ARMS

__all__ = ["PAPER_PROJECTION_ARMS", "ESTIMATOR_PROJECTION_MODE", "validate_public_projection",
           "implementation_settings", "models", "experiments", "preset", "custom_model", "default_experiment"]


# Public paper workflows. Other maps remain available only in internal helpers.
ESTIMATOR_PROJECTION_MODE = GEOMETRY_AWARE


def validate_public_projection(settings: dict) -> None:
    if settings.get("projection_mode") != ESTIMATOR_PROJECTION_MODE:
        raise ValueError("Public estimation uses the unscaled factorized covariance-shaped map. "
                         "Alternative projections are internal helpers, not paper presets.")
    if "projection_seed" in settings:
        raise ValueError("The projection seed is internal and cannot be overridden by a public preset.")


def implementation_settings(settings: dict) -> dict:
    """Translate only the method label for the protected production interface."""
    validate_public_projection(settings)
    return dict(settings, projection_mode=IMPLEMENTATION_ESTIMATOR_MODE)


def models() -> dict:
    return json.loads(files("raretrap").joinpath("presets/models.json").read_text())


def experiments() -> dict:
    return json.loads(files("raretrap").joinpath("presets/experiments.json").read_text())


def preset(name: str) -> dict:
    registry = experiments()
    if name not in registry:
        raise ValueError(f"Unknown preset {name!r}. Run 'raretrap presets' to list choices.")
    return copy.deepcopy(registry[name])


def custom_model(model_id: str, revision: str, dtype: str = "float16", thinking: str = "auto") -> dict:
    """An explicitly pinned Hub checkpoint; never accept tokens or remote URLs."""
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(model_id)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError("Custom HF models require --revision with a full 40-character commit hash.")
    if dtype not in {"float16", "bfloat16", "float32"} or thinking not in {"auto", "enabled", "disabled"}:
        raise ValueError("Unsupported model precision or thinking-template mode.")
    return dict(model=model_id, revision=revision, dtype=dtype, thinking=thinking,
                backend="transformers", custom=True)


def default_experiment(model: str, surrogate: str = "self", *, target: dict | None = None) -> dict:
    registry = models()
    target = registry[model] if target is None else target
    surrogate_spec = target if surrogate == "self" else registry[surrogate]
    settings = {
        "tested_llm_model": target["model"],
        "tested_model_revision": target["revision"],
        "surrogate_token_model": surrogate_spec["model"],
        "surrogate_model_revision": surrogate_spec["revision"],
        "seed": 1010,
        "input_prompt_len": 40,
        "ss_sample_space_dim": 200,
        "ss_samples_per_level": 1000,
        "ss_conditional_prob": 0.1,
        "ss_max_levels": 5,
        "llm_output_len_threshold": 20000,
        "performance_metric": "output_length",
        "repetition_score_threshold": 0.99,
        "generation_batch_size": 1,
        "llm_output_sampler": "greedy",
        "chat_template_thinking": target["thinking"],
        "chat_template_reasoning_effort": None,
        "objective_repeats": 1,
        "objective_aggregation": "mean",
        "projection_mode": ESTIMATOR_PROJECTION_MODE,
        "covariance_eps": 1e-5,
    }
    return {"arguments": settings, "environment": {"THINKTRAP_MODEL_DTYPE": target["dtype"]},
            "backend": target["backend"], "model": model, "surrogate": surrogate,
            "recorded_map_matches_release": not target.get("custom", False),
            "custom_model": bool(target.get("custom", False))}
