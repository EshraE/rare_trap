"""Model and surrogate asset loading for ThinkTrap experiments."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file
from safetensors import safe_open

from utils.config import DEVICE_ORDER


@dataclass
class ModelAssets:
    model: Any
    tokenizer: AutoTokenizer
    surrogate_tokenizer: AutoTokenizer
    surrogate_embedding_weight: torch.Tensor
    surrogate_name: str
    device: torch.device
    dtype: torch.dtype
    tested_model_revision: Optional[str] = None
    tested_tokenizer_revision: Optional[str] = None
    surrogate_model_revision: Optional[str] = None
    surrogate_tokenizer_revision: Optional[str] = None
    requested_device_map: Optional[str] = None
    resolved_device_map: Optional[dict[str, Any]] = None
    generator_backend: str = "transformers"


def _resolved_revision(asset) -> Optional[str]:
    config = getattr(asset, "config", None)
    revision = getattr(config, "_commit_hash", None)
    if revision:
        return str(revision)
    revision = getattr(asset, "_commit_hash", None)
    if revision:
        return str(revision)
    init_kwargs = getattr(asset, "init_kwargs", {})
    if isinstance(init_kwargs, dict) and init_kwargs.get("_commit_hash"):
        return str(init_kwargs["_commit_hash"])
    return None


def pick_device(order: Tuple[str, ...] = DEVICE_ORDER) -> torch.device:
    for name in order:
        name = str(name).lower()
        if name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def choose_model_dtype(dev: torch.device) -> torch.dtype:
    dtype_override = os.environ.get("THINKTRAP_MODEL_DTYPE", "").strip().lower()
    if dtype_override:
        if dtype_override in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_override in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_override in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(
            "THINKTRAP_MODEL_DTYPE must be one of bf16, fp16, or fp32 "
            f"(got {dtype_override!r})."
        )
    if dev.type == "cuda":
        return torch.float16
    if dev.type == "mps":
        return torch.float16
    return torch.float32


def requested_model_device_map() -> Optional[str]:
    """Return an opt-in Transformers device map without changing legacy runs."""

    value = os.environ.get("THINKTRAP_MODEL_DEVICE_MAP", "").strip().lower()
    if not value:
        return None
    allowed = {"auto", "balanced", "balanced_low_0", "sequential"}
    if value not in allowed:
        raise ValueError(
            "THINKTRAP_MODEL_DEVICE_MAP must be one of auto, balanced, "
            f"balanced_low_0, or sequential (got {value!r})."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("THINKTRAP_MODEL_DEVICE_MAP requires CUDA.")
    return value


def requested_model_max_memory(device_map: Optional[str]) -> Optional[dict[int, str]]:
    """Build a per-visible-GPU memory ceiling for Accelerate dispatch."""

    value = os.environ.get("THINKTRAP_MODEL_MAX_MEMORY_GIB", "").strip()
    if not value:
        return None
    if device_map is None:
        raise ValueError(
            "THINKTRAP_MODEL_MAX_MEMORY_GIB requires THINKTRAP_MODEL_DEVICE_MAP."
        )
    try:
        limit_gib = float(value)
    except ValueError as exc:
        raise ValueError("THINKTRAP_MODEL_MAX_MEMORY_GIB must be numeric.") from exc
    if limit_gib <= 0.0:
        raise ValueError("THINKTRAP_MODEL_MAX_MEMORY_GIB must be positive.")
    gpu_count = int(torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("No visible CUDA GPUs are available for device mapping.")
    rendered = f"{limit_gib:g}GiB"
    return {gpu_index: rendered for gpu_index in range(gpu_count)}


def _module_device(module, fallback: torch.device) -> torch.device:
    weight = getattr(module, "weight", None)
    device = getattr(weight, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    return fallback


def requested_generator_backend() -> str:
    value = os.environ.get("THINKTRAP_GENERATOR_BACKEND", "transformers").strip().lower()
    if value not in {"transformers", "vllm"}:
        raise ValueError(
            "THINKTRAP_GENERATOR_BACKEND must be transformers or vllm "
            f"(got {value!r})."
        )
    return value


def _load_self_surrogate_embedding(
    model_name: str,
    revision: Optional[str],
) -> torch.Tensor:
    """Load only the checkpoint's input embedding table from safetensors."""

    index_path = cached_file(
        model_name,
        "model.safetensors.index.json",
        revision=revision,
    )
    if index_path is None:
        raise RuntimeError(f"Missing model.safetensors.index.json for {model_name}.")
    with Path(index_path).open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    preferred_names = (
        "backbone.embeddings.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    )
    embedding_name = next((name for name in preferred_names if name in weight_map), None)
    if embedding_name is None:
        candidates = [
            str(name)
            for name in weight_map
            if str(name).endswith(("embed_tokens.weight", "embeddings.weight", "wte.weight"))
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely identify the input embedding tensor; "
                f"candidates={candidates}."
            )
        embedding_name = candidates[0]
    shard_name = str(weight_map[embedding_name])
    shard_path = cached_file(model_name, shard_name, revision=revision)
    if shard_path is None:
        raise RuntimeError(f"Missing embedding shard {shard_name} for {model_name}.")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor(embedding_name)
    return embedding.detach().to("cpu", torch.float32)


def load_model_assets(
    *,
    tested_llm_model_name: str,
    surrogate_token_model_name: Optional[str],
    tested_model_revision: Optional[str] = None,
    surrogate_model_revision: Optional[str] = None,
) -> ModelAssets:
    device = pick_device()
    dtype = choose_model_dtype(device)
    generator_backend = requested_generator_backend()
    device_map = requested_model_device_map()
    max_memory = requested_model_max_memory(device_map)

    tokenizer = AutoTokenizer.from_pretrained(
        tested_llm_model_name,
        revision=tested_model_revision,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Target tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tested_tokenizer_revision = _resolved_revision(tokenizer)

    surrogate_name = surrogate_token_model_name or tested_llm_model_name
    if generator_backend == "vllm":
        if surrogate_name == tested_llm_model_name:
            surrogate_tokenizer = tokenizer
            embed_weight = _load_self_surrogate_embedding(
                tested_llm_model_name,
                tested_model_revision,
            )
            surrogate_model_revision = tested_model_revision or tested_tokenizer_revision
            surrogate_tokenizer_revision = tested_tokenizer_revision
        else:
            surrogate_tokenizer = AutoTokenizer.from_pretrained(
                surrogate_name,
                revision=surrogate_model_revision,
            )
            surrogate_model = AutoModelForCausalLM.from_pretrained(
                surrogate_name,
                revision=surrogate_model_revision,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to("cpu").eval()
            embed_weight = (
                surrogate_model.get_input_embeddings().weight.detach().to("cpu", torch.float32)
            )
            surrogate_model_revision = _resolved_revision(surrogate_model)
            surrogate_tokenizer_revision = _resolved_revision(surrogate_tokenizer)
            del surrogate_model
        tested_model_revision = tested_model_revision or tested_tokenizer_revision
        return ModelAssets(
            model=None,
            tokenizer=tokenizer,
            surrogate_tokenizer=surrogate_tokenizer,
            surrogate_embedding_weight=embed_weight,
            surrogate_name=surrogate_name,
            device=device,
            dtype=dtype,
            tested_model_revision=tested_model_revision,
            tested_tokenizer_revision=tested_tokenizer_revision,
            surrogate_model_revision=surrogate_model_revision,
            surrogate_tokenizer_revision=surrogate_tokenizer_revision,
            requested_device_map=None,
            resolved_device_map=None,
            generator_backend=generator_backend,
        )

    model_load_kwargs: dict[str, Any] = {
        "revision": tested_model_revision,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        model_load_kwargs["device_map"] = device_map
    if max_memory is not None:
        model_load_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(
        tested_llm_model_name,
        **model_load_kwargs,
    )
    if device_map is None:
        model = model.to(device)
    model = model.eval()
    device = _module_device(model.get_input_embeddings(), fallback=device)
    tested_model_revision = _resolved_revision(model)
    resolved_device_map = getattr(model, "hf_device_map", None)
    if isinstance(resolved_device_map, dict):
        resolved_device_map = {str(key): value for key, value in resolved_device_map.items()}
    else:
        resolved_device_map = None

    if surrogate_name == tested_llm_model_name:
        surrogate_tokenizer = tokenizer
        embed_weight = model.get_input_embeddings().weight.detach().to("cpu", torch.float32)
        surrogate_model_revision = tested_model_revision
        surrogate_tokenizer_revision = tested_tokenizer_revision
    else:
        surrogate_tokenizer = AutoTokenizer.from_pretrained(
            surrogate_name,
            revision=surrogate_model_revision,
        )
        surrogate_model = AutoModelForCausalLM.from_pretrained(
            surrogate_name,
            revision=surrogate_model_revision,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cpu").eval()
        embed_weight = surrogate_model.get_input_embeddings().weight.detach().to("cpu", torch.float32)
        surrogate_model_revision = _resolved_revision(surrogate_model)
        surrogate_tokenizer_revision = _resolved_revision(surrogate_tokenizer)
        del surrogate_model

    return ModelAssets(
        model=model,
        tokenizer=tokenizer,
        surrogate_tokenizer=surrogate_tokenizer,
        surrogate_embedding_weight=embed_weight,
        surrogate_name=surrogate_name,
        device=device,
        dtype=dtype,
        tested_model_revision=tested_model_revision,
        tested_tokenizer_revision=tested_tokenizer_revision,
        surrogate_model_revision=surrogate_model_revision,
        surrogate_tokenizer_revision=surrogate_tokenizer_revision,
        requested_device_map=device_map,
        resolved_device_map=resolved_device_map,
        generator_backend=generator_backend,
    )
