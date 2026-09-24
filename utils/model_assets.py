"""Model and surrogate asset loading for ThinkTrap experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import DEVICE_ORDER


@dataclass
class ModelAssets:
    model: AutoModelForCausalLM
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


def load_model_assets(
    *,
    tested_llm_model_name: str,
    surrogate_token_model_name: Optional[str],
    tested_model_revision: Optional[str] = None,
    surrogate_model_revision: Optional[str] = None,
) -> ModelAssets:
    device = pick_device()
    dtype = choose_model_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        tested_llm_model_name,
        revision=tested_model_revision,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Target tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        tested_llm_model_name,
        revision=tested_model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    tested_model_revision = _resolved_revision(model)
    tested_tokenizer_revision = _resolved_revision(tokenizer)

    surrogate_name = surrogate_token_model_name or tested_llm_model_name
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
    )
