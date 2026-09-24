"""vLLM adapter for large or quantized models."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from utils.llm_generator import GenerationResult


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1.")
    return raw == "1"


def _env_optional(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


def build_vllm_engine(
    *,
    model_name: str,
    revision: Optional[str],
    sampling_seed: int,
    max_new_tokens: int,
    generation_batch_size: int,
) -> Any:
    """Build a vLLM engine with auditable environment overrides."""

    from vllm import LLM

    tensor_parallel_size = _env_int("THINKTRAP_VLLM_TENSOR_PARALLEL_SIZE", 1)
    pipeline_parallel_size = _env_int("THINKTRAP_VLLM_PIPELINE_PARALLEL_SIZE", 1)
    gpu_memory_utilization = _env_float("THINKTRAP_VLLM_GPU_MEMORY_UTILIZATION", 0.90)
    max_model_len = _env_int(
        "THINKTRAP_VLLM_MAX_MODEL_LEN",
        max(22048, int(max_new_tokens) + 2048),
    )
    max_num_batched_tokens = _env_int(
        "THINKTRAP_VLLM_MAX_NUM_BATCHED_TOKENS",
        max(32768, max_model_len),
    )
    quantization = _env_optional(
        "THINKTRAP_VLLM_QUANTIZATION",
        "modelopt_fp4",
    )
    engine_kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": model_name,
        "revision": revision,
        "tokenizer_revision": revision,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "dtype": os.environ.get("THINKTRAP_VLLM_DTYPE", "bfloat16").strip(),
        "quantization": quantization,
        "enforce_eager": _env_bool("THINKTRAP_VLLM_ENFORCE_EAGER", True),
        "seed": int(sampling_seed),
        "max_model_len": max_model_len,
        "max_num_seqs": max(1, int(generation_batch_size)),
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_dtype": os.environ.get(
            "THINKTRAP_VLLM_KV_CACHE_DTYPE",
            "bfloat16",
        ).strip(),
        "attention_config": {
            "backend": os.environ.get(
                "THINKTRAP_VLLM_ATTENTION_BACKEND",
                "TRITON_ATTN",
            ).strip()
        },
        "enable_prefix_caching": _env_bool(
            "THINKTRAP_VLLM_ENABLE_PREFIX_CACHING",
            True,
        ),
        "disable_custom_all_reduce": True,
        "trust_remote_code": False,
    }
    if _env_bool("THINKTRAP_VLLM_LANGUAGE_MODEL_ONLY", False):
        engine_kwargs["language_model_only"] = True
    backend_defaults = quantization == "modelopt_fp4"
    optional_backends = {
        "moe_backend": _env_optional(
            "THINKTRAP_VLLM_MOE_BACKEND",
            "humming" if backend_defaults else None,
        ),
        "linear_backend": _env_optional(
            "THINKTRAP_VLLM_LINEAR_BACKEND",
            "humming" if backend_defaults else None,
        ),
        "mamba_backend": _env_optional(
            "THINKTRAP_VLLM_MAMBA_BACKEND",
            "triton" if backend_defaults else None,
        ),
        "mamba_cache_mode": _env_optional(
            "THINKTRAP_VLLM_MAMBA_CACHE_MODE",
            "align" if backend_defaults else None,
        ),
    }
    engine_kwargs.update(
        {name: value for name, value in optional_backends.items() if value is not None}
    )
    return LLM(**engine_kwargs)


@dataclass
class VLLMTextGenerator:
    engine: Any
    tokenizer: Any
    max_new_tokens: int
    input_mode: str
    do_sample: bool
    temperature: float
    top_p: float
    chat_template_kwargs: Optional[dict[str, object]] = None

    def _render_prompt(self, prompt_text: str) -> str:
        if self.input_mode == "chat" and hasattr(self.tokenizer, "apply_chat_template"):
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **dict(self.chat_template_kwargs or {}),
                )
            )
        return str(prompt_text)

    def generate_batch(
        self,
        prompt_texts: list[str],
        output_token_callbacks: Optional[list[Callable[[int], None]]] = None,
    ) -> list[GenerationResult]:
        if not prompt_texts:
            raise ValueError("prompt_texts must not be empty.")
        from vllm import SamplingParams

        rendered_prompts = [self._render_prompt(prompt_text) for prompt_text in prompt_texts]
        params = SamplingParams(
            n=1,
            temperature=float(self.temperature) if self.do_sample else 0.0,
            top_p=float(self.top_p) if self.do_sample else 1.0,
            max_tokens=int(self.max_new_tokens),
            skip_special_tokens=False,
        )
        started = time.perf_counter()
        request_outputs = self.engine.generate(
            rendered_prompts,
            params,
            use_tqdm=False,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        if len(request_outputs) != len(prompt_texts):
            raise RuntimeError(
                f"vLLM returned {len(request_outputs)} outputs for {len(prompt_texts)} prompts."
            )
        results: list[GenerationResult] = []
        for prompt_text, request_output in zip(prompt_texts, request_outputs):
            if len(request_output.outputs) != 1:
                raise RuntimeError("Expected exactly one vLLM completion per prompt.")
            completion = request_output.outputs[0]
            completion_token_ids = [int(token_id) for token_id in completion.token_ids]
            model_input_token_ids = [
                int(token_id) for token_id in (request_output.prompt_token_ids or [])
            ]
            completion_length = len(completion_token_ids)
            results.append(
                GenerationResult(
                    prompt_text=str(prompt_text),
                    input_token_count=len(model_input_token_ids),
                    completion_length=completion_length,
                    completion_text=str(completion.text),
                    completion_token_ids=completion_token_ids,
                    elapsed_sec=float(elapsed),
                    tokens_per_second=float(completion_length) / float(elapsed),
                    hit_cap=bool(completion_length >= int(self.max_new_tokens)),
                    model_input_token_ids=model_input_token_ids,
                )
            )
        if output_token_callbacks is not None:
            if len(output_token_callbacks) != len(results):
                raise ValueError(
                    "output_token_callbacks must match the number of prompts."
                )
            for callback, result in zip(output_token_callbacks, results):
                callback(int(result.completion_length))
        return results

    def generate(
        self,
        prompt_text: str,
        output_token_callback: Optional[Callable[[int], None]] = None,
    ) -> GenerationResult:
        result = self.generate_batch([prompt_text])[0]
        if output_token_callback is not None:
            output_token_callback(int(result.completion_length))
        return result
