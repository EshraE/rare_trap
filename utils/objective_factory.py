"""Build ThinkTrap objectives from model and projection settings."""

from __future__ import annotations

import random

import numpy as np
import torch

from utils.config import PROJECTION_SEED
from utils.llm_generator import HFTextGenerator
from utils.model_assets import load_model_assets
from utils.projection_space import build_projection_space
from utils.thinktrap_objective import ThinkTrapObjective
from utils.template_probe import write_template_probe


CANDIDATE_MODE = "visible_nonspecial"
NOISE_K = 0
INPUT_MODE = "chat"
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 1.0


def build_obj(
    *,
    tested_llm_model_name: str,
    surrogate_token_model_name: str | None,
    tested_model_revision: str | None = None,
    surrogate_model_revision: str | None = None,
    sampling_seed: int,
    input_prompt_len: int,
    ss_sample_space_dim: int,
    threshold: int,
    llm_output_sampler: str,
    chat_template_thinking: str = "auto",
    chat_template_reasoning_effort: str | None = None,
    require_thinking_template_effect: bool = False,
    template_probe_output: str = "",
    generation_batch_size: int = 1,
    worst_case_count: int = 10,
    case_output_text_count: int = 2,
    raw_samples_path: str = "",
    objective_repeats: int = 1,
    objective_aggregation: str = "mean",
    performance_metric: str = "output_length",
    repetition_score_threshold: float = 0.9,
    projection_mode: str = "embedding-whitened",
    covariance_eps: float = 1e-5,
) -> ThinkTrapObjective:
    np.random.seed(int(sampling_seed))
    random.seed(int(sampling_seed))
    torch.manual_seed(int(sampling_seed))
    do_sample = str(llm_output_sampler) == "sample"
    max_new_tokens = int(threshold) + 3
    thinking_mode = str(chat_template_thinking)
    chat_template_kwargs: dict[str, object] = {}
    if thinking_mode == "enabled":
        chat_template_kwargs["enable_thinking"] = True
    elif thinking_mode == "disabled":
        chat_template_kwargs["enable_thinking"] = False
    if chat_template_reasoning_effort is not None:
        chat_template_kwargs["reasoning_effort"] = str(chat_template_reasoning_effort)

    assets = load_model_assets(
        tested_llm_model_name=tested_llm_model_name,
        surrogate_token_model_name=surrogate_token_model_name,
        tested_model_revision=tested_model_revision,
        surrogate_model_revision=surrogate_model_revision,
    )
    template_probe = write_template_probe(
        tokenizer=assets.tokenizer,
        model_name=str(tested_llm_model_name),
        model_revision=assets.tested_tokenizer_revision,
        active_kwargs=chat_template_kwargs,
        output_path=str(template_probe_output),
        require_thinking_effect=bool(require_thinking_template_effect),
    )
    print(f"[init] Device: {assets.device}  Dtype: {assets.dtype}")
    requested_generation_batch_size = max(1, int(generation_batch_size))
    effective_generation_batch_size = requested_generation_batch_size
    if assets.device.type == "mps":
        effective_generation_batch_size = min(requested_generation_batch_size, 8)

    projection_space = build_projection_space(
        tokenizer=assets.surrogate_tokenizer,
        embedding_weight=assets.surrogate_embedding_weight,
        input_prompt_len=int(input_prompt_len),
        ss_sample_space_dim=int(ss_sample_space_dim),
        noise_k=NOISE_K,
        candidate_mode=CANDIDATE_MODE,
        projection_mode=str(projection_mode),
        covariance_eps=float(covariance_eps),
        lookup_device=assets.device,
    )

    print(f"[init] Tested LLM model: {tested_llm_model_name}")
    print(f"[init] Surrogate token model: {assets.surrogate_name}")
    print(
        f"[init] tested_revision={assets.tested_model_revision or 'unresolved'} "
        f"| surrogate_revision={assets.surrogate_model_revision or 'unresolved'}"
    )
    print(
        f"[init] Prompt length: {projection_space.prompt_length} | Latent D: {projection_space.d_latent} "
        f"| SurrogatePool K: {projection_space.candidate_count}"
    )
    print(f"[init] candidate_filter={CANDIDATE_MODE} | embed_dim={projection_space.embed_dim}")
    print(f"[init] max_new_tokens={max_new_tokens} | llm_output_len_threshold={threshold}")
    print(
        f"[init] performance_metric={performance_metric} "
        f"| repetition_score_threshold={float(repetition_score_threshold):.6f}"
    )
    print(f"[init] projection_mode={projection_space.mode}")
    print(
        f"[init] projection_seed={PROJECTION_SEED} "
        f"| sampling_seed={int(sampling_seed)}"
    )
    print(f"[init] projection_lookup_device={projection_space.lookup_device}")
    print(
        f"[init] input_mode={INPUT_MODE} | llm_output_sampler={llm_output_sampler} "
        f"| temperature={SAMPLE_TEMPERATURE} | top_p={SAMPLE_TOP_P}"
    )
    print(f"[init] chat_template_thinking={thinking_mode}")
    print(
        f"[init] objective_repeats={int(objective_repeats)} "
        f"| objective_aggregation={objective_aggregation}"
    )
    print(f"[init] generation_batch_size={effective_generation_batch_size}")
    if effective_generation_batch_size != requested_generation_batch_size:
        print(
            f"[init-warning] requested_generation_batch_size={requested_generation_batch_size} "
            f"reduced_to={effective_generation_batch_size} on MPS to limit Metal graph "
            "and KV-cache memory growth."
        )

    generator = HFTextGenerator(
        model=assets.model,
        tokenizer=assets.tokenizer,
        device=assets.device,
        max_new_tokens=int(max_new_tokens),
        input_mode=INPUT_MODE,
        do_sample=bool(do_sample),
        temperature=SAMPLE_TEMPERATURE,
        top_p=SAMPLE_TOP_P,
        cache_implementation="static" if assets.device.type == "mps" else None,
        generation_chunk_size=512 if assets.device.type == "mps" else None,
        chat_template_kwargs=chat_template_kwargs or None,
    )
    print(
        f"[init] generation_cache="
        f"{generator.cache_implementation or 'transformers-default'}"
    )
    print(
        f"[init] generation_chunk_size="
        f"{generator.generation_chunk_size or 'one-shot'}"
    )

    return ThinkTrapObjective(
        generator=generator,
        projection_space=projection_space,
        threshold=int(threshold),
        generation_batch_size=effective_generation_batch_size,
        max_saved_records=max(10, int(worst_case_count) * 20),
        max_saved_output_text_records=max(0, int(case_output_text_count)),
        raw_samples_path=str(raw_samples_path),
        objective_repeats=max(1, int(objective_repeats)),
        objective_aggregation=str(objective_aggregation),
        performance_metric=str(performance_metric),
        repetition_score_threshold=float(repetition_score_threshold),
        run_metadata={
            "projection_seed": PROJECTION_SEED,
            "tested_llm_model": str(tested_llm_model_name),
            "surrogate_token_model": str(assets.surrogate_name),
            "tested_model_revision_requested": tested_model_revision,
            "surrogate_model_revision_requested": (
                surrogate_model_revision
                if assets.surrogate_name != str(tested_llm_model_name)
                else tested_model_revision
            ),
            "tested_model_revision": assets.tested_model_revision,
            "tested_tokenizer_revision": assets.tested_tokenizer_revision,
            "surrogate_model_revision": assets.surrogate_model_revision,
            "surrogate_tokenizer_revision": assets.surrogate_tokenizer_revision,
            "model_dtype": str(assets.dtype),
            "model_device": str(assets.device),
            "chat_template_thinking": thinking_mode,
            "chat_template_reasoning_effort": chat_template_reasoning_effort,
            "thinking_switch_changes_template_tokens": bool(
                template_probe["thinking_switch_changes_tokens"]
            ),
        },
    )
