#!/usr/bin/env python3
"""Lossless paired ablation, reusing the audited projection and generation code."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.run_projection_ablation import (
    _prompt_for_arm, _hash_text, _summary, _write_outputs,
)
from utils.model_assets import load_model_assets
from utils.llm_generator import HFTextGenerator
from utils.repetition_metrics import repetition_metrics
from utils.reproducibility import runtime_metadata
from utils.config import PROJECTION_SEED
from raretrap.lifecycle import run_log
from raretrap.configuration import PAPER_PROJECTION_ARMS
from raretrap.artifacts import atomic_json
from raretrap.projection import build_paper_projections
from raretrap.terminology import PROJECTION_EQUATIONS
from raretrap.model_loading import local_model_code_only

ARMS = PAPER_PROJECTION_ARMS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_projection(*, model: dict, output: Path, samples: int = 100, seed: int = 20260806,
                   require_cuda: bool = False):
    """Run the recorded two-arm, self-surrogate experiment."""
    root = Path(output)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    with run_log(root):
        # The public CLI checks CUDA inside the logged lifecycle so failures
        # before model loading are retained alongside the requested settings.
        atomic_json(root / "configuration.json", dict(model=model, samples=samples, sampling_seed=seed,
                    cap=20003, latent_dimension=200, prompt_length=40, arms=list(ARMS)))
        if require_cuda:
            from raretrap.cli import _require_cuda
            _require_cuda()
        if model.get("custom"):
            from raretrap.custom_models import preflight
            atomic_json(root / "model_preflight.json", preflight(model, 20003))
        _run_projection(model=model, output=root, samples=samples, seed=seed)


def _run_projection(*, model: dict, output: Path, samples: int, seed: int):
    job = model
    root = Path(output)
    plan = {"samples": samples, "max_new_tokens": 20003,
            "projection_seed": PROJECTION_SEED, "latent_seed": seed}
    torch.set_num_threads(4)
    random.seed(plan["latent_seed"])
    np.random.seed(plan["latent_seed"])
    torch.manual_seed(plan["latent_seed"])
    atomic_json(root / "progress.json", {"stage": "loading_model", "updated": time.time(), "job": job})
    from utils import model_assets as asset_module
    with local_model_code_only(asset_module):
        assets = load_model_assets(tested_llm_model_name=job["model"],
            surrogate_token_model_name=job["model"], tested_model_revision=job["revision"],
            surrogate_model_revision=job["revision"])
    template_kwargs = {"enable_thinking": True} if job["thinking"] == "enabled" else (
        {"enable_thinking": False} if job["thinking"] == "disabled" else None)
    generator = HFTextGenerator(model=assets.model, tokenizer=assets.tokenizer,
        device=assets.device, max_new_tokens=plan["max_new_tokens"], input_mode="chat",
        do_sample=False, temperature=1.0, top_p=1.0, chat_template_kwargs=template_kwargs)
    probe = {"thinking": job["thinking"], "chat_template_kwargs": template_kwargs,
        "input_ids": generator._prompt_to_cpu_inputs("Hello.")["input_ids"].tolist()}
    if job["thinking"] == "enabled":
        messages = [{"role": "user", "content": "Hello."}]
        on = assets.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        off = assets.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        probe.update(enabled_text=on, disabled_text=off, thinking_has_effect=on != off)
        if on == off:
            raise RuntimeError("Requested thinking template has no effect")
    atomic_json(root / "template_probe.json", probe)
    atomic_json(root / "progress.json", {"stage": "building_projection", "updated": time.time()})
    arms = build_paper_projections(tokenizer=assets.surrogate_tokenizer,
        embedding_weight=assets.surrogate_embedding_weight, S=40, D=200,
        epsilon=1e-5, lookup_device=torch.device("cpu"))
    latents = np.random.default_rng(plan["latent_seed"]).standard_normal((plan["samples"], 200)).astype(np.float32)
    np.save(root / "latents.npy", latents, allow_pickle=False)
    first = arms[ARMS[0]].projection
    torch.save({"candidate_ids": first.token_ids.cpu(), "candidate_embeddings": first.embed_table.cpu(),
        "candidate_norms": first.embed_norms.cpu()}, root / "candidates.pt")
    for name, arm in arms.items():
        p = arm.projection
        torch.save({key: None if getattr(p, key) is None else getattr(p, key).cpu()
            for key in ("bias", "matrix", "shared_matrix", "slot_transforms")}, root / (name + "_factors.pt"))
    atomic_json(root / "tensor_sha256.json", {p.name: digest(p) for p in root.iterdir() if p.suffix in (".pt", ".npy")})
    metadata = {"complete": False, "model": job["model"], "surrogate": job["model"],
        "model_revision": job["revision"], "samples": plan["samples"], "arms": list(ARMS),
        "artifact_schema_version": 3,
        "arm_definitions": {name: PROJECTION_EQUATIONS[name] for name in ARMS},
        "threshold": 20000, "max_new_tokens": 20003, "latent_dim": 200, "prompt_length": 40,
        "projection_seed": plan["projection_seed"], "latent_seed": plan["latent_seed"],
        "candidate_filter": "visible_nonspecial", "candidate_count": int(first.token_ids.numel()),
        "surrogate_embedding_dim": int(first.embed_table.shape[1]), "covariance_eps": 1e-5,
        "projection_lookup_device": "cpu", "paired_latent_samples": True,
        "generation_batch_size": 1, "thinking": job["thinking"], "dtype": str(assets.dtype),
        "runtime": runtime_metadata(assets.device), "plan": plan,
        "special_token_ids": sorted(int(i) for i in assets.tokenizer.all_special_ids)}
    atomic_json(root / "metadata.json", metadata)
    atomic_json(root / "generation_config.json", assets.model.generation_config.to_dict())
    special = set(int(i) for i in assets.tokenizer.all_special_ids)
    records = []
    arm_records = {name: [] for name in ARMS}
    token_rng = np.random.default_rng(0)  # Unused by both projection arms.
    with (root / "raw_samples.jsonl").open("x", encoding="utf-8") as raw:
        for name, arm in arms.items():
            for index, latent in enumerate(latents):
                prompt = _prompt_for_arm(arm=arm, z=latent, rng=token_rng)
                input_length = int(generator._prompt_to_cpu_inputs(prompt.text)["input_ids"].numel())
                config = getattr(assets.model.config, "text_config", assets.model.config)
                context = getattr(config, "max_position_embeddings", None)
                if context and input_length + plan["max_new_tokens"] > context:
                    raise RuntimeError(f"Context budget exceeded: {input_length}+20003 > {context}")
                last_progress = [0.0]
                def progress(count):
                    now = time.time()
                    if now - last_progress[0] >= 15:
                        atomic_json(root / "progress.json", {"stage": "generating", "updated": now,
                            "arm": name, "sample_index": index, "completed": len(records), "output_tokens_in_flight": count})
                        print(f"[generating] arm={name} index={index} tokens={count}", flush=True)
                        last_progress[0] = now
                result = generator.generate(prompt.text, output_token_callback=progress)
                repetition = repetition_metrics(result.completion_token_ids, special_token_ids=special)
                record = {"arm": name, "sample_index": index, "output_tokens": result.completion_length,
                    "repetition_score": repetition.repetition_score, "rep_2": repetition.rep_2,
                    "rep_3": repetition.rep_3, "rep_4": repetition.rep_4,
                    "success": result.completion_length >= 20000, "hit_cap": result.hit_cap,
                    "input_token_count": result.input_token_count, "elapsed_sec": result.elapsed_sec,
                    "tokens_per_second": result.tokens_per_second, "prompt_hash": _hash_text(prompt.text),
                    "output_hash": _hash_text(result.completion_text), "surrogate_prompt_token_count": len(prompt.token_ids),
                    "prompt_char_count": len(prompt.text), "token_ids": " ".join(map(str, prompt.token_ids)),
                    "prompt_text": prompt.text, "completion_text": result.completion_text}
                lossless = dict(record, latent_vector_float32=latent.tolist(),
                    latent_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),
                    surrogate_token_ids=prompt.token_ids, model_input_token_ids=result.model_input_token_ids,
                    completion_token_ids=result.completion_token_ids)
                raw.write(json.dumps(lossless, ensure_ascii=False) + "\n")
                raw.flush()
                os.fsync(raw.fileno())
                records.append(record)
                arm_records[name].append(record)
                payload = {"metadata": metadata, "summaries": {a: _summary(r, 20000) for a, r in arm_records.items()}, "records": records}
                _write_outputs(output_json=root / "summary.json", output_csv=root / "samples.csv", payload=payload, records=records)
                print(f"[sample] {len(records)}/{2 * samples} arm={name} index={index} length={result.completion_length} repetition={repetition.repetition_score:.6f}", flush=True)
    metadata["complete"] = True
    payload["metadata"] = metadata
    _write_outputs(output_json=root / "summary.json", output_csv=root / "samples.csv", payload=payload, records=records)
    atomic_json(root / "metadata.json", metadata)
    atomic_json(root / "progress.json", {"stage": "complete", "updated": time.time(), "completed": len(records)})
