"""Small public interface; experimental internals are not command-line knobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from raretrap.configuration import (
    PAPER_PROJECTION_ARMS, default_experiment, experiments, models, preset,
    validate_public_projection, custom_model, implementation_settings,
)
from raretrap.lifecycle import run_log, redact


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raretrap", description="RareTrap paper experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="List pinned model checkpoints")
    listing = sub.add_parser("presets", help="List recorded experiment configurations")
    listing.add_argument("--group", choices=("matched", "additional", "validation", "mc", "extreme"))

    projection = sub.add_parser("projection", help="Paired self-surrogate projection comparison")
    _model_options(projection, [k for k in models() if k not in
        {"qwen3-0.6b", "qwen2.5-0.5b", "qwen3.8-27b", "nemotron3.5-30b"}])
    projection.add_argument("--samples", type=int, default=100, help="IID points per projection arm")
    projection.add_argument("--seed", type=int, default=20260806, help="IID latent-sampling seed only")

    run = sub.add_parser("run", help="Execute a paper configuration without parameter overrides")
    run.add_argument("--preset", required=True)

    estimate = sub.add_parser("estimate", help="Estimate length or repetition tail probability")
    _model_options(estimate, models())
    estimate.add_argument("--surrogate", choices=("self", "qwen3-0.6b", "qwen2.5-0.5b"), default="self")
    estimate.add_argument("--metric", choices=("length", "repetition"), default="length")
    estimate.add_argument("--threshold", type=int, default=20000, help="Length threshold; cap is this value plus three")
    estimate.add_argument("--repetition-threshold", type=float, default=0.99)
    estimate.add_argument("--samples-per-level", type=int, default=1000)
    estimate.add_argument("--conditional-prob", type=float, default=0.1)
    estimate.add_argument("--max-levels", type=int, default=5, help="Conditional levels; zero selects IID Monte Carlo")
    estimate.add_argument("--latent-dim", type=int, default=200)
    estimate.add_argument("--prompt-length", type=int, default=40)
    estimate.add_argument("--seed", type=int, default=1010, help="Estimator/MCMC seed only")
    for command in (projection, run, estimate):
        command.add_argument("--output", type=Path, required=True, help="New output directory (never overwritten)")
        command.add_argument("--dry-run", action="store_true", help="Print settings without downloading or evaluating a model")

    summarize = sub.add_parser("summarize", help="Summarize generated projection or estimator outputs")
    summarize.add_argument("--input", type=Path, nargs="+", required=True)
    status = sub.add_parser("status", help="Read a run's live status without loading a model")
    status.add_argument("--input", type=Path, required=True)
    dataset = sub.add_parser("export-dataset", help="Rebuild a dataset snapshot from saved raw evaluations")
    dataset.add_argument("--input", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report", help="Export tables, level diagnostics, and optional figures/cases")
    report.add_argument("--input", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--figures", action="store_true", help="Requires the analysis extra")
    report.add_argument("--cases", action="store_true", help="Export all raw prompt/response records; can be large")
    return parser


def _model_options(parser, choices):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=choices, default="deepseek8b", help="Pinned paper model alias")
    selection.add_argument("--hf-model", help="Custom Hugging Face model ID (not a paper reproduction)")
    parser.add_argument("--revision", help="Required full commit hash for --hf-model")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), help="Custom model only; default float16")
    parser.add_argument("--thinking", choices=("auto", "enabled", "disabled"), help="Custom model only; default auto")


def _target(args):
    if args.hf_model:
        return custom_model(args.hf_model, args.revision, args.dtype or "float16", args.thinking or "auto")
    if args.revision is not None or args.dtype is not None or args.thinking is not None:
        raise ValueError("Revision, precision, and thinking overrides require --hf-model; paper aliases are pinned.")
    return models()[args.model]


def resolve_experiment(args: argparse.Namespace) -> dict:
    if args.command == "run":
        spec = preset(args.preset)
    else:
        spec = default_experiment(args.hf_model or args.model, args.surrogate, target=_target(args))
        if spec["backend"] == "vllm":
            raise ValueError("Use an additional-model preset to preserve its recorded vLLM settings.")
        spec["arguments"].update(
            seed=args.seed, input_prompt_len=args.prompt_length,
            ss_sample_space_dim=args.latent_dim, ss_samples_per_level=args.samples_per_level,
            ss_conditional_prob=args.conditional_prob, ss_max_levels=args.max_levels,
            llm_output_len_threshold=args.threshold, repetition_score_threshold=args.repetition_threshold,
            performance_metric="output_length" if args.metric == "length" else "repetition_score",
        )
    _validate_seed(int(spec["arguments"]["seed"]))
    validate_public_projection(spec["arguments"])
    return spec


def _validate_seed(seed: int) -> None:
    # Production also seeds NumPy's legacy global RNG; keep its actual range.
    if not 0 <= seed < 2**32:
        raise ValueError("The sampling seed must be between 0 and 4294967295.")


def _environment(settings: dict) -> None:
    # Prevent stale research environment settings from silently modifying a preset.
    for key in list(os.environ):
        if key.startswith("THINKTRAP_"):
            del os.environ[key]
    os.environ.update(settings)
    os.environ["THINKTRAP_DEVICE_ORDER"] = "cuda"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Paper model experiments require CUDA. Use --dry-run or pytest on CPU.")
    from utils.model_assets import pick_device
    if pick_device().type != "cuda":
        raise RuntimeError("A previously imported device configuration selects a non-CUDA device. "
                           "Start a fresh process without a THINKTRAP_DEVICE_ORDER override.")


def _run_estimator(spec: dict, output: Path) -> None:
    validate_public_projection(spec["arguments"])
    _environment(spec["environment"])
    from raretrap import _experiment
    from utils.cli import validate_args

    if spec["backend"] == "vllm":
        from raretrap.backends.objective_factory import build_obj
        from raretrap.backends import model_assets as asset_module
    else:
        from utils.objective_factory import build_obj
        from utils import model_assets as asset_module
    from raretrap.model_loading import local_model_code_only
    previous_factory = _experiment.build_obj
    settings = implementation_settings(spec["arguments"])
    settings.update(
        require_thinking_template_effect=settings["chat_template_thinking"] == "enabled",
        save_template_probe=str(output / "template_probe.json"),
        save_raw_samples=str(output / "raw_samples.jsonl"),
        save_sampler_trace=str(output / "sampler_trace"),
        save_all_evaluations=str(output / "all_evaluations.json"),
        save_worst_cases=str(output / "cases.json"),
        worst_case_count=100, case_output_text_count=100, sampler_progress=True,
    )
    namespace = argparse.Namespace(**settings)
    validate_args(make_parser(), namespace)
    namespace.projection_mode = spec["arguments"]["projection_mode"]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with run_log(output):
        try:
            from raretrap.artifacts import atomic_json
            atomic_json(output / "configuration.json", spec)
            _require_cuda()
            def paper_factory(**kwargs):
                with local_model_code_only(asset_module):
                    return build_obj(**implementation_settings(kwargs))
            factory = paper_factory
            if spec.get("custom_model"):
                from raretrap.custom_models import preflight, ContextCheckedGenerator
                custom = dict(model=settings["tested_llm_model"], revision=settings["tested_model_revision"],
                              thinking=settings["chat_template_thinking"])
                checks = preflight(custom, settings["llm_output_len_threshold"] + 3)
                atomic_json(output / "model_preflight.json", checks)
                def checked_factory(**kwargs):
                    objective = paper_factory(**kwargs)
                    objective.generator = ContextCheckedGenerator(objective.generator, checks["context_limit"])
                    return objective
                factory = checked_factory
            _experiment.build_obj = factory
            _experiment.run_experiment(namespace)
        finally:
            _experiment.build_obj = previous_factory
    (output / "DONE").touch(exist_ok=False)


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            for alias, item in models().items():
                print(f"{alias:20s} {item['model']}  {item['revision']}")
            return
        if args.command == "presets":
            for name, item in experiments().items():
                if args.group is None or item["group"] == args.group:
                    note = "fixed-map rerun of historical settings" if not item["recorded_map_matches_release"] else "recorded fixed map"
                    print(f"{name}  [{note}]")
            return
        if args.command == "summarize":
            from raretrap.summarize import summarize
            summarize(args.input)
            return
        if args.command == "status":
            import time
            status = json.loads((args.input / "status.json").read_text())
            status["heartbeat_age_sec"] = max(0, time.time() - status["updated"])
            status["stale"] = status.get("state") == "running" and status["heartbeat_age_sec"] > 60
            print(json.dumps(status, indent=2))
            return
        if args.output.exists():
            raise ValueError(f"Output directory already exists: {args.output}. Choose a new directory.")
        if args.command == "report":
            from raretrap.analysis import report
            report(args.input, args.output, figures=args.figures, cases=args.cases)
            return
        if args.command == "export-dataset":
            from raretrap.artifacts import export_dataset
            print(json.dumps(export_dataset(args.input / "raw_samples.jsonl", args.output), indent=2))
            return
        if args.command == "projection":
            if args.samples < 1:
                raise ValueError("Samples must be positive.")
            _validate_seed(args.seed)
            model = _target(args)
            configuration = dict(model=model, samples=args.samples, sampling_seed=args.seed,
                                 cap=20003, prompt_length=40, latent_dimension=200,
                                 arms=list(PAPER_PROJECTION_ARMS))
            if args.dry_run:
                print(json.dumps(configuration, indent=2))
                return
            _environment({"THINKTRAP_MODEL_DTYPE": model["dtype"]})
            from raretrap._projection import run_projection
            run_projection(model=model, output=args.output, samples=args.samples, seed=args.seed, require_cuda=True)
            (args.output / "DONE").touch(exist_ok=False)
            return
        spec = resolve_experiment(args)
        from utils.cli import validate_args
        validation_settings = dict(implementation_settings(spec["arguments"]), worst_case_count=100, case_output_text_count=100)
        validate_args(parser, argparse.Namespace(**validation_settings))
        if spec.get("custom_model"):
            print("NOTE: custom checkpoint; this is not a reproduction of a paper result.", file=sys.stderr)
        elif not spec["recorded_map_matches_release"]:
            print("NOTE: this uses the release's fixed map, not the historical projection realization.", file=sys.stderr)
        if args.dry_run:
            print(json.dumps(spec, indent=2))
            return
        _run_estimator(spec, args.output)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"raretrap: {redact(str(exc))}\n")
