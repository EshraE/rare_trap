#!/usr/bin/env python3
"""Production estimator workflow used by the public RareTrap runner."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from estimators_samplers import SubsetTraceObserver
from utils.estimator_runner import run_subset
from utils.objective_factory import build_obj
from utils.reporting import print_final_recap, save_all_evaluations, save_worst_cases
from utils.subset_trace_reporting import (
    save_subset_trace,
    save_subset_trace_checkpoint,
)


def run_experiment(args: argparse.Namespace) -> None:
    verbose = sys.stdout.isatty()
    obj = build_obj(
        tested_llm_model_name=args.tested_llm_model,
        surrogate_token_model_name=args.surrogate_token_model,
        tested_model_revision=args.tested_model_revision,
        surrogate_model_revision=args.surrogate_model_revision,
        sampling_seed=int(args.seed),
        input_prompt_len=int(args.input_prompt_len),
        ss_sample_space_dim=int(args.ss_sample_space_dim),
        threshold=int(args.llm_output_len_threshold),
        llm_output_sampler=str(args.llm_output_sampler),
        chat_template_thinking=str(args.chat_template_thinking),
        chat_template_reasoning_effort=args.chat_template_reasoning_effort,
        require_thinking_template_effect=bool(args.require_thinking_template_effect),
        template_probe_output=str(args.save_template_probe),
        generation_batch_size=int(args.generation_batch_size),
        worst_case_count=int(args.worst_case_count),
        case_output_text_count=int(args.case_output_text_count),
        raw_samples_path=str(args.save_raw_samples),
        objective_repeats=int(args.objective_repeats),
        objective_aggregation=str(args.objective_aggregation),
        performance_metric=str(args.performance_metric),
        repetition_score_threshold=float(args.repetition_score_threshold),
        projection_mode=str(args.projection_mode),
        covariance_eps=float(getattr(args, "covariance_eps", 1e-5)),
    )

    obj.initialize_raw_sample_stream(
        {
            "arguments": vars(args),
            "run_metadata": dict(obj.run_metadata),
        }
    )

    trace_observer = SubsetTraceObserver() if str(args.save_sampler_trace) else None
    completed_level_summaries: list[object] = []

    def checkpoint_completed_level(summary: object) -> None:
        completed_level_summaries.append(summary)
        save_all_evaluations(
            path=str(args.save_all_evaluations),
            obj=obj,
            args=args,
            result=None,
        )
        save_subset_trace_checkpoint(
            path=str(args.save_sampler_trace),
            observer=trace_observer,
            completed_level_summaries=completed_level_summaries,
        )

    level_complete_callback = (
        checkpoint_completed_level
        if str(args.save_all_evaluations) or str(args.save_sampler_trace)
        else None
    )

    result = run_subset(
        obj=obj,
        samples_per_level=int(args.ss_samples_per_level),
        conditional_prob=float(args.ss_conditional_prob),
        max_levels=int(args.ss_max_levels),
        seed=int(args.seed),
        sampler_progress=bool(args.sampler_progress),
        verbose=verbose,
        trace_observer=trace_observer,
        level_complete_callback=level_complete_callback,
    )
    save_worst_cases(
        path=str(args.save_worst_cases),
        obj=obj,
        args=args,
        case_count=int(args.worst_case_count),
        result=result,
    )
    save_all_evaluations(
        path=str(args.save_all_evaluations),
        obj=obj,
        args=args,
        result=result,
    )
    save_subset_trace(
        path=str(args.save_sampler_trace),
        observer=trace_observer,
        obj=obj,
        args=args,
        result=result,
    )
    print_final_recap("subset", result, obj.threshold, obj=obj)
