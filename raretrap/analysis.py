"""Source-only tools for exporting generated evidence, without pooling runs."""
from __future__ import annotations

from collections import defaultdict
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np

from raretrap.summarize import estimator_row, is_fixed_budget_mc, projection_rows, print_tables
from raretrap.terminology import projection_label, projection_name
from raretrap.artifacts import jsonl_records


def records(path, *, committed_only=False, byte_limit=None):
    yield from jsonl_records(path, committed_only=committed_only, byte_limit=byte_limit)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def latex_escape(value):
    translations = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "→": r"$\rightarrow$", "—": "--"}
    return "".join(translations.get(c, c) for c in str(value))


def write_latex(path, rows, columns):
    if not rows:
        return
    lines = [r"% Requires booktabs. Partial/nonfinal rows remain labeled.",
             r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(latex_escape(title) for _, title in columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(key, "")) for key, _ in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def clopper_pearson(hits, count, confidence=0.95):
    """Two-sided fixed-budget binomial interval, including zero/all hits."""
    from scipy.stats import beta
    if type(hits) is not int or type(count) is not int or count < 1 or not 0 <= hits <= count or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence level")
    tail = (1 - confidence) / 2
    return (0.0 if hits == 0 else float(beta.ppf(tail, hits, count - hits + 1)),
            1.0 if hits == count else float(beta.ppf(1 - tail, hits + 1, count - hits)))


def estimator_details(root):
    metadata = json.loads((root / "cases.json").read_text())["metadata"]
    config_path = root / "configuration.json"
    if config_path.exists():
        metadata["release_environment"] = json.loads(config_path.read_text()).get("environment", {})
    result = metadata["estimator_result"]
    row = estimator_row(root)
    row.update(seed=metadata.get("seed"), method=result["probability_method"],
               target_reached=result["target_reached"], stop_reason=result["stop_reason"],
               cap=metadata.get("max_new_tokens"), ci_lower=None, ci_upper=None)
    # A checkpoint selected by its success count is not a fixed-budget IID design.
    fixed_mc = is_fixed_budget_mc(metadata)
    row["fixed_budget_mc"] = fixed_mc
    if fixed_mc:
        n, hits = int(result["final_population_size"]), int(result["final_failure_count"])
        row.update(mc_hits=hits, mc_samples=n)
        try:
            row["ci_lower"], row["ci_upper"] = clopper_pearson(hits, n)
            row["ci_method"] = "95% Clopper-Pearson (fixed-budget IID only)"
        except ImportError:
            row["ci_method"] = "unavailable: install raretrap[analysis]"
    else:
        row["ci_method"] = "not computed: conditional or checkpoint-selected sample"
    return row, metadata


def level_rows(root, run):
    path = root / "sampler_trace/population_slots.jsonl"
    if not path.exists():
        return []
    levels = defaultdict(list)
    for row in records(path):
        levels[int(row["subset_level"])].append(row)
    result = []
    for level, slots in sorted(levels.items()):
        item = dict(run=run, subset_level=level, population_slots=len(slots))
        for metric in ("output_tokens", "repetition_score", "performance_score"):
            values = [float(s[metric]) for s in slots if s.get(metric) is not None]
            item[metric + "_observations"] = len(values)
            if values:
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite population score in {root.name}")
                for q, value in zip((25, 50, 75, 95), np.percentile(values, [25, 50, 75, 95])):
                    item[f"{metric}_p{q}"] = float(value)
        for field in ("state_id", "input_hash", "output_hash"):
            values = [s[field] for s in slots if s.get(field) is not None]
            item["unique_" + field] = len(set(values))
        result.append(item)
    return result


def evaluation_counts(level_sampler):
    """Derive model-query counts without relabeling the preserved raw history."""
    if not level_sampler:
        return []
    previous = 0
    counts = []
    for item in sorted(level_sampler.values(), key=lambda row: int(row["level"])):
        cumulative = item["evaluations_cumulative"]
        if type(cumulative) is not int or cumulative < previous:
            raise ValueError("Cumulative evaluation counts must be nondecreasing integers")
        counts.append(dict(level=int(item["level"]),
                           model_evaluations=cumulative - previous,
                           model_evaluations_cumulative=cumulative))
        previous = cumulative
    return counts


def export_cases(root, destination, *, byte_limit=None):
    """All observed evaluations, not just selected successes or worst cases."""
    source = root / "raw_samples.jsonl"
    if not source.exists():
        raise ValueError(f"Missing raw samples in {root}")
    count = 0
    for row in records(source, committed_only=True, byte_limit=byte_limit):
        if row.get("record_type") == "metadata":
            continue
        # Folder names never derive from untrusted model text or recorded paths.
        folder = destination / f"case_{count:06d}"
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "record.json", row)
        prompt = row.get("prompt_text", row.get("input_text", ""))
        (folder / "input.txt").write_text(prompt, encoding="utf-8")
        if "generations" in row:
            for j, generation in enumerate(row["generations"]):
                (folder / f"output_{j:03d}.txt").write_text(generation["completion_text"], encoding="utf-8")
        else:
            (folder / "output_000.txt").write_text(row["completion_text"], encoding="utf-8")
        count += 1
    return count


def plot_outputs(root, destination, populations, *, byte_limit=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{stem}.{extension}", dpi=180)
        plt.close(fig)

    if (root / "metadata.json").exists():
        values = defaultdict(list)
        for row in records(root / "raw_samples.jsonl", committed_only=True, byte_limit=byte_limit):
            values[projection_name(row["arm"])].append(
                {key: row[key] for key in ("output_tokens", "repetition_score")})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for ax, metric in zip(axes, ("output_tokens", "repetition_score")):
            for arm, rows in values.items():
                ordered = np.sort([r[metric] for r in rows])
                ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", label=f"{projection_label(arm)} (n={len(rows)})")
            ax.set(xlabel=metric, ylabel="Empirical cumulative fraction")
            ax.legend(fontsize=7)
        save(fig, "projection_ecdf")
    elif populations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for ax, metric in zip(axes, ("output_tokens", "repetition_score")):
            available = [p for p in populations if metric + "_p50" in p]
            x = [p["subset_level"] for p in available]
            ax.plot(x, [p[metric + "_p50"] for p in available], "o-", label="P50")
            ax.fill_between(x, [p[metric + "_p25"] for p in available],
                            [p[metric + "_p75"] for p in available], alpha=0.2, label="P25–P75")
            ax.set(xlabel="Conditional level (initial = 0)", ylabel=metric)
            ax.legend()
        save(fig, "retained_population_quantiles")


def replication_summaries(entries):
    """Descriptive seed-to-seed SD, never an MCMC binomial standard error."""
    grouped = defaultdict(list)
    fields = ("tested_llm_model", "surrogate_token_model", "tested_model_revision",
              "surrogate_model_revision", "projection_seed", "projection_mode", "covariance_eps",
              "candidate_filter", "candidate_count", "ss_sample_space_dim", "input_prompt_len",
              "performance_metric", "performance_threshold", "max_new_tokens", "model_dtype",
              "chat_template_thinking", "chat_template_reasoning_effort", "generation_batch_size",
              "llm_output_sampler", "objective_repeats", "objective_aggregation",
              "ss_samples_per_level", "ss_max_levels", "temperature", "top_p",
              "tested_tokenizer_revision", "surrogate_tokenizer_revision", "release_environment")
    for row, metadata in entries:
        signature = {key: metadata.get(key) for key in fields}
        runtime = metadata.get("runtime", {})
        signature["runtime"] = {k: runtime.get(k) for k in ("packages", "torch_build", "accelerator")}
        signature["conditional_factors"] = sorted(set(metadata["estimator_result"].get("level_probabilities", [])))
        signature["method"] = row["method"]
        signature["backend_settings"] = {k: v for k, v in metadata.items()
                                         if k.startswith("vllm_") or k == "generator_backend"}
        key = json.dumps(signature, sort_keys=True)
        grouped[key].append(row)
    output = []
    for key, rows in grouped.items():
        seeds = [r["seed"] for r in rows]
        comparable = all(r["status"] == "final" and r["method"] == "subset_simulation" for r in rows)
        independent_seeds = None not in seeds and len(set(seeds)) == len(seeds)
        probabilities = [r["probability"] for r in rows]
        output.append(dict(configuration=json.loads(key), runs=[r["run"] for r in rows],
            seeds=seeds, count=len(rows), probabilities=probabilities,
            distinct_seeds=independent_seeds,
            mean=float(np.mean(probabilities)) if comparable and independent_seeds else None,
            sample_sd=float(np.std(probabilities, ddof=1)) if comparable and independent_seeds and len(rows) > 1 else None,
            note="Descriptive across-run variation; not a convergence test. No pooling of MC chunks."))
    return output


def plot_probabilities(rows, destination):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, max(3, .48 * len(rows))))
    for i, row in enumerate(rows):
        p = row["probability"]
        if row["ci_lower"] is not None:
            ax.errorbar(p, i, xerr=[[p - row["ci_lower"]], [row["ci_upper"] - p]], fmt="s", color="black", capsize=3)
        else:
            ax.plot(p, i, "o" if row["status"] == "final" else "x", color="tab:blue")
    ax.set_yticks(range(len(rows)), [f"{r['run']} | {r['model'].split('/')[-1]} | {r['metric']} ≥ {r['threshold']:g} | n-eval={r['evaluations']}"
                                   for r in rows], fontsize=7)
    ax.set(xlabel="Estimated probability (linear scale retains zero estimates)",
           title="Individual runs; squares: fixed-budget MC with 95% CP interval; x: nonfinal")
    ax.invert_yaxis()
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(destination / f"probabilities.{extension}", dpi=180)
    plt.close(fig)


def report(paths, output, *, figures=False, cases=False):
    paths = [Path(p).resolve() for p in paths]
    if not paths:
        raise ValueError("At least one input run is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate run directories are not independent replications")
    if figures:
        # Fail before writing a partial report if optional plotting is unavailable.
        try:
            __import__("matplotlib")
        except ImportError as exc:
            raise ValueError("Figures require the analysis extra: pip install 'raretrap[analysis]'") from exc
    if cases:
        for root in paths:
            if not (root / "raw_samples.jsonl").is_file():
                raise ValueError(f"Missing raw samples in {root}")
    # Validate input artifacts before making the new report directory.
    projections, estimates, levels, diagnostics, manifests, replication_entries = [], [], [], [], [], []
    for index, root in enumerate(paths):
        label = f"run_{index:03d}"
        raw = root / "raw_samples.jsonl"
        raw_bytes = raw.stat().st_size if raw.is_file() else None
        if (root / "metadata.json").exists() and raw_bytes is not None:
            projections.extend(dict(run=label, **row) for row in projection_rows(root, byte_limit=raw_bytes))
        elif (root / "cases.json").exists():
            row, metadata = estimator_details(root)
            estimates.append(dict(run=label, **row))
            replication_entries.append((estimates[-1], metadata))
            levels.extend(level_rows(root, label))
            diagnostics.append(dict(run=label, level_sampler=metadata.get("level_sampler"),
                                    evaluation_counts=evaluation_counts(metadata.get("level_sampler")),
                                    threshold_probability_trace=metadata.get("threshold_probability_trace"),
                                    estimator_result=metadata["estimator_result"]))
        else:
            raise ValueError(f"No supported run artifacts in {root}")
        manifests.append(dict(run=label, source=str(root), raw_bytes_snapshot=raw_bytes))
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, rows in (("projection", projections), ("estimates", estimates), ("populations", levels)):
        write_json(output / f"{name}.json", rows)
        write_csv(output / f"{name}.csv", rows)
    write_json(output / "diagnostics.json", diagnostics)
    replications = replication_summaries(replication_entries)
    write_json(output / "replications.json", replications)
    write_csv(output / "replications.csv", replications)
    write_json(output / "sources.json", manifests)
    if figures:
        plot_probabilities(estimates, output)
    with contextlib.redirect_stdout(io.StringIO()) as buffer:
        print_tables(projections, estimates)
    (output / "tables.md").write_text(buffer.getvalue(), encoding="utf-8")
    write_latex(output / "projection.tex", [dict(row, arm=projection_label(row["arm"])) for row in projections],
        [("model", "Model"), ("arm", "Projection"), ("samples", "Samples"),
         ("length", "Length P50/P75/P95"), ("repetition", "Repetition P50/P75/P95"), ("status", "Status")])
    write_latex(output / "estimates.tex", estimates,
        [("model", "Target"), ("surrogate", "Surrogate"), ("metric", "Metric"), ("threshold", "Threshold"),
         ("probability", "Estimate"), ("evaluations", "Evals"), ("levels", "Levels"),
         ("trajectory", "Threshold trajectory"), ("status", "Status")])
    for item, root in zip(manifests, paths):
        destination = output / item["run"]
        if figures or cases:
            destination.mkdir()
        if figures:
            plot_outputs(root, destination, [p for p in levels if p["run"] == item["run"]],
                         byte_limit=item["raw_bytes_snapshot"])
        if cases:
            item["exported_cases"] = export_cases(root, destination / "cases", byte_limit=item["raw_bytes_snapshot"])
    write_json(output / "sources.json", manifests)
    (output / "README.md").write_text(
        "# Generated analysis\n\nEach run stays separate. No automatic pooling, IID confidence intervals for MCMC, "
        "or convergence claim is made. Conditional quantiles count all retained population occurrences. "
        "Projection quantiles are P50/P75/P95; cap is each run's recorded generation limit. "
        "Population exports also include P25 and P95. Partial and nonfinal runs stay labeled. "
        "Binomial intervals require a fixed-budget level-zero MC run and the analysis extra. "
        "Diagnostic histories retain recorded thresholds, proposal widths, and acceptance information. "
        "Sources and optional raw-case exports may contain private paths or generated unsafe text; "
        "review this output before sharing it.\n", encoding="utf-8")
    print(f"Report written to {output}")
