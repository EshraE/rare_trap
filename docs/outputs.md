# Outputs and analysis

[Back to RareTrap](../README.md) · [Experiments](experiments.md)

## Saved artifacts

Every run uses a new output directory and continuously saves completed evaluations.

| Artifact | Contents |
|---|---|
| `configuration.json` | Requested model and experiment settings |
| `run.log`, `events.jsonl`, `status.json` | Logs, heartbeat, progress, and terminal status |
| `raw_samples.jsonl` | Full evaluation records, flushed and fsynced after each completed evaluation |
| `dataset/samples.jsonl` | One dataset row per completed generation |
| `dataset/manifest.json` | Schema, row count, state, and checksum |
| `cases.json`, `all_evaluations.json` | Estimator results and evaluation summaries |
| `sampler_trace/` | Retained populations, lineage, transitions, and completed-level checkpoints |
| `DONE` / `FAILED` | Execution completion or caught failure/interruption |

Raw records contain latent vectors, surrogate IDs, prompt text, target input IDs,
completion IDs/text, and both response scores. Projection comparisons also save
running summaries, candidate embeddings, projection factors, and tensor checksums.

`DONE` indicates execution completed. For conditional runs, check
`target_probability_status` and `target_reached` before treating a probability
as final. Reports label a valid, completed fixed-budget Monte Carlo run as final
even with zero hits. A heartbeat does not necessarily mean generation is advancing.

## Monitor and recover saved evaluations

```bash
raretrap status --input outputs/qwen-length
tail -f outputs/qwen-length/run.log

raretrap export-dataset --input outputs/qwen-length \
  --output outputs/qwen-length-recovered
```

Dataset recovery leaves the raw journal unchanged, ignores an unfinished final
line, and rejects malformed complete records. The live writer detects raw-journal
replacement, disappearance, or truncation. Recovery refuses an existing destination.
Completed evaluations survive ordinary interruption; an in-flight response is
not saved token by token. Automatic MCMC resume is not implemented. Use completed
runs or recovered snapshots for consistent analysis.

To read a dataset with the optional Hugging Face Datasets package:

```python
from datasets import load_dataset

samples = load_dataset(
    "json", data_files="outputs/qwen-length/dataset/samples.jsonl",
    split="train", streaming=True,
)
```

## Summaries and reports

```bash
raretrap summarize --input outputs/qwen-length outputs/qwen-repetition
raretrap report --input outputs/qwen-length outputs/qwen-repetition \
  --output outputs/paper-report --figures --cases
```

Install the `analysis` extra for figures and exact binomial intervals.

| Report | Contents |
|---|---|
| `projection.tex` | Length/repetition P50/P75/P95, without Spearman correlation |
| `estimates.tex` | Probabilities, evaluation counts, levels, and threshold trajectories |
| `populations.csv` | Per-level quantiles and unique-state/prompt/output counts |
| `diagnostics.json` | Acceptance, proposal widths, and threshold histories |
| `replications.csv` | Descriptive mean and sample SD for matching final Subset Simulation runs with distinct seeds |
| `--figures` | Projection CDFs, population trajectories, and probability comparisons |
| `--cases` | Every evaluated prompt, response, and lossless record in separate directories |

Reports retain partial arms and nonfinal estimates. Projection tables, figures,
and case exports use the same saved raw-file prefix,
recorded in `sources.json`, even when more evaluations arrive during export.
Levels include level zero; displayed terminal thresholds are clipped to the requested event threshold.
Only fixed-budget IID Monte Carlo (`--max-levels 0`) receives a 95%
Clopper–Pearson interval. Correlated chain populations and the success-selected
early screen do not receive IID binomial intervals.

Raw evaluations include rejected proposals. Use `sampler_trace` for retained
conditional populations; their quantiles include inherited and repeated states.
In conditional levels, `level_sampler.evaluations_this_level` counts proposed
sweeps; reports expose actual model calls through `evaluation_counts`.
Unique counts and replication SD are diagnostics, not convergence tests.

Compare runs with matching maps, model/surrogate revisions, thresholds, caps,
and generation settings. Keep split Monte Carlo jobs and independent seeds
identifiable. Review generated text before sharing and keep outputs outside the
source repository; see [Security](../SECURITY.md).
