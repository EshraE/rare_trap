# Experiments

[Back to RareTrap](../README.md) · [Setup](setup.md) · [Saved outputs](outputs.md)

## Estimate a behavioral-tail probability

```bash
CUDA_VISIBLE_DEVICES=0 raretrap estimate \
  --model qwen3-14b --surrogate qwen3-0.6b \
  --metric length --output outputs/qwen-length --dry-run
```

Inspect the settings, then remove `--dry-run` to execute. Use
`--metric repetition` for the repetition event and a new output directory
for every run. `raretrap models` lists pinned model aliases.

Defaults are $D=200$, $S=40$ surrogate-token slots, $N=1000$, $p_0=0.1$,
five conditional levels, greedy generation, and estimator seed 1010. The length
threshold is 20,000 and the repetition threshold is 0.99. The generation cap
is the length threshold plus three, including for repetition runs.

Subset Simulation uses an initial 200-evaluation Monte Carlo screen and retains
inherited seeds and every population-filling chain state without burn-in or
thinning. The initial population does not count toward `--max-levels`.
Set `--max-levels 0` for fixed-budget IID Monte Carlo.

## Latent-to-prompt map and score

The geometry-aware projection is

$$
e_s(z)=\mu+C_\epsilon^{1/2}Q R_s z,
\qquad \epsilon=10^{-5}.
$$

It uses all visible, non-special surrogate input embeddings, covariance shaping,
and raw Euclidean nearest-token lookup, with no radial or dimension-matching
rescaling. Selected surrogate IDs are decoded jointly to text, then processed
by the target's chat template and tokenizer. $S$ counts surrogate-token slots.

The projection seed is fixed internally at 1010 and has no command-line option.
`--seed` controls estimator/MCMC randomness, or IID latent sampling in the
projection comparison.

For repetition, remove special IDs from the completion. Let $D_n$ be the
fraction of distinct overlapping token n-grams for $n\in\{2,3,4\}$, with
$D_n=1$ when that order is unavailable. The score and limit-state function are

$$
r_{\mathrm{rep}}=1-D_2D_3D_4,\qquad g(z)=\tau-r(y(z)).
$$

Success is `g <= 0`. Both length and repetition are saved for every response.

### Paper-to-code reference

| Component | Implementation |
|---|---|
| Candidate embeddings, covariance, projection, nearest-token decoding | [projection_space.py](../utils/projection_space.py) |
| Chat encoding and generated-token accounting | [llm_generator.py](../utils/llm_generator.py) |
| Multi-order repetition score | [repetition_metrics.py](../utils/repetition_metrics.py) |
| Subset levels, componentwise Metropolis, proposal-width adaptation | [engine.py](../estimators_samplers/engine.py) |
| Paired geometry-aware / embedding-agnostic comparison | [projection.py](../raretrap/projection.py), [_projection.py](../raretrap/_projection.py) |
| Model revisions and experiment settings | [models.json](../raretrap/presets/models.json), [experiments.json](../raretrap/presets/experiments.json) |

The [production-fidelity tests](../tests/test_production_fidelity.py) guard the
preserved numerical implementation; mathematical and workflow tests exercise
its behavior independently. Start with the [offline checks](setup.md#verify-without-model-downloads)
before allocating resources for a full run.

## Projection comparison

```bash
CUDA_VISIBLE_DEVICES=0 raretrap-ablation \
  --model deepseek8b --output outputs/deepseek-ablation --dry-run
# Remove --dry-run to execute, then:
raretrap summarize --input outputs/deepseek-ablation
```

The dedicated runner is `scripts/run_paper_ablation.py`;
`raretrap projection` provides the same workflow. It compares
`geometry_aware` with `embedding_agnostic`, using a self surrogate and
the same 100 IID Gaussian latent vectors for both arms. The Gaussian baseline
is $e_s(z)=A_s z$, with independent entries of variance $1/D$.

The comparison uses $S=40$, $D=200$, sampling seed 20260806, greedy generation,
batch size one, and a 20,003-token cap. The paper reports `deepseek8b`,
`qwen3.5-9b`, `olmo3-7b`, and `nemotron9b`. Other available targets
are listed by `raretrap-ablation --help`.

Outputs include full prompt/response records, latent vectors, projection
tensors, and P50/P75/P95 summaries. In tables, `cap` denotes the generation
limit; unfinished arms are marked partial.

## Paper configurations

```bash
raretrap presets --group validation
raretrap run \
  --preset validation.qwen3-14b.qwen3-0.6b.repetition.seed1010 \
  --output outputs/qwen-validation --dry-run
```

| Preset group | Configurations | Purpose |
|---|---:|---|
| `matched` | 32 | Eight core targets, two surrogates, two metrics |
| `additional` | 8 | Two larger targets, two external surrogates, two metrics |
| `validation` | 30 | Six configurations and five estimator seeds |
| `mc` | 7 | Independent Monte Carlo jobs, including split jobs |
| `extreme` | 4 | Extreme-length runs with 1,000 samples per level |

Presets contain settings, not measured outcomes. `raretrap run` preserves
them without overrides; use `estimate` for a new configuration.

The original core-table runs used different projection realizations.
`matched.*` presets rerun those settings with the release's fixed projection
seed 1010; they do not exactly regenerate the printed historical cells.
The other preset groups and the projection comparison retain their recorded
projection seed. Numerical results can vary across execution environments.

### Extreme generation

List the runs with `raretrap presets --group extreme`. They use $N=1000$,
at most 12 conditional levels, and thresholds of 100,000 for DeepSeek and
Nemotron, 65,000 for OLMo, and 32,768 for Qwen3-14B. Caps are threshold plus three.

### Prompt-length and latent-dimension sensitivity

The study uses DeepSeek-8B and Qwen3-14B, each with self and Qwen3-0.6B
surrogates and both response metrics.

| Input | Values | Held fixed |
|---|---|---|
| `--prompt-length` ($S$) | 20, 40, 80 | $D=200$ |
| `--latent-dim` ($D$) | 100, 200, 400 | $S=40$ |

Use `estimate` with estimator seed 20260917, $N=1000$, $p_0=0.1$, at most
five conditional levels, and the main thresholds and cap. Each cell needs a
separate output directory. Changing $S$ or $D$ changes the reference prompt
distribution. The paper's principal-configuration rows come from its main
table rather than the sensitivity seed.

### Larger-model additions

The `additional.*` presets require a compatible vLLM installation and two-way
tensor parallelism. They specify BF16 KV cache, batch size eight, model
quantization, and thinking-template settings. Qwen3.8 uses unquantized weights;
Nemotron3.5 uses its NVFP4 checkpoint with `modelopt_fp4`.

```bash
CUDA_VISIBLE_DEVICES=0,1 raretrap run \
  --preset additional.qwen3.8-27b.qwen3-0.6b.length \
  --output outputs/qwen38-length --dry-run
```

vLLM is installed separately. Backend or quantization failures are reported
without substituting a different configuration.
