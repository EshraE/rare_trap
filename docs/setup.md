# Setup

[Back to RareTrap](../README.md)

## Installation

Use Python 3.11 on Linux with a CUDA-enabled PyTorch installation for model
experiments. Create a fresh environment and install RareTrap:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch build appropriate to your device, then:
python -m pip install -e '.[analysis]'
python -m pip check
raretrap --help
```

The `analysis` extra enables plots and exact binomial intervals. For development
and CPU tests, see [Contributing](../CONTRIBUTING.md).

## Verify without model downloads

From the source directory, after installing the dependencies:

```bash
python -m pip install -e '.[dev,analysis]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/check_readme.py
```

The tests need no GPU, credentials, or downloaded weights. They check projection
equations, response scores, sampler behavior, saved outputs, and a tiny randomly
initialized Transformers model. Passing these tests verifies functionality, not
the paper's empirical results. Full paper experiments use the pinned checkpoints
and configurations in [Experiments](experiments.md).

Paper model aliases pin checkpoint revisions, precision, and generation settings.
Authenticate gated models through the Hugging Face credential store or
`HF_TOKEN`; keep credentials out of commands, configuration files, and commits.

## Custom Hugging Face models

Use `--hf-model` with a full 40-character checkpoint commit hash:

```bash
MODEL_COMMIT=REPLACE_WITH_FULL_40_CHARACTER_COMMIT
CUDA_VISIBLE_DEVICES=0 raretrap estimate \
  --hf-model example-org/chat-model --revision "$MODEL_COMMIT" \
  --dtype bfloat16 --surrogate self \
  --output outputs/custom-length --dry-run
```

Replace the model ID and revision, inspect the settings, then remove
`--dry-run` to execute. `raretrap-ablation` accepts the same custom-model
options. Custom checkpoints use the method defaults and define new experiments.

The model must be supported by Transformers `AutoModelForCausalLM`, provide
input embeddings, and have a tokenizer with a chat template and EOS/pad token.
Remote model Python code is disabled. Context limits are checked before
generation; prompts and generation caps are never silently shortened.
Use `--threshold` for shorter-context estimation. The ablation cap is fixed
at 20,003 new tokens.

`--dtype` and `--thinking {auto,enabled,disabled}` apply to custom checkpoints.
`auto` uses the tokenizer's default thinking behavior. Paper aliases retain
their recorded settings. Run one experiment per process.

## Devices and reproducibility

Select an available device with `CUDA_VISIBLE_DEVICES` and allow memory for
both weights and generation. Additional-model presets require vLLM and the
tensor-parallel configuration described in [Experiments](experiments.md#larger-model-additions).
RareTrap does not schedule jobs or change other processes.

Pinned settings support repeatable experiments, but bitwise identity also depends
on library versions and numerical kernels. Preserve generated projection tensors
and raw samples when exact replay is needed. See the
[reproduction scope](experiments.md#paper-configurations) for historical settings.
