# Contributing

For bug reports, include package versions, expected behavior, and a small
reproducible example without credentials or private data. See the
[README](README.md) for setup and experiment commands.

Keep pull requests focused and include a CPU regression test. Tests must run
without model downloads, credentials, or paid services.

```bash
python -m pip install -e '.[dev,analysis]'
python -m pip check
python -m pytest -q
python scripts/check_readme.py
python scripts/check_public_release.py --history
```

## Preserve the paper implementation

Keep the paper's equations, defaults, pinned model revisions, projection, scores,
and sampling behavior unchanged. Discuss methodological changes before submitting
them; do not update numerical fingerprints just to make tests pass.

Do not commit credentials, private data, or generated experiment outputs.
Report vulnerabilities as described in [SECURITY.md](SECURITY.md).
