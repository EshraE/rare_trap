# Contributing

Keep changes focused and include a small CPU regression test. Installation and
experiment commands are in the [README](README.md). Tests must not download model
weights, require credentials, or call paid services.

```bash
python -m pip install -e '.[dev,analysis]'
python -m pip check
python -m pytest -q
python scripts/check_readme.py
python scripts/check_public_release.py --anonymous
```

## Preserve the paper implementation

Do not silently change production equations, parameter defaults, model revisions,
projection construction, score definitions, or sampler random-number order.
The internal projection seed stays fixed and is not a CLI option. Numerical
fingerprint tests protect the recorded implementation: do not update fingerprints
merely to make a changed method pass. Discuss any proposed methodological change
as a separate, explicitly labeled experiment.

## Source hygiene

Do not add personal identifiers, credentials, local paths, server addresses,
generated results, manuscript sources, or private datasets. The source check
also audits reachable history, so removing a file in a later commit does not
erase it from an existing repository.

The README figures are intentionally hash-pinned. Replacements require
visual inspection and a metadata review before updating their accepted hashes.
Do not broaden the binary allowlist to bypass a failed check.

For ordinary bug reports, include the command with sensitive values removed,
package versions, expected behavior, and a synthetic minimal reproduction. Do not
attach raw experiment logs by default. Use [SECURITY.md](SECURITY.md) for security
issues. Licensing remains undecided; do not introduce code or assets whose terms
have not been reviewed, or represent the repository as already open-source licensed.
