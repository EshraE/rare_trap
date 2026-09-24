# Security and responsible use

Run RareTrap only against models and infrastructure you own or are authorized to
evaluate. Long generations can consume substantial GPU time, memory, and storage.
Inspect `--dry-run`, allocate an explicit device, and set an appropriate budget.
This repository does not launch evaluations against third-party hosted services.

## Trust boundaries

- Use approved, pinned model checkpoints. Public commands explicitly disable
  remote model Python code;
  this does not make an arbitrary checkpoint or dependency trustworthy.
- Keep credentials in the supported Hugging Face credential store or environment,
  not command arguments, committed files, examples, or bug reports.
- Experiment, dataset-export, and report directories are created with owner-only
  permissions on POSIX systems. This is not encryption or a substitute for your
  storage access policy; verify platform permissions and any inherited ACLs.
  Prompts, responses, and logs may contain sensitive or offensive text. Review before sharing;
  do not execute generated text, follow its instructions, or upload it automatically.
- Raw CSV exports preserve generated text verbatim. Import text columns as plain
  text; spreadsheet applications may interpret generated text as formulas.
- Console redaction is best effort, not a general data-loss-prevention system.
  Source anonymity does not anonymize runtime paths, private model IDs, or datasets.
- Parse artifacts only from trusted experiments. Resource bounds and validation
  are safeguards, not a sandbox for arbitrary hostile files or model outputs.

## Reporting a vulnerability

Do not put credentials, private prompts, infrastructure addresses, or identifying
details in public issues. If the hosting service provides private vulnerability
reporting, use its Security tab. If it is unavailable, request a private reporting
channel without disclosing the vulnerability or private data publicly.

Use synthetic inputs for reproductions. No security-response SLA or independent
security certification is claimed. The repository remains a research implementation;
passing tests is not proof of safety for every model or deployment environment.
