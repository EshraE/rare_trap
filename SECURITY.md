# Security and responsible use

Run RareTrap only against models and infrastructure you own or are authorized to
evaluate. Long generations can consume substantial GPU time, memory, and storage.
Inspect `--dry-run`, select a device, and set an appropriate budget.

## Safe use

- Use trusted, pinned checkpoints and dependencies. Public commands disable remote
  model Python code, but this is not a sandbox for untrusted models or artifacts.
- Store credentials in the Hugging Face credential store or environment, never in
  committed files or bug reports.
- Prompts, responses, and logs may contain sensitive text, paths, or model IDs.
  Verify output-directory permissions and review files before sharing; console
  redaction is best effort. Do not execute generated text.
- Import CSV text columns as plain text to prevent spreadsheet formula execution.

## Reporting a vulnerability

Do not disclose vulnerabilities or sensitive data in public issues. Use private
vulnerability reporting in the repository's Security tab if available; otherwise,
request a private contact channel without posting details. Use synthetic examples
for reproductions.
