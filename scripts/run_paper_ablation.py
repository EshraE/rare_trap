"""Run the paper's paired self-surrogate projection ablation.

Usage after installing RareTrap:
    python -m scripts.run_paper_ablation --model deepseek8b --output outputs/ablation

This entry point delegates to the tested public projection workflow so settings,
validation, raw outputs, and completion markers cannot drift between runners.
The alternative projection builders remain internal helpers.
"""
from __future__ import annotations

import sys

from raretrap import cli


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    cli.main(["projection", *arguments])


if __name__ == "__main__":
    main()
