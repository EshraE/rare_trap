"""Offline checks for documentation links and named paper configurations."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def anchors(text):
    # Our plain-text headings use GitHub's lowercase/hyphen fragment convention.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return {re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
            for heading in re.findall(r"^#{1,6} (.+)$", text, re.M)}


def check(root=ROOT):
    documents = [root / "README.md"]
    documents.extend(path for path in (root / "SECURITY.md", root / "CONTRIBUTING.md") if path.exists())
    documents.extend(sorted((root / "docs").glob("*.md")))
    problems = []
    for document in documents:
        problems.extend(check_document(root, document))
    return problems


def check_document(root, document):
    text = document.read_text(encoding="utf-8")
    problems = []
    for image, target in re.findall(r"(!?)\[[^\]]*\]\(([^)]+)\)", text):
        url = urlsplit(target)
        if url.scheme or url.netloc:
            if image:
                problems.append("Use reviewed local images, not third-party tracking/badge URLs.")
            continue
        path = (document.parent / unquote(url.path)).resolve() if url.path else document
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            problems.append(f"Missing or external local link: {target}")
        elif url.fragment and path.suffix == ".md":
            if unquote(url.fragment) not in anchors(path.read_text(encoding="utf-8")):
                problems.append(f"Missing heading: {target}")
    from raretrap.configuration import models, experiments
    for model in re.findall(r"--model ([a-z][\w.-]+)", text):
        if model not in models():
            problems.append(f"Unknown model in example: {model}")
    for preset in re.findall(r"--preset ([a-z][\w.-]+)", text):
        if preset not in experiments():
            problems.append(f"Unknown preset in example: {preset}")
    return problems


def main():
    # Direct-script invocation needs the source root for the small config module.
    import sys
    sys.path.insert(0, str(ROOT))
    findings = check()
    if findings:
        raise SystemExit("README check failed:\n" + "\n".join(findings))
    print("Documentation check passed: local links, figures, model aliases, and preset names.")


if __name__ == "__main__":
    main()
