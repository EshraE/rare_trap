"""Fail closed on accidental private material in the publication tree.

This heuristic complements manual review; it is not a guarantee that arbitrary
secrets can be detected. It never prints matching credential values.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {".github", "raretrap", "utils", "estimators_samplers", "scripts", "tests"}
ALLOWED_FILES = {"README.md", "SECURITY.md", "CONTRIBUTING.md", "pyproject.toml", ".gitignore", "MANIFEST.in"}
ALLOWED_FILES.update({"docs/setup.md", "docs/experiments.md", "docs/outputs.md"})
# Only these manually reviewed, rendered figures may bypass UTF-8 inspection.
# Updating one requires inspecting the visible content and PNG metadata again.
# Never accept arbitrary binary files merely because their extension is PNG.
REVIEWED_IMAGES = {
    "assets/mc_comparison.png": "d93a267f4499293abaefa83f1d393a3192cb9d3b69223df27c6edea435581635",
    "assets/overview.png": "40c1bdb612453efa51c544bff8cc0fb13af265d42d12131c5c88c067822e32dd",
    "assets/pipeline.png": "92cd1a17335ac36638b8a6dcf64886dcde393fc2e28d90a2873f7a58908851a5",
}
TEXT_EXTENSIONS = {".py", ".json", ".txt", ".md", ".toml", ".yml", ".yaml"}
BLOCKED_EXTENSIONS = {".pdf", ".docx", ".tex", ".pem", ".key", ".p12", ".pt", ".pth",
                      ".safetensors", ".npz", ".npy", ".csv", ".jsonl", ".log", ".ipynb"}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"),
    "home directory": re.compile(r"/(?:Users|home)/[^\s\"']+"),
    "Windows home directory": re.compile(r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"),
    "hardware or node identifier": re.compile(r"\b(?:[AHV][1-9]\d{1,3}|node[-_ ]?\d{2,})\b", re.I),
    "accelerator build fingerprint": re.compile(r"\+cu\d{3}\b|\bCUDA\s+\d+\.\d+\b", re.I),
    "identifying commit trailer": re.compile(r"^(?:Co-authored-by|Signed-off-by):", re.M | re.I),
    "IPv4 address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credential URL": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
}


def candidate_files() -> list[Path]:
    # Never inherit an unrelated enclosing checkout when reviewing an unpacked
    # source release. Permit only routine local build/test products to be skipped.
    if not (ROOT / ".git").exists():
        excluded = {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}
        paths = []
        for directory, folders, files in os.walk(ROOT, followlinks=False):
            for folder in list(folders):
                path = Path(directory) / folder
                if path.is_symlink():
                    paths.append(path.relative_to(ROOT))
                    folders.remove(folder)
                elif folder in excluded or folder.endswith(".egg-info"):
                    folders.remove(folder)
            paths.extend((Path(directory) / name).relative_to(ROOT) for name in files
                         if not name.endswith((".pyc", ".pyo")))
        return sorted(paths)
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    return [Path(name) for name in result.stdout.split("\0") if name]


def inspect_blob(relative: Path, data: bytes, *, symlink=False) -> list[str]:
    problems = []
    if relative.as_posix() in REVIEWED_IMAGES:
        if symlink:
            return [f"symlink: {relative}"]
        if hashlib.sha256(data).hexdigest() != REVIEWED_IMAGES[relative.as_posix()]:
            return [f"unreviewed image contents: {relative}"]
        return []
    if relative.parts[0] not in ALLOWED_ROOTS and str(relative) not in ALLOWED_FILES:
        problems.append(f"unexpected path: {relative}")
    if symlink:
        return problems + [f"symlink: {relative}"]
    if relative.suffix.lower() in BLOCKED_EXTENSIONS:
        problems.append(f"non-source artifact: {relative}")
    if relative.suffix.lower() not in TEXT_EXTENSIONS and str(relative) not in ALLOWED_FILES:
        problems.append(f"unsupported source format: {relative}")
    if any(part == ".env" or part.startswith(".env.") or part in
           {".ssh", ".aws", ".netrc", ".npmrc", ".pypirc"} for part in relative.parts):
        problems.append(f"private configuration path: {relative}")
    if len(data) > 1_000_000:
        problems.append(f"large file: {relative}")
    try:
        contents = data.decode("utf-8")
    except UnicodeDecodeError:
        return problems + [f"binary file: {relative}"]
    for label, pattern in PATTERNS.items():
        if pattern.search(contents):
            problems.append(f"{label}: {relative}")
    return problems


def check() -> list[str]:
    problems = []
    for relative in candidate_files():
        path = ROOT / relative
        if path.is_symlink():
            problems.extend(inspect_blob(relative, b"", symlink=True))
        elif not path.is_file():
            problems.append(f"missing or non-regular file: {relative}")
        else:
            problems.extend(inspect_blob(relative, path.read_bytes()))
    return problems


def check_history() -> list[str]:
    """Audit reachable source history and identities, without printing identities."""
    if not (ROOT / ".git").exists():
        return ["no local Git history: use the source check or verify_public_archive for an exported release"]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        return ["anonymous history check requires a non-shallow clone"]
    commits = git("rev-list", "HEAD").decode().splitlines()
    allowed = {("RareTrap contributors", "contributors@raretrap.invalid"),
               ("Anonymous", "anonymous@example.invalid")}
    problems, seen = [], set()
    for commit in commits:
        fields = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", commit).decode().strip().split("\0")
        if len(fields) != 4 or tuple(fields[:2]) not in allowed or tuple(fields[2:]) not in allowed:
            problems.append(f"non-anonymous commit identity: {commit[:12]}")
        message = git("show", "-s", "--format=%B", commit).decode()
        if any(pattern.search(message) for pattern in PATTERNS.values()):
            problems.append(f"sensitive commit message: {commit[:12]}")
        for entry in git("ls-tree", "-rz", commit).split(b"\0"):
            if not entry:
                continue
            header, name = entry.split(b"\t", 1)
            mode, kind, oid = header.decode().split()
            relative = Path(name.decode())
            key = (mode, oid, str(relative))
            if key in seen:
                continue
            seen.add(key)
            if kind != "blob":
                problems.append(f"non-source history entry: {relative}")
                continue
            data = git("cat-file", "blob", oid)
            problems.extend("history: " + item for item in inspect_blob(relative, data, symlink=mode == "120000"))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymous", action="store_true", help="Also inspect full history and commit identities")
    args = parser.parse_args()
    try:
        findings = check()
        if args.anonymous:
            findings.extend(check_history())
    except (OSError, subprocess.CalledProcessError):
        findings = ["release inspection could not complete; check source access and Git availability"]
    if findings:
        print("Release check failed:\n" + "\n".join(findings), file=sys.stderr)
        raise SystemExit(1)
    print("Release check passed: source/configuration allowlist and credential/path scan.")


if __name__ == "__main__":
    main()
