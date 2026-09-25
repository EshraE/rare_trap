"""Create a normalized source archive without workstation or Git metadata."""
from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path
import subprocess
import tarfile

from scripts import check_public_release
from scripts.verify_public_archive import verify


def normalized_archive(source: bytes, destination: Path) -> None:
    """Strip all input archive metadata; never extract or follow archive paths."""
    members = []
    with tarfile.open(fileobj=io.BytesIO(source), mode="r:") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
                raise ValueError("Unsafe source-archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Source export refuses links and non-regular files")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Unreadable archive member")
            members.append((member.name, stream.read(), bool(member.mode & 0o111)))
    destination = Path(destination)
    with destination.open("xb") as handle:
        # An empty gzip filename also prevents leaking the caller's output name.
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name, contents, executable in sorted(members):
                    item = tarfile.TarInfo("raretrap/" + name)
                    item.size = len(contents)
                    item.mode = 0o755 if executable else 0o644
                    item.uid = item.gid = item.mtime = 0
                    item.uname = item.gname = ""
                    archive.addfile(item, io.BytesIO(contents))


def export(destination: Path) -> None:
    root = check_public_release.ROOT
    findings = check_public_release.check() + check_public_release.check_history()
    if findings:
        raise ValueError("Release checks failed:\n" + "\n".join(findings))
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    if dirty.strip():
        raise ValueError("Commit reviewed source changes before exporting")
    source = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=root)
    normalized_archive(source, destination)
    verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        export(args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Export failed: {exc}\n")
    print(f"Source archive without Git or workstation metadata: {args.output}")


if __name__ == "__main__":
    main()
