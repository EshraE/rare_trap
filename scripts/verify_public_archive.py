"""Inspect a normalized publication archive without extracting or executing it."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import zlib

from scripts.check_public_release import inspect_blob


REQUIRED = {"README.md", "pyproject.toml", "raretrap/cli.py", "raretrap/presets/models.json",
            "raretrap/presets/experiments.json", "scripts/run_paper_ablation.py",
            "tests/test_production_fidelity.py", "tests/production_fingerprints.json"}


def verify(path: Path) -> dict:
    path = Path(path)
    # Bound decompression and reject hidden trailing gzip members/data as well
    # as identifying gzip headers. No files from the archive are executed.
    digest = hashlib.sha256()
    decoded = bytearray()
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with path.open("rb") as handle:
        header = handle.read(10)
        if len(header) != 10 or header[:3] != b"\x1f\x8b\x08" or header[3] != 0 or any(header[4:8]):
            raise ValueError("gzip header contains non-normalized metadata")
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            decoded.extend(decoder.decompress(chunk, 25_000_001 - len(decoded)))
            if len(decoded) > 25_000_000 or decoder.unconsumed_tail:
                raise ValueError("Decompressed archive exceeds inspection limit.")
            if decoder.unused_data:
                raise ValueError("Archive contains trailing compressed members or data.")
    if not decoder.eof:
        raise ValueError("Incomplete gzip archive.")
    seen, problems, total = set(), [], 0
    end = 0
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:") as archive:
        for index, item in enumerate(archive):
            if index >= 1000 or item.size > 1_000_000:
                raise ValueError("Archive exceeds source-release inspection limits.")
            parts = PurePosixPath(item.name).parts
            if (not parts or parts[0] != "raretrap" or len(parts) < 2 or
                    ".." in parts or ".git" in parts or "\\" in item.name or
                    item.name != "/".join(parts)):
                raise ValueError("Archive contains an unsafe or noncanonical path.")
            name = "/".join(parts[1:])
            if name in seen or not item.isfile():
                raise ValueError("Archive contains duplicate paths, links, or non-file entries.")
            seen.add(name)
            end = item.offset_data + ((item.size + 511) // 512) * 512
            if (item.uid or item.gid or item.mtime or item.uname or item.gname or
                    item.pax_headers or item.mode not in {0o644, 0o755}):
                problems.append(f"non-normalized metadata: {name}")
            total += item.size
            if total > 20_000_000:
                raise ValueError("Archive exceeds total source-size limit.")
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError("Unreadable archive member.")
            problems.extend(inspect_blob(Path(name), stream.read()))
    if len(decoded) - end < 1024 or any(decoded[end:]):
        problems.append("tar archive contains missing terminators or hidden trailing data")
    if REQUIRED - seen:
        problems.append("missing required publication source files")
    if problems:
        raise ValueError("Archive verification failed:\n" + "\n".join(problems))
    return dict(status="passed", files=len(seen), source_bytes=total, sha256=digest.hexdigest(),
                scope="Archive contents and metadata only; not a GPU reproduction or licensing review.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.archive), indent=2))
    except (OSError, ValueError, tarfile.TarError, EOFError, zlib.error) as exc:
        parser.exit(2, f"Archive check failed: {exc}\n")


if __name__ == "__main__":
    main()
