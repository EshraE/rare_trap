import gzip
import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import check_public_release as checker
from scripts.export_public_release import normalized_archive
from scripts.verify_public_archive import REQUIRED, verify


def source_tar(extra=()):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(REQUIRED):
            data = b"source\n"
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            archive.addfile(item, io.BytesIO(data))
        for item, data in extra:
            archive.addfile(item, io.BytesIO(data))
    return stream.getvalue()


def test_verify_normalized_archive_is_read_only(tmp_path):
    archive = tmp_path / "release.tar.gz"
    normalized_archive(source_tar(), archive)
    before = archive.read_bytes()
    result = verify(archive)
    assert result["status"] == "passed" and result["files"] == len(REQUIRED)
    assert len(result["sha256"]) == 64
    assert archive.read_bytes() == before
    assert list(tmp_path.iterdir()) == [archive]


@pytest.mark.parametrize("change", ["extra_gzip", "extra_bytes", "hidden_tar", "gzip_timestamp"])
def test_archive_rejects_hidden_data_and_identifying_headers(tmp_path, change):
    path = tmp_path / "release.tar.gz"
    normalized_archive(source_tar(), path)
    payload = path.read_bytes()
    if change == "extra_gzip":
        payload += gzip.compress(b"hidden", mtime=0)
    elif change == "extra_bytes":
        payload += b"hidden"
    elif change == "hidden_tar":
        payload = gzip.compress(gzip.decompress(payload) + b"hidden", mtime=0)
    else:
        payload = gzip.compress(gzip.decompress(payload), mtime=100)
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        verify(path)


@pytest.mark.parametrize("change", ["duplicate", "owner", "unsafe", "missing", "result"])
def test_archive_rejects_unpublishable_members(tmp_path, change):
    path = tmp_path / "release.tar.gz"
    normalized_archive(source_tar(), path)
    original = gzip.decompress(path.read_bytes())
    stream = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original)) as source, tarfile.open(fileobj=stream, mode="w") as destination:
        members = list(source)
        if change == "missing":
            members = members[1:]
        for member in members:
            if change == "owner":
                member.uname = "workstation-owner"
            destination.addfile(member, source.extractfile(member))
        if change == "duplicate":
            destination.addfile(members[0], source.extractfile(members[0]))
        if change in {"unsafe", "result"}:
            item = tarfile.TarInfo("raretrap/../escape" if change == "unsafe" else "raretrap/results/data.json")
            destination.addfile(item, io.BytesIO())
    path.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
    with pytest.raises(ValueError):
        verify(path)


def test_unpacked_source_checker_never_reads_parent_git(monkeypatch, tmp_path):
    root = tmp_path / "exported"
    root.mkdir()
    (root / "README.md").write_text("public source")
    (root / "__pycache__").mkdir()
    (root / "__pycache__/cache.pyc").write_bytes(b"generated")
    (root / "raretrap.egg-info").mkdir()
    (root / "raretrap.egg-info/PKG-INFO").write_text("generated package metadata")
    monkeypatch.setattr(checker, "ROOT", root)
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: pytest.fail("must not invoke parent Git"))
    assert checker.candidate_files() == [Path("README.md")]
    assert checker.check() == []
    assert "no local Git history" in checker.check_history()[0]
    (root / "results").mkdir()
    (root / "results/data.json").write_text(json.dumps({"private": "data"}))
    assert any("unexpected path" in message for message in checker.check())


def test_unpacked_source_directory_symlink_is_not_followed(monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    (tmp_path / "utils").symlink_to(tmp_path, target_is_directory=True)
    assert checker.candidate_files() == [Path("utils")]
    assert checker.check() == ["symlink: utils"]
