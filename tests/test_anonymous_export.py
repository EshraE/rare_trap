import io
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import check_public_release as checker
from scripts.export_public_release import normalized_archive


def test_archive_removes_owner_names_timestamps_and_extended_attributes(tmp_path):
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w", format=tarfile.PAX_FORMAT) as archive:
        item = tarfile.TarInfo("README.md")
        item.uname, item.gname = "local-owner", "local-group"
        item.uid, item.gid, item.mtime = 501, 20, 123456789
        item.pax_headers = {"SCHILY.xattr.user.test": "private metadata"}
        item.size = 4
        archive.addfile(item, io.BytesIO(b"test"))
    destination = tmp_path / "source.tar.gz"
    normalized_archive(source.getvalue(), destination)
    with tarfile.open(destination) as archive:
        member = archive.getmember("raretrap/README.md")
        assert (member.uid, member.gid, member.mtime) == (0, 0, 0)
        assert member.uname == member.gname == ""
        assert not member.pax_headers
        assert archive.extractfile(member).read() == b"test"
    with pytest.raises(FileExistsError):
        normalized_archive(source.getvalue(), destination)


@pytest.mark.parametrize("name,kind", [("../outside", tarfile.REGTYPE), (".git/config", tarfile.REGTYPE),
                                     ("link", tarfile.SYMTYPE)])
def test_archive_refuses_unsafe_members_before_creating_output(tmp_path, name, kind):
    source = io.BytesIO()
    with tarfile.open(fileobj=source, mode="w") as archive:
        item = tarfile.TarInfo(name)
        item.type = kind
        archive.addfile(item)
    with pytest.raises(ValueError):
        normalized_archive(source.getvalue(), tmp_path / "source.tar.gz")
    assert not (tmp_path / "source.tar.gz").exists()


def test_history_scan_catches_removed_private_files_and_nonanonymous_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    env = dict(os.environ, GIT_AUTHOR_NAME="Identifiable contributor", GIT_COMMITTER_NAME="Identifiable contributor",
               GIT_AUTHOR_EMAIL="person@example.invalid", GIT_COMMITTER_EMAIL="person@example.invalid")

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True)

    git("init")
    (tmp_path / "notes.pdf").write_bytes(b"not a publishable source file")
    git("add", "notes.pdf")
    git("-c", "commit.gpgsign=false", "commit", "-m", "initial")
    git("rm", "notes.pdf")
    (tmp_path / "README.md").write_text("Clean source")
    git("add", "README.md")
    env.update(GIT_AUTHOR_NAME="Anonymous", GIT_COMMITTER_NAME="Anonymous",
               GIT_AUTHOR_EMAIL="anonymous@example.invalid", GIT_COMMITTER_EMAIL="anonymous@example.invalid")
    git("-c", "commit.gpgsign=false", "commit", "-m", "clean current files")
    assert not checker.check()
    findings = checker.check_history()
    assert any("non-anonymous commit identity" in item for item in findings)
    assert any("non-source artifact" in item for item in findings)


def test_symlink_checker_does_not_read_target(monkeypatch, tmp_path):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    (tmp_path / "README.md").symlink_to(tmp_path / "missing")
    monkeypatch.setattr(checker, "candidate_files", lambda: [Path("README.md")])
    assert checker.check() == ["symlink: README.md"]
