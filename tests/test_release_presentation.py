from pathlib import Path
import struct
import zlib

import pytest

from scripts import check_public_release as checker
from scripts.check_readme import check


ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_and_configuration_examples():
    assert check(ROOT) == []


def test_documentation_links_resolve_relative_to_their_own_file(tmp_path):
    (tmp_path / "README.md").write_text("# Example\n[setup](docs/setup.md)\n")
    (tmp_path / "docs").mkdir()
    setup = tmp_path / "docs/setup.md"
    setup.write_text("# Setup\n[home](../README.md#example)\n")
    assert check(tmp_path) == []
    setup.write_text("# Setup\n[broken](../README.md#missing)\n")
    assert len(check(tmp_path)) == 1


@pytest.mark.parametrize("name", sorted(checker.REVIEWED_IMAGES))
def test_reviewed_illustrations_are_metadata_free_pngs(name):
    data = (ROOT / name).read_bytes()
    assert checker.inspect_blob(Path(name), data) == []
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    position, kinds = 8, []
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        body = data[position + 8:position + 8 + length]
        crc = struct.unpack(">I", data[position + 8 + length:position + 12 + length])[0]
        assert zlib.crc32(kind + body) & 0xffffffff == crc
        # No text, EXIF, time, embedded file/profile, or application-specific chunks.
        assert kind in {b"IHDR", b"IDAT", b"IEND", b"pHYs"}
        kinds.append(kind)
        position += length + 12
        if kind == b"IEND":
            assert length == 0 and position == len(data)
            break
    assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND" and b"IDAT" in kinds
    assert checker.inspect_blob(Path(name), data + b"extra")
    assert checker.inspect_blob(Path(name), data, symlink=True)


@pytest.mark.parametrize("name", ["assets/new.png", "utils/image.png", "utils/archive.zip",
                                  "utils/.env", "utils/.env.local", "utils/.aws/config.json"])
def test_nonreviewed_assets_and_private_configuration_are_rejected(name):
    assert checker.inspect_blob(Path(name), b"innocent-looking text")


def test_readme_checker_detects_broken_assets_and_tracking_images(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Example\n![missing](assets/missing.png)\n![remote](https://example.invalid/badge)\n"
        "[heading](#missing)\n--model not-a-model\n--preset missing.preset\n"
    )
    assert len(check(tmp_path)) == 5


@pytest.mark.parametrize("payload", ["A" + "100", "node" + "-02", "torch==2.13.0" + "+" + "cu130",
    "github_" + "pat_" + "x" * 25, "C:" + "\\Users\\example\\project",
    "Co-authored" + "-by: Example <example@example.invalid>"])
def test_private_metadata_is_detected_without_printing_its_value(payload):
    findings = checker.inspect_blob(Path("README.md"), payload.encode())
    assert findings and all(payload not in message for message in findings)
