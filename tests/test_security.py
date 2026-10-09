"""F31 - the hardening on an endpoint that accepts hostile video.

Every check here corresponds to a real attack the threat model (docs/THREAT_MODEL.md) names:

* a renamed executable (extension trust)
* a declared-small / actually-huge upload (disk exhaustion)
* a long clip (CPU exhaustion)
* ``../../etc/passwd`` as an artifact name (path traversal)
* an HLS/concat container that makes ffmpeg fetch a URL (SSRF via ffmpeg)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from api.security import (
    UploadRejected,
    ffmpeg_hardened_args,
    new_storage_key,
    probe_and_validate,
    safe_display_name,
    sniff_container,
    storage_path,
    stream_to_disk,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "tests" / "fixtures" / "videos"


class _FakeUpload:
    """Minimal UploadFile stand-in."""

    def __init__(self, data: bytes, chunk: int = 1 << 16) -> None:
        self._data = data
        self._pos = 0
        self._chunk = chunk

    async def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._data):
            return b""
        n = self._chunk if n in (-1, None) else min(n, self._chunk)
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out


def _run(coro):
    return asyncio.run(coro)


# ==========================================================================
# content sniffing -- never trust the extension
# ==========================================================================
def test_sniff_accepts_real_containers():
    assert sniff_container(b"\x00\x00\x00\x20ftypisom")  # mp4
    assert sniff_container(b"\x1aE\xdf\xa3" + b"\x00" * 12)  # matroska
    assert sniff_container(b"RIFF\x00\x00\x00\x00AVI ")  # avi


def test_sniff_rejects_non_video():
    assert not sniff_container(b"MZ\x90\x00" + b"\x00" * 12)  # PE executable
    assert not sniff_container(b"\x7fELF" + b"\x00" * 12)  # ELF binary
    assert not sniff_container(b"#!/bin/sh\necho hi\n")  # shell script
    assert not sniff_container(b"%PDF-1.7" + b"\x00" * 8)  # pdf
    assert not sniff_container(b"<html><body>x</body></html>")  # html


def test_renamed_executable_is_rejected(tmp_path):
    """The headline case: ``payload.exe`` renamed to ``clip.mp4``."""
    fake = _FakeUpload(b"MZ\x90\x00" + b"\x41" * 4096)
    with pytest.raises(UploadRejected, match="container"):
        _run(stream_to_disk(fake, tmp_path / "clip.mp4"))
    assert not (tmp_path / "clip.mp4").exists(), "the rejected file was left on disk"


def test_real_fixture_passes_sniffing_and_probing(tmp_path):
    src = VIDEOS / "real_sync_audio_v0.mp4"
    if not src.exists():
        pytest.skip("fixtures not generated")
    dest = tmp_path / f"{new_storage_key()}.bin"
    size = _run(stream_to_disk(_FakeUpload(src.read_bytes()), dest))
    assert size == src.stat().st_size

    info = probe_and_validate(dest, size)
    assert info.duration_s > 0 and info.width > 0
    assert info.vcodec in ("h264", "hevc", "mpeg4")


# ==========================================================================
# resource caps
# ==========================================================================
def test_oversize_upload_is_rejected_while_streaming(tmp_path):
    # A valid header followed by more bytes than the cap: the limit must be
    # enforced per chunk, not from a client-supplied Content-Length.
    blob = b"\x00\x00\x00\x20ftypisom" + b"\x00" * (3 * 1024 * 1024)
    with pytest.raises(UploadRejected, match="exceeds"):
        _run(stream_to_disk(_FakeUpload(blob), tmp_path / "big.bin", max_mb=1))
    assert not (tmp_path / "big.bin").exists()


def test_empty_upload_is_rejected(tmp_path):
    with pytest.raises(UploadRejected, match="Empty"):
        _run(stream_to_disk(_FakeUpload(b""), tmp_path / "empty.bin"))


def test_overlong_clip_is_rejected(tmp_path, monkeypatch):
    src = VIDEOS / "real_sync_audio_v0.mp4"
    if not src.exists():
        pytest.skip("fixtures not generated")
    dest = tmp_path / "v.bin"
    size = _run(stream_to_disk(_FakeUpload(src.read_bytes()), dest))

    monkeypatch.setattr("api.security.MAX_DURATION_S", 0.5)
    with pytest.raises(UploadRejected, match="limit is"):
        probe_and_validate(dest, size)


def test_undecodable_file_is_rejected_cleanly(tmp_path):
    # Valid magic, garbage body: ffprobe must fail and we must return a
    # client-safe message rather than a traceback.
    dest = tmp_path / "bad.bin"
    dest.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\xff" * 2048)
    with pytest.raises(UploadRejected, match=r"Could not decode|video"):
        probe_and_validate(dest, dest.stat().st_size)


# ==========================================================================
# path traversal
# ==========================================================================
@pytest.mark.parametrize(
    "key",
    [
        "../etc/passwd",
        "../../root",
        "a/b",
        "a\x00b",
        "",
        "." * 80,
        "key with space",
        "/absolute",
        "..",
        "key;rm -rf /",
    ],
)
def test_storage_path_rejects_hostile_keys(tmp_path, key):
    with pytest.raises(UploadRejected, match="Invalid storage key"):
        storage_path(tmp_path, key)


def test_storage_path_accepts_generated_keys(tmp_path):
    for _ in range(5):
        k = new_storage_key()
        p = storage_path(tmp_path, k)
        assert str(p).startswith(str(tmp_path.resolve()))


def test_display_name_is_sanitised_and_never_a_path():
    assert safe_display_name("../../etc/passwd") == "passwd"
    assert safe_display_name("my clip (1).mp4") == "my_clip__1_.mp4"
    assert safe_display_name(None) == "upload"
    assert safe_display_name("") == "upload"
    assert "/" not in safe_display_name("a/b/c.mp4")
    assert len(safe_display_name("x" * 500)) <= 120


def test_api_rejects_traversal_in_artifact_names(tmp_path):
    """The HTTP-level check, not just the helper."""
    pytest.importorskip("fastapi")
    import importlib
    import os

    os.environ.setdefault("DDETECT_RUN_DIR", "runs/smoke_test/seed0")
    os.environ["DDETECT_ARTIFACT_DIR"] = str(tmp_path)
    import api.main as m

    importlib.reload(m)
    from fastapi.testclient import TestClient

    c = TestClient(m.app)
    job = m.store.create("x.mp4", new_storage_key())
    for name in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "a.txt", "x.png.exe"):
        r = c.get(f"/v1/analyses/{job.job_id}/artifacts/{name}")
        assert r.status_code in (400, 404), f"{name} returned {r.status_code}"


# ==========================================================================
# ffmpeg hardening -- SSRF
# ==========================================================================
def test_ffmpeg_args_disable_network_protocols():
    args = ffmpeg_hardened_args()
    assert "-protocol_whitelist" in args
    i = args.index("-protocol_whitelist")
    assert args[i + 1] == "file", (
        "ffmpeg must be restricted to the file protocol; otherwise a crafted "
        "container can make it issue outbound requests (SSRF)"
    )
    assert "-nostdin" in args


def test_preprocess_audio_extraction_is_also_hardened():
    """The same hardening must apply on the path that actually runs."""
    import inspect

    from ddetect.data import preprocess

    src = inspect.getsource(preprocess.extract_audio)
    assert "-protocol_whitelist" in src and '"file"' in src
    assert "-nostdin" in src
    assert "shell=True" not in src


def test_no_shell_true_anywhere_in_subprocess_calls():
    """A grep-level guard: ``shell=True`` on a path that touches user input."""
    import re

    root = Path(__file__).resolve().parent.parent
    offenders = []
    for p in list((root / "ddetect").rglob("*.py")) + list((root / "api").rglob("*.py")):
        txt = p.read_text()
        if re.search(r"shell\s*=\s*True", txt):
            offenders.append(str(p.relative_to(root)))
    assert not offenders, f"shell=True found in {offenders}"
