"""F31 - upload validation and hardening.

This endpoint accepts arbitrary video from the internet and hands it to
ffmpeg and OpenCV. The threat model (docs/THREAT_MODEL.md) treats every
upload as hostile.

Checks, in order of how much they matter:

1. **Content sniffing, never the extension.** A ``.mp4`` suffix means
   nothing. We read the container magic bytes.
2. **Size and duration caps**, enforced while streaming to disk so a
   declared-small/actually-huge upload cannot exhaust the disk.
3. **Path-traversal-proof storage keys.** The client's filename never
   reaches the filesystem; a random id does.
4. **ffmpeg with the protocol whitelist emptied**, so a crafted container
   cannot make it fetch a URL (the classic SSRF-via-ffmpeg concat trick).
"""

from __future__ import annotations

import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_DURATION_S = float(os.environ.get("MAX_DURATION_S", "60"))
CHUNK = 1 << 20

#: Container magic. Checked against the first bytes of the upload.
#: (offset, bytes) -- ISO-BMFF puts 'ftyp' at offset 4.
MAGIC = (
    (4, b"ftyp"),  # mp4 / mov / m4v
    (0, b"\x1aE\xdf\xa3"),  # matroska / webm
    (0, b"RIFF"),  # avi
    (0, b"FLV"),  # flv
)

ALLOWED_VIDEO_CODECS = {
    "h264",
    "hevc",
    "vp8",
    "vp9",
    "av1",
    "mpeg4",
    "mjpeg",
    "theora",
}


class UploadRejected(Exception):
    """Raised with a message safe to return to the client."""


@dataclass
class UploadInfo:
    key: str
    path: Path
    size_bytes: int
    duration_s: float
    width: int
    height: int
    vcodec: str
    has_audio: bool


def new_storage_key() -> str:
    """A random, filesystem-safe key. The client's filename is never used."""
    return secrets.token_hex(16)


def safe_display_name(filename: str | None) -> str:
    """Sanitise a filename for DISPLAY only; never for a path."""
    if not filename:
        return "upload"
    base = Path(filename).name  # strip any directory parts
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120]
    return base or "upload"


def sniff_container(head: bytes) -> bool:
    return any(head[off : off + len(sig)] == sig for off, sig in MAGIC)


class _Readable(Protocol):
    """The slice of FastAPI's UploadFile we actually use."""

    async def read(self, size: int = -1) -> bytes: ...


async def stream_to_disk(upload: _Readable, dest: Path, max_mb: int = MAX_UPLOAD_MB) -> int:
    """Stream an UploadFile to disk, enforcing the size cap as we go.

    The cap is checked per chunk rather than from ``Content-Length``, which a
    client controls and can lie about.
    """
    limit = max_mb * (1 << 20)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    head = b""
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(CHUNK)
            if not chunk:
                break
            if not head:
                head = chunk[:16]
                if not sniff_container(head):
                    dest.unlink(missing_ok=True)
                    raise UploadRejected(
                        "Not a recognised video container. Supported: MP4/MOV, "
                        "WebM/Matroska, AVI, FLV. (A file extension is not checked; "
                        "the container header is.)"
                    )
            total += len(chunk)
            if total > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise UploadRejected(f"Upload exceeds the {max_mb} MB limit.")
            fh.write(chunk)
    if total == 0:
        dest.unlink(missing_ok=True)
        raise UploadRejected("Empty upload.")
    return total


def probe_and_validate(path: Path, size_bytes: int) -> UploadInfo:
    """ffprobe the stored file and enforce the content policy."""
    from ddetect.data.manifest_io import probe_video

    try:
        p = probe_video(path)
    except RuntimeError as e:
        raise UploadRejected(f"Could not decode this file as video. ({e})") from e

    if p["duration_s"] > MAX_DURATION_S:
        raise UploadRejected(
            f"Clip is {p['duration_s']:.1f}s; the limit is {MAX_DURATION_S:.0f}s. "
            "Trim it and try again."
        )
    if p["width"] <= 0 or p["height"] <= 0:
        raise UploadRejected("Video has no usable picture dimensions.")
    if min(p["width"], p["height"]) < 64:
        raise UploadRejected(f"Video is {p['width']}x{p['height']}; too small for face analysis.")
    if p["vcodec"] not in ALLOWED_VIDEO_CODECS:
        raise UploadRejected(f"Unsupported video codec {p['vcodec']!r}.")

    return UploadInfo(
        key=path.stem,
        path=path,
        size_bytes=size_bytes,
        duration_s=p["duration_s"],
        width=p["width"],
        height=p["height"],
        vcodec=p["vcodec"],
        has_audio=p["has_audio"],
    )


def storage_path(root: Path, key: str) -> Path:
    """Resolve a storage key under ``root``, refusing anything that escapes it.

    ``key`` comes from a URL path parameter in the download handlers, so this
    is the last line of defence against ``../../etc/passwd``.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key):
        raise UploadRejected("Invalid storage key.")
    root = root.resolve()
    p = (root / key).resolve()
    if not str(p).startswith(str(root) + os.sep) and p != root:
        raise UploadRejected("Invalid storage key.")
    return p


def ffmpeg_hardened_args() -> list[str]:
    """Baseline ffmpeg flags used anywhere we touch an upload.

    ``-protocol_whitelist file`` is the important one: without it a crafted
    container (an HLS playlist, a concat demuxer script) can make ffmpeg issue
    outbound HTTP requests from inside our network.
    """
    return [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file",
        "-loglevel",
        "error",
    ]


def verify_ffmpeg_available() -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, check=True
        )
        return True, out.stdout.splitlines()[0]
    except Exception as e:
        return False, str(e)
