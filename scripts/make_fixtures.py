#!/usr/bin/env python3
"""Generate the synthetic test fixtures.

Committed fixtures must be tiny and legally unencumbered, so we cannot ship
frames of real people. These clips are drawn procedurally: a moving
face-like oval with blinking eyes and an opening/closing mouth, plus a tone
whose envelope either tracks the mouth (``sync``) or does not (``desync``).

They exercise the pipeline's control flow -- decode, crop, mouth extraction,
audio presence, modality masking, the no-face and no-audio branches -- which is
what ``make smoke`` and most unit tests actually need. They are NOT a test of
detection accuracy; that needs real video and lives behind the ``slow`` marker.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 384, 288, 25
FFMPEG = "ffmpeg"


def draw_face(t: float, mouth_open: float, jitter: float = 0.0) -> np.ndarray:
    """One frame: skin-tone oval, eyes, mouth whose aperture is `mouth_open`."""
    img = np.full((H, W, 3), 40, dtype=np.uint8)
    img[:, :, 0] = 60  # faint blue-grey background

    cx = int(W / 2 + 8 * np.sin(t * 1.3) + jitter)
    cy = int(H / 2 + 4 * np.cos(t * 0.9))
    fw, fh = 92, 116

    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, (190, 160, 140), -1)
    # brow/hair mass gives the detector some vertical gradient to work with
    cv2.ellipse(img, (cx, cy - fh + 24), (fw - 4, 32), 0, 0, 360, (60, 45, 40), -1)

    blink = abs(np.sin(t * 0.8)) < 0.05
    eye_h = 3 if blink else 11
    for dx in (-33, 33):
        cv2.ellipse(img, (cx + dx, cy - 24), (17, eye_h), 0, 0, 360, (250, 250, 250), -1)
        if not blink:
            cv2.circle(img, (cx + dx, cy - 24), 6, (35, 30, 28), -1)

    cv2.line(img, (cx, cy - 10), (cx, cy + 18), (150, 120, 105), 4)  # nose

    m_h = int(4 + 19 * np.clip(mouth_open, 0, 1))
    cv2.ellipse(img, (cx, cy + 52), (30, m_h), 0, 0, 360, (90, 50, 55), -1)
    if m_h > 7:
        cv2.ellipse(img, (cx, cy + 52), (23, m_h - 5), 0, 0, 360, (40, 20, 25), -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def mouth_envelope(t: float, kind: str) -> float:
    """Mouth aperture over time. `desync` shifts the speech pattern in time."""
    base = 0.5 * (1 + np.sin(t * 7.5)) * (0.5 * (1 + np.sin(t * 1.1)))
    if kind == "desync":
        base = 0.5 * (1 + np.sin((t + 0.45) * 7.5)) * (0.5 * (1 + np.sin((t + 0.45) * 1.1)))
    return float(base)


def write_clip(
    out: Path,
    seconds: float,
    kind: str,
    audio: bool,
    faceless: bool = False,
    jitter: float = 0.0,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")

    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        if faceless:
            # Structured noise: no face anywhere. Exercises the no-face branch.
            rng = np.random.default_rng(i)
            frame = (rng.random((H, W, 3)) * 70 + 30).astype(np.uint8)
            cv2.rectangle(frame, (48, 48), (336, 240), (120, 120, 120), 3)
        else:
            frame = draw_face(t, mouth_envelope(t, kind), jitter=jitter)
        vw.write(frame)
    vw.release()

    if audio:
        # Tone whose amplitude follows the (possibly shifted) mouth envelope,
        # so SyncNet-style correlation has something real to find.
        sr = 16000
        ts = np.arange(int(seconds * sr)) / sr
        env = np.array([mouth_envelope(t, kind) for t in ts])
        sig = 0.3 * env * np.sin(2 * np.pi * 180 * ts)
        sig += 0.12 * env * np.sin(2 * np.pi * 420 * ts)
        sig += 0.01 * np.random.default_rng(0).standard_normal(ts.size)
        wav = out.with_suffix(".wav")
        import soundfile as sf

        sf.write(str(wav), sig.astype(np.float32), sr)
        cmd = [
            FFMPEG,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        wav.unlink()
    else:
        cmd = [
            FFMPEG,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "26",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    raw.unlink()
    print(f"  {out.name:28s} {out.stat().st_size / 1024:6.1f} KB  audio={audio}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/videos")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument(
        "--variants",
        type=int,
        default=3,
        help="variants per class. The fixture parser maps variant 0->train, "
        "1->val, 2->test, so >=3 is needed for the trainer's val loop.",
    )
    a = ap.parse_args()
    out = Path(a.out)

    print("generating fixtures:")
    for v in range(a.variants):
        # Jitter shifts the face horizontally so the variants are not byte
        # identical -- otherwise the duplicate sweep would (correctly) flag
        # them as a train/val leak.
        jit = (v - a.variants // 2) * 11
        # real_* : authentic, lips track the audio
        write_clip(out / f"real_sync_audio_v{v}.mp4", a.seconds, "sync", True, jitter=jit)
        write_clip(out / f"real_silent_v{v}.mp4", a.seconds, "sync", False, jitter=jit)
        # fake_* : stands in for a forgery whose face and voice came from
        # different tools -- the AV desynchronisation the sync stream looks for
        write_clip(out / f"fake_desync_audio_v{v}.mp4", a.seconds, "desync", True, jitter=jit)
        write_clip(out / f"fake_silent_v{v}.mp4", a.seconds, "desync", False, jitter=jit)
    # edge cases named in the edge-case list
    write_clip(out / "edge_no_face.mp4", 1.0, "sync", audio=True, faceless=True)
    print(f"\nwrote {len(list(out.glob('*.mp4')))} clips to {out}")


if __name__ == "__main__":
    main()
