"""Re-encode stability: does the verdict survive a social-media re-encode?

A clip that is forwarded through a messaging app or re-uploaded is recompressed,
often more than once. A verdict that flips under that treatment was resting on
compression artefacts rather than on the manipulation, and the user should be
told so. This module re-encodes a clip at the same CRF levels the degradation
grid trains and evaluates on, scores every variant with the SAME
``Detector``, and reports whether they agree.

It never edits faces or audio: the variants are plain lossy re-encodes of the
original, so this stays inside the project's detection-only scope.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ddetect.contracts import Result

#: name -> extra ffmpeg video arguments. CRF 32/40 match the ``crf32``/``crf40``
#: compression tags in ``contracts.CompressionTag``; ``half_res`` mimics the
#: downscale most platforms apply to uploads.
DEFAULT_VARIANTS: dict[str, list[str]] = {
    "crf32": ["-crf", "32"],
    "crf40": ["-crf", "40"],
    "half_res": ["-crf", "28", "-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2"],
}

#: Above this spread in calibrated probability the result is called unstable
#: even when every variant lands on the same verdict label.
MAX_PROB_SPREAD = 0.15


@dataclass
class VariantScore:
    name: str
    verdict: str
    calibrated_prob: float
    error: str = ""


@dataclass
class StabilityReport:
    original_verdict: str
    original_prob: float
    variants: list[VariantScore] = field(default_factory=list)

    @property
    def scored(self) -> list[VariantScore]:
        return [v for v in self.variants if not v.error]

    @property
    def verdict_agreement(self) -> float:
        """Fraction of successfully scored variants matching the original verdict."""
        ok = self.scored
        if not ok:
            return 0.0
        return sum(v.verdict == self.original_verdict for v in ok) / len(ok)

    @property
    def prob_spread(self) -> float:
        probs = [self.original_prob] + [v.calibrated_prob for v in self.scored]
        return max(probs) - min(probs)

    @property
    def stable(self) -> bool:
        """True only if at least one variant was scored, all agree, and the
        probability barely moved. A check that could not run is not a pass."""
        return (
            bool(self.scored)
            and self.verdict_agreement == 1.0
            and self.prob_spread <= MAX_PROB_SPREAD
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_verdict": self.original_verdict,
            "original_prob": round(self.original_prob, 4),
            "variants": [asdict(v) for v in self.variants],
            "verdict_agreement": round(self.verdict_agreement, 4),
            "prob_spread": round(self.prob_spread, 4),
            "stable": self.stable,
        }

    def summary(self) -> str:
        if not self.scored:
            return "stability check could not run: no re-encoded variant was scored"
        word = "stable" if self.stable else "UNSTABLE"
        return (
            f"{word} under re-encoding: {self.verdict_agreement:.0%} of variants agree, "
            f"probability spread {self.prob_spread:.1%}"
        )


def reencode(src: Path, dst: Path, video_args: list[str]) -> None:
    """Lossy H.264 re-encode; audio is re-encoded to AAC so it survives too."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        *video_args,
        "-c:a", "aac", "-b:a", "64k",
        str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-300:]}")


def stability_check(
    detector: Any,
    video: str | Path,
    original: Result | None = None,
    variants: dict[str, list[str]] | None = None,
) -> StabilityReport:
    """Score ``video`` and each re-encoded variant with ``detector``.

    ``original`` may be passed to avoid scoring the source clip twice. A
    variant that fails (ffmpeg error, undecodable output) is recorded with its
    error rather than silently dropped, and counts against ``stable``.
    """
    video = Path(video)
    if original is None:
        original = detector.predict(video, explain=False)
    rep = StabilityReport(original.verdict.value, float(original.calibrated_prob))

    with tempfile.TemporaryDirectory(prefix="ddetect_stab_") as tmp:
        for name, args in (variants or DEFAULT_VARIANTS).items():
            # Same stem as the source so the variant gets the same video_id.
            dst = Path(tmp) / name / f"{video.stem}.mp4"
            dst.parent.mkdir()
            try:
                reencode(video, dst, args)
                r = detector.predict(dst, explain=False)
                rep.variants.append(VariantScore(name, r.verdict.value, float(r.calibrated_prob)))
            except Exception as e:  # recorded, see docstring
                rep.variants.append(VariantScore(name, "error", 0.0, f"{type(e).__name__}: {e}"))
    return rep
