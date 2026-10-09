"""Shared parser scaffolding.

A parser's only job is to turn a dataset's on-disk layout into
``ManifestRow``s. Everything after that -- splits, sampling, augmentation,
caching -- is dataset-agnostic, which is the property that lets a new forgery
dataset become a new test set without retraining.

Each parser must derive ``source_identity`` honestly: it is the key the leakage
guard groups on, so a parser that returns the filename as the identity silently
defeats F5.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Sequence

from tqdm import tqdm

from ddetect.contracts import ManifestRow
from ddetect.data.manifest_io import probe_video, sha256_file
from ddetect.utils.log import get_logger

log = get_logger(__name__)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


class DatasetParser(ABC):
    """Base parser. Subclasses implement :meth:`discover`."""

    name: str = "custom"
    #: Documented audio situation, surfaced by ``build_manifest --audit``.
    #: The no-audio finding: FF++ and Celeb-DF are silent, which is *the* reason
    #: the audio-visual protocol runs on DFDC -> FakeAVCeleb instead.
    expected_audio: str = "unknown"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.name}: dataset root not found: {self.root}\n"
                f"  see docs/DATASET_CARD.md for the download/access steps"
            )

    # ---- subclass contract ------------------------------------------------
    @abstractmethod
    def discover(self) -> Iterator[dict]:
        """Yield partial rows: path, label, forgery_method, source_identity,
        split, compression. The probe fields are filled in by :meth:`build`.
        """

    # ---- shared ------------------------------------------------------------
    def build(self, workers: int = 0, limit: int | None = None) -> list[ManifestRow]:
        """Probe every discovered video and assemble validated rows.

        ffprobe is IO-bound and per-file, so this parallelises across
        processes; a DFDC part has ~2,000 videos and serial probing is minutes.
        """
        partials = list(self.discover())
        if limit:
            partials = partials[:limit]
        if not partials:
            raise RuntimeError(f"{self.name}: discovered 0 videos under {self.root}")

        workers = workers or min(os.cpu_count() or 4, 8)
        rows: list[ManifestRow] = []
        failures: list[tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_probe_one, p): p for p in partials}
            for fut in tqdm(
                as_completed(futs), total=len(futs), desc=f"probe {self.name}", unit="vid"
            ):
                partial = futs[fut]
                try:
                    rows.append(ManifestRow(**fut.result()))
                except Exception as e:  # noqa: BLE001 - one bad file must not kill the build
                    failures.append((partial["path"], f"{type(e).__name__}: {e}"))

        if failures:
            log.warning(
                "%s: skipped %d unreadable/invalid videos (first 3: %s)",
                self.name, len(failures), [f[0] for f in failures[:3]],
            )
            Path("results").mkdir(exist_ok=True)
            with (Path("results") / f"{self.name}_probe_failures.txt").open("w") as fh:
                fh.writelines(f"{p}\t{e}\n" for p, e in failures)

        return rows


def _probe_one(partial: dict) -> dict:
    """Module-level so ProcessPoolExecutor can pickle it."""
    probe = probe_video(partial["path"])
    return {
        **partial,
        "fps": probe["fps"],
        "n_frames": probe["n_frames"],
        "duration_s": probe["duration_s"],
        "has_audio": probe["has_audio"],
        "sha256": sha256_file(partial["path"]),
    }


def iter_videos(root: Path, exts: Sequence[str] = VIDEO_EXTS) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def split_by_identity(
    identities: Sequence[str], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, str]:
    """Deterministic identity-disjoint split.

    Used only by datasets that ship no official split (DFDC parts,
    FakeAVCeleb). Splitting on identity rather than video is non-negotiable:
    a video-level split puts the same person's real and fake clips on both
    sides and the model learns the person, not the forgery.
    """
    import hashlib

    uniq = sorted(set(identities))
    tr, va, _ = ratios
    out: dict[str, str] = {}
    for ident in uniq:
        h = hashlib.sha256(f"{seed}:{ident}".encode()).digest()
        frac = int.from_bytes(h[:4], "big") / 2**32
        out[ident] = "train" if frac < tr else ("val" if frac < tr + va else "test")
    return out
