"""Fixture parser -- the synthetic clips in ``tests/fixtures/videos``.

Exists so the walking skeleton and ``make smoke`` can run the
entire chain with no dataset access at all. The naming convention carries the
label: ``real_*`` -> 0, ``fake_*`` -> 1, ``edge_*`` -> an edge case kept out of
train/val.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser


class FixtureParser(DatasetParser):
    name = "custom"
    expected_audio = "mixed by design: two clips with audio, two silent"

    #: variant suffix -> split, so ``make smoke`` has a real val loop.
    VARIANT_SPLIT = {0: "train", 1: "val", 2: "test"}

    def _split_for(self, stem: str) -> str:
        if "_v" in stem:
            try:
                return self.VARIANT_SPLIT.get(int(stem.rsplit("_v", 1)[1]), "train")
            except ValueError:
                pass
        return "train"

    def discover(self) -> Iterator[dict]:
        for p in sorted(Path(self.root).glob("*.mp4")):
            stem = p.stem
            if stem.startswith("real_"):
                label, method, split = 0, "real", self._split_for(stem)
            elif stem.startswith("fake_"):
                label, method, split = 1, "fixture_synthetic", self._split_for(stem)
            else:
                # Edge cases (no face, etc.) are test-only: they would poison a
                # training batch and they exist to be probed, not learned from.
                label, method, split = 0, "real", "test"
            yield {
                "video_id": f"fx_{stem}",
                "path": str(p),
                "label": label,
                "dataset": "custom",
                "forgery_method": method,
                "split": split,
                # Every fixture is its own identity: no leakage possible, and
                # the integrity test stays meaningful on this manifest.
                "source_identity": f"fx_{stem}",
                "compression": "unknown",
            }
