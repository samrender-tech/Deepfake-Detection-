"""DeeperForensics-1.0 parser (Jiang et al., CVPR 2020).

Used as a ROBUSTNESS test set rather than a generalisation one. Its value is
the perturbation suite: the same manipulated videos are released at several
distortion types and levels (compression, colour saturation, local block
noise, Gaussian blur, and mixtures), which is exactly the axis Objective 4
cares about and which no other dataset in this project provides.

Layout::

    DeeperForensics-1.0/
      source_videos/<id>/...                      real
      manipulated_videos/
        end_to_end/<id>.mp4                       standard quality
        end_to_end_level_<1..5>/<id>.mp4          distortion levels
        end_to_end_<distortion>/<id>.mp4          named distortion types

AUDIO: none. This is a visual-protocol dataset.

Default is test-only: nothing here should enter a training loader, both
because it is a held-out set and because its perturbed copies of the same
source video are near-duplicates of each other by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser, split_by_identity
from ddetect.utils.log import get_logger

log = get_logger(__name__)

#: Perturbation directories, mapped to a readable tag kept in forgery_method
#: so the robustness breakdown can group on it.
LEVEL_RE = re.compile(r"end_to_end(?:_level_(\d))?(?:_(\w+))?$")


class DeeperForensicsParser(DatasetParser):
    name = "deeperforensics"
    expected_audio = "none (visual protocol only)"

    def __init__(self, root: str | Path, test_only: bool = True) -> None:
        super().__init__(root)
        self.test_only = test_only

    @staticmethod
    def _identity(stem: str) -> str:
        # Filenames look like "<source_id>_<target_id>.mp4" or "<id>.mp4".
        parts = [p for p in stem.split("_") if p]
        return "dfo_" + "+".join(sorted(set(parts[:2]))) if len(parts) >= 2 else f"dfo_{stem}"

    def discover(self) -> Iterator[dict]:
        rows: list[dict] = []

        real_dir = self.root / "source_videos"
        if real_dir.exists():
            for p in sorted(real_dir.rglob("*.mp4")):
                rows.append({
                    "video_id": f"dfo_real_{p.stem}",
                    "path": str(p), "label": 0, "dataset": "deeperforensics",
                    "forgery_method": "real",
                    "source_identity": self._identity(p.stem),
                    "compression": "unknown",
                })
        else:
            log.warning("DeeperForensics source_videos not found at %s", real_dir)

        man_dir = self.root / "manipulated_videos"
        if not man_dir.exists():
            log.warning("DeeperForensics manipulated_videos not found at %s", man_dir)
        else:
            for sub in sorted(d for d in man_dir.iterdir() if d.is_dir()):
                m = LEVEL_RE.match(sub.name)
                if m:
                    level, distortion = m.group(1), m.group(2)
                    tag = (
                        f"dfo_level{level}" if level
                        else f"dfo_{distortion}" if distortion
                        else "dfo_end_to_end"
                    )
                else:
                    tag = f"dfo_{sub.name}"
                for p in sorted(sub.rglob("*.mp4")):
                    rows.append({
                        "video_id": f"dfo_{sub.name}_{p.stem}",
                        "path": str(p), "label": 1, "dataset": "deeperforensics",
                        "forgery_method": tag,
                        "source_identity": self._identity(p.stem),
                        "compression": "unknown",
                    })

        if self.test_only:
            for r in rows:
                r["split"] = "test"
        else:
            sp = split_by_identity([r["source_identity"] for r in rows])
            for r in rows:
                r["split"] = sp[r["source_identity"]]

        by_tag: dict[str, int] = {}
        for r in rows:
            by_tag[r["forgery_method"]] = by_tag.get(r["forgery_method"], 0) + 1
        log.info("DeeperForensics: %d videos across %d perturbation groups %s",
                 len(rows), len(by_tag), dict(list(by_tag.items())[:6]))
        yield from rows
