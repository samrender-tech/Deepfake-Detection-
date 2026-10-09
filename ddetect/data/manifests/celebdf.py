"""Celeb-DF v2 parser (Li et al., CVPR 2020).

Layout::

    Celeb-DF-v2/
      Celeb-real/id0_0000.mp4
      Celeb-synthesis/id0_id1_0000.mp4
      YouTube-real/00000.mp4
      List_of_testing_videos.txt        <- official test list

Used as a cross-dataset TEST set (Baseline B). The official test list is
honoured exactly; everything else becomes train/val only so that a future
fine-tuning experiment cannot accidentally touch the published test split.

AUDIO: absent or not forgery-aligned (the no-audio finding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser, split_by_identity
from ddetect.utils.log import get_logger

log = get_logger(__name__)


class CelebDFParser(DatasetParser):
    name = "celebdf"
    expected_audio = "none / not forgery-aligned - see the no-audio finding"

    def __init__(self, root: str | Path, test_only: bool = False) -> None:
        super().__init__(root)
        self.test_only = test_only
        self._official_test = self._load_test_list()

    def _load_test_list(self) -> set[str]:
        """Parse ``List_of_testing_videos.txt`` ("<label> <relpath>" per line)."""
        p = self.root / "List_of_testing_videos.txt"
        if not p.exists():
            log.warning(
                "Celeb-DF official test list missing at %s -- using a "
                "deterministic identity split instead. E2 numbers will not be "
                "comparable to published Celeb-DF results.", p
            )
            return set()
        names: set[str] = set()
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            names.add(Path(parts[-1]).name)
        log.info("Celeb-DF official test list: %d videos", len(names))
        return names

    @staticmethod
    def _identity(stem: str, kind: str) -> str:
        """``id0_0000`` -> id0; ``id0_id1_0000`` -> id0+id1 (both involved)."""
        if kind == "youtube":
            return f"celebdf_yt_{stem}"
        bits = [b for b in stem.split("_") if b.startswith("id")]
        if not bits:
            return f"celebdf_{stem}"
        return "celebdf_" + "+".join(sorted(set(bits)))

    def discover(self) -> Iterator[dict]:
        sources = [
            ("Celeb-real", 0, "real", "celeb"),
            ("YouTube-real", 0, "real", "youtube"),
            ("Celeb-synthesis", 1, "celebdf_synthesis", "celeb"),
        ]
        rows: list[dict] = []
        for subdir, label, method, kind in sources:
            d = self.root / subdir
            if not d.exists():
                log.warning("Celeb-DF subdir missing: %s", d)
                continue
            for p in sorted(d.glob("*.mp4")):
                rows.append(
                    {
                        "video_id": f"celebdf_{subdir.lower()}_{p.stem}",
                        "path": str(p),
                        "label": label,
                        "dataset": "celebdf",
                        "forgery_method": method,
                        "source_identity": self._identity(p.stem, kind),
                        "compression": "unknown",
                        "_name": p.name,
                    }
                )

        if self._official_test:
            fallback = split_by_identity(
                [r["source_identity"] for r in rows], ratios=(0.8, 0.2, 0.0)
            )
            for r in rows:
                r["split"] = (
                    "test" if r.pop("_name") in self._official_test
                    else fallback[r["source_identity"]]
                )
        else:
            sp = split_by_identity([r["source_identity"] for r in rows])
            for r in rows:
                r.pop("_name", None)
                r["split"] = sp[r["source_identity"]]

        if self.test_only:
            # Cross-dataset evaluation only: relabel everything as test so no
            # Celeb-DF video can ever enter a training loader by accident.
            for r in rows:
                r["split"] = "test"

        n_test = sum(r["split"] == "test" for r in rows)
        log.info("Celeb-DF: %d videos, %d in test", len(rows), n_test)
        yield from rows
