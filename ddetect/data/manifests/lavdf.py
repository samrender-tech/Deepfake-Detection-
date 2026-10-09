"""LAV-DF parser (Cai et al., 2022) - Localized Audio-Visual DeepFake.

The backup audio-visual test set, and interesting for a reason
FakeAVCeleb is not: its forgeries are **temporally localised**. Only a short
segment of a clip is manipulated, with precise frame-level annotations of
where.

That matters for this project specifically. Our per-frame timeline and the
sync curve both claim to localise evidence, and LAV-DF is the only dataset
here that can check whether the frames the model flags are the frames that
were actually edited. A video-level AUC cannot test that at all.

Layout::

    LAV-DF/
      metadata.json           list of {file, n_fakes, fake_periods, modify_video,
                                       modify_audio, split, video_frames, ...}
      train/ dev/ test/       the mp4 files

``fake_periods`` is kept in forgery_method only as a coarse tag; the precise
intervals are written alongside the manifest for the localisation analysis,
since the manifest schema is deliberately one row per video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser
from ddetect.utils.log import get_logger

log = get_logger(__name__)

SPLIT_MAP = {"train": "train", "dev": "val", "val": "val", "test": "test"}


class LAVDFParser(DatasetParser):
    name = "lavdf"
    expected_audio = "yes, with temporally localised audio and/or video forgeries"

    def __init__(self, root: str | Path, test_only: bool = True) -> None:
        super().__init__(root)
        #: Default test-only: this is a held-out cross-dataset AV set.
        self.test_only = test_only
        self.periods: dict[str, list] = {}

    def _metadata(self) -> list[dict]:
        for name in ("metadata.json", "metadata.min.json"):
            p = self.root / name
            if p.exists():
                data = json.loads(p.read_text())
                return data if isinstance(data, list) else list(data.values())
        raise FileNotFoundError(
            f"LAV-DF metadata.json not found under {self.root}. The dataset "
            f"ships one; without it the fake intervals are unavailable."
        )

    def discover(self) -> Iterator[dict]:
        meta = self._metadata()
        rows: list[dict] = []
        n_missing = 0

        for item in meta:
            rel = item.get("file") or item.get("filename") or ""
            path = self.root / rel
            if not path.exists():
                n_missing += 1
                continue

            periods = item.get("fake_periods") or []
            mod_v = bool(item.get("modify_video", False))
            mod_a = bool(item.get("modify_audio", False))
            label = 1 if (periods or mod_v or mod_a) else 0

            if label == 0:
                method = "real"
            elif mod_v and mod_a:
                method = "lavdf_both"
            elif mod_a:
                method = "lavdf_audio"
            elif mod_v:
                method = "lavdf_video"
            else:
                method = "lavdf_unknown"

            vid = f"lavdf_{Path(rel).stem}"
            self.periods[vid] = periods

            rows.append({
                "video_id": vid,
                "path": str(path),
                "label": label,
                "dataset": "lavdf",
                "forgery_method": method,
                # LAV-DF derives fakes from a source speaker; the split field in
                # its own metadata is identity-disjoint, so honour it.
                "source_identity": f"lavdf_{item.get('original') or Path(rel).stem}",
                "compression": "unknown",
                "split": "test" if self.test_only
                         else SPLIT_MAP.get(str(item.get("split", "test")), "test"),
            })

        if n_missing:
            log.warning("LAV-DF: %d entries in metadata have no file on disk", n_missing)

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["forgery_method"]] = counts.get(r["forgery_method"], 0) + 1
        log.info("LAV-DF: %d videos %s", len(rows), counts)

        # The fake intervals do not fit the one-row-per-video manifest schema,
        # so they go beside it for the localisation analysis.
        if self.periods:
            out = Path("data/manifests/lavdf_fake_periods.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self.periods, indent=2))
            log.info("LAV-DF: wrote fake intervals to %s "
                     "(for per-frame localisation analysis)", out)

        yield from rows
