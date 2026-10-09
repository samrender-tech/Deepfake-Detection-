"""DFDC parser (Dolhansky et al., 2020).

The only dataset here that is instantly available (Kaggle) AND carries audio,
which makes it the training set for the audio-visual protocol.

Two layouts are supported:

  preview   dfdc_preview/{dataset.json, method_A/..., original_videos/...}
  full      dfdc_train_part_N/{metadata.json, *.mp4}

``metadata.json`` maps filename -> {label: REAL|FAKE, original: <real file>,
split: train}. ``original`` is the provenance link we need: a fake and the real
video it was made from share an identity, so they must land in the same split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser, split_by_identity
from ddetect.utils.log import get_logger

log = get_logger(__name__)


class DFDCParser(DatasetParser):
    name = "dfdc"
    expected_audio = "yes, including audio-only fakes - the AV training set"

    def __init__(
        self,
        root: str | Path,
        parts: tuple[int, ...] | None = None,
        split_by_part: bool = True,
    ) -> None:
        super().__init__(root)
        self.parts = parts
        #: DFDC parts are disjoint in identity by construction, so splitting on
        #: part folder is both cheap and safe. Set False to split by identity
        #: within a single part (what you want when only part 0 was downloaded).
        self.split_by_part = split_by_part

    def _part_dirs(self) -> list[Path]:
        dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )
        if not dirs and (self.root / "metadata.json").exists():
            dirs = [self.root]
        if self.parts is not None:
            keep = {f"dfdc_train_part_{i}" for i in self.parts} | {str(i) for i in self.parts}
            dirs = [d for d in dirs if d.name in keep or d.name.endswith(tuple(f"_{i}" for i in self.parts))]
        return dirs

    def discover(self) -> Iterator[dict]:
        part_dirs = self._part_dirs()
        if not part_dirs:
            raise RuntimeError(
                f"DFDC: no part directory with metadata.json under {self.root}\n"
                "  expected e.g. dfdc_train_part_0/metadata.json"
            )

        rows: list[dict] = []
        for pd in part_dirs:
            meta = json.loads((pd / "metadata.json").read_text())
            part_name = pd.name
            for fname, info in meta.items():
                vpath = pd / fname
                if not vpath.exists():
                    continue
                is_fake = str(info.get("label", "")).upper() == "FAKE"
                original = info.get("original")

                # Identity = the real source video. A fake inherits its
                # original's identity so the two cannot be split apart; a real
                # video is its own identity.
                ident_src = original if (is_fake and original) else fname
                rows.append(
                    {
                        "video_id": f"dfdc_{part_name}_{vpath.stem}",
                        "path": str(vpath),
                        "label": 1 if is_fake else 0,
                        "dataset": "dfdc",
                        "forgery_method": "dfdc_unknown" if is_fake else "real",
                        "source_identity": f"dfdc_{Path(str(ident_src)).stem}",
                        "compression": "unknown",
                        "_part": part_name,
                    }
                )

        if self.split_by_part and len(part_dirs) >= 3:
            names = sorted({r["_part"] for r in rows})
            # Hold out the last two parts: test then val, train on the rest.
            assign = {n: "train" for n in names}
            assign[names[-1]] = "test"
            assign[names[-2]] = "val"
            for r in rows:
                r["split"] = assign[r.pop("_part")]
            log.info("DFDC split by part: %s", assign)
        else:
            sp = split_by_identity([r["source_identity"] for r in rows])
            for r in rows:
                r.pop("_part", None)
                r["split"] = sp[r["source_identity"]]
            log.info("DFDC split by identity across %d part(s)", len(part_dirs))

        n_fake = sum(r["label"] for r in rows)
        log.info("DFDC: %d videos (%d real, %d fake)", len(rows), len(rows) - n_fake, n_fake)
        yield from rows
