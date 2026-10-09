"""FakeAVCeleb parser (Khalid et al., NeurIPS 2021 D&B).

Added because FF++ and Celeb-DF are silent, so the audio-visual objective has
nowhere to be evaluated with those two datasets alone. FakeAVCeleb is the cross-dataset AV test set -- it carries all
four combinations, which is exactly what the modality mask needs to be tested
against:

    RealVideo-RealAudio     label 0
    FakeVideo-RealAudio     label 1   visual-only forgery
    RealVideo-FakeAudio     label 1   voice clone over a real face  <- the case
                                      a video-only detector cannot see at all
    FakeVideo-FakeAudio     label 1   both

Directory tree is ``<combination>/<race>/<gender>/<id>/<clip>.mp4``; the
identity is the ``id`` directory, and the demographic path components are kept
so the model card can report per-group performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser, split_by_identity
from ddetect.utils.log import get_logger

log = get_logger(__name__)

COMBOS = {
    "RealVideo-RealAudio": (0, "real"),
    "FakeVideo-RealAudio": (1, "fav_fake_video"),
    "RealVideo-FakeAudio": (1, "fav_fake_audio"),
    "FakeVideo-FakeAudio": (1, "fav_fake_both"),
}


class FakeAVCelebParser(DatasetParser):
    name = "fakeavceleb"
    expected_audio = "yes, all four video/audio real-fake combinations"

    def __init__(self, root: str | Path, test_only: bool = True) -> None:
        super().__init__(root)
        #: Default True: this is a held-out cross-dataset test set, and the
        #: test-set firewall is easier to honour if nothing here can
        #: enter a training loader in the first place.
        self.test_only = test_only

    def discover(self) -> Iterator[dict]:
        rows: list[dict] = []
        for combo, (label, method) in COMBOS.items():
            cdir = self.root / combo
            if not cdir.exists():
                log.warning("FakeAVCeleb combination missing: %s", cdir)
                continue
            for p in sorted(cdir.rglob("*.mp4")):
                rel = p.relative_to(cdir).parts
                # <race>/<gender>/<id>/<clip>.mp4 -- id is the directory above the file
                ident = rel[-2] if len(rel) >= 2 else p.stem
                demo = "/".join(rel[:-2]) if len(rel) > 2 else "unknown"
                rows.append(
                    {
                        "video_id": f"fav_{combo}_{ident}_{p.stem}",
                        "path": str(p),
                        "label": label,
                        "dataset": "fakeavceleb",
                        "forgery_method": method,
                        "source_identity": f"fav_{ident}",
                        "compression": "unknown",
                        "_demo": demo,
                    }
                )

        if self.test_only:
            for r in rows:
                r["split"] = "test"
                r.pop("_demo", None)
        else:
            sp = split_by_identity([r["source_identity"] for r in rows])
            for r in rows:
                r["split"] = sp[r["source_identity"]]
                r.pop("_demo", None)

        by = {}
        for r in rows:
            by[r["forgery_method"]] = by.get(r["forgery_method"], 0) + 1
        log.info("FakeAVCeleb: %d videos %s", len(rows), by)
        yield from rows
