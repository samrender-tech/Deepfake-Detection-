"""FaceForensics++ parser (Rossler et al., ICCV 2019).

Layout after ``download_ffpp.py``::

    FaceForensics++/
      original_sequences/youtube/{raw,c23,c40}/videos/000.mp4
      manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/
          {raw,c23,c40}/videos/000_003.mp4

Identity: a fake named ``000_003`` swaps source 003 onto target 000. BOTH
identities are involved, so the identity key is the sorted pair -- using only
the target would let 000's face appear in train (as 000_003) and test (as
000_007) and the split would not be identity-disjoint.

AUDIO: FF++ videos carry no usable audio stream. This is the empirical basis
for the no-audio finding and ``expected_audio`` below; ``build_manifest --audit``
prints what ffprobe actually found so the claim is evidence, not folklore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ddetect.data.manifests.base import DatasetParser
from ddetect.utils.log import get_logger

log = get_logger(__name__)

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMPRESSIONS = {"raw": "raw", "c23": "c23", "c40": "c40"}


class FFPPParser(DatasetParser):
    name = "ffpp"
    expected_audio = "none (silent videos) - see the no-audio finding"

    def __init__(
        self,
        root: str | Path,
        compression: str = "c23",
        methods: tuple[str, ...] = METHODS,
        splits_dir: str | Path | None = None,
    ) -> None:
        super().__init__(root)
        if compression not in COMPRESSIONS:
            raise ValueError(f"compression must be one of {sorted(COMPRESSIONS)}")
        self.compression = compression
        self.methods = methods
        self.splits_dir = Path(splits_dir) if splits_dir else self.root / "splits"
        self._split_of = self._load_official_splits()

    def _load_official_splits(self) -> dict[str, str]:
        """Load the official identity-disjoint split JSONs.

        FF++ ships ``train.json``/``val.json``/``test.json``, each a list of
        ``[target, source]`` id pairs. These are authoritative: hand-rolling a
        split here would make our numbers incomparable to every published
        FF++ result, which defeats experiment E1 (reproduce the literature).
        """
        out: dict[str, str] = {}
        found = []
        for split in ("train", "val", "test"):
            p = self.splits_dir / f"{split}.json"
            if not p.exists():
                continue
            found.append(split)
            for pair in json.loads(p.read_text()):
                for vid in pair:
                    out[str(vid)] = split
        if not out:
            log.warning(
                "FF++ official splits not found under %s -- falling back to a "
                "deterministic identity hash. Numbers will NOT be comparable to "
                "published FF++ results; download the splits before running E1.",
                self.splits_dir,
            )
        else:
            log.info("FF++ official splits loaded (%s): %d ids", "+".join(found), len(out))
        return out

    def _split_for(self, ids: tuple[str, ...]) -> str:
        if self._split_of:
            # All ids involved must agree, else the pair straddles the split
            # boundary and we drop it rather than guess.
            labels = {self._split_of.get(i) for i in ids}
            labels.discard(None)
            if len(labels) == 1:
                return labels.pop()  # type: ignore[return-value]
            return ""   # straddles -> caller skips
        from ddetect.data.manifests.base import split_by_identity

        return split_by_identity(["_".join(sorted(ids))])["_".join(sorted(ids))]

    def discover(self) -> Iterator[dict]:
        comp = self.compression
        n_real = n_fake = n_skip = 0

        real_dir = self.root / "original_sequences" / "youtube" / comp / "videos"
        if real_dir.exists():
            for p in sorted(real_dir.glob("*.mp4")):
                vid = p.stem                      # "000"
                split = self._split_for((vid,))
                if not split:
                    n_skip += 1
                    continue
                n_real += 1
                yield {
                    "video_id": f"ffpp_real_{comp}_{vid}",
                    "path": str(p),
                    "label": 0,
                    "dataset": "ffpp",
                    "forgery_method": "real",
                    "split": split,
                    "source_identity": f"ffpp_{vid}",
                    "compression": comp,
                }
        else:
            log.warning("FF++ real videos not found at %s", real_dir)

        for method in self.methods:
            mdir = self.root / "manipulated_sequences" / method / comp / "videos"
            if not mdir.exists():
                log.warning("FF++ method %s missing at %s", method, mdir)
                continue
            for p in sorted(mdir.glob("*.mp4")):
                parts = p.stem.split("_")         # "000_003"
                ids = tuple(parts[:2]) if len(parts) >= 2 else (parts[0],)
                split = self._split_for(ids)
                if not split:
                    n_skip += 1
                    continue
                n_fake += 1
                yield {
                    "video_id": f"ffpp_{method}_{comp}_{p.stem}",
                    "path": str(p),
                    "label": 1,
                    "dataset": "ffpp",
                    "forgery_method": method,
                    "split": split,
                    # Pair key: both identities are present in the forgery.
                    "source_identity": "ffpp_" + "+".join(sorted(ids)),
                    "compression": comp,
                }

        log.info("FF++ %s: %d real, %d fake, %d skipped (split straddle)",
                 comp, n_real, n_fake, n_skip)
