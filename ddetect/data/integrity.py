"""F5 - leakage and integrity guard.

This is the most important defensive code in the project. The headline claim is
a cross-dataset accuracy drop (Baseline B). If identities or near-duplicate
videos leak across splits, the drop shrinks for a reason that has nothing to do
with generalisation and the whole result is void.

The leakage rule: "Baseline B drops clearly (if it does not, suspect leakage --
this is the most likely silent failure in the whole project)". So these checks
run as a pytest that fails CI, not as a notebook someone remembers to open.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ddetect.utils.log import get_logger

log = get_logger(__name__)


@dataclass
class LeakReport:
    """Result of an integrity sweep. ``ok`` is what CI asserts on."""

    identity_overlaps: dict[str, list[str]] = field(default_factory=dict)
    duplicate_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    cross_dataset_dupes: list[tuple[str, str, int]] = field(default_factory=list)
    label_conflicts: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.identity_overlaps
            or self.duplicate_pairs
            or self.label_conflicts
            or self.missing_files
        )

    def render(self) -> str:
        if self.ok and not self.cross_dataset_dupes:
            return "integrity: clean"
        out = ["INTEGRITY REPORT"]
        for pair, ids in self.identity_overlaps.items():
            out.append(
                f"  [FAIL] identity overlap {pair}: {len(ids)} shared "
                f"source_identity -> {ids[:5]}"
            )
        if self.duplicate_pairs:
            out.append(f"  [FAIL] {len(self.duplicate_pairs)} near-duplicate pairs across splits")
            for a, b, d in self.duplicate_pairs[:5]:
                out.append(f"           {a} ~ {b}  (hamming {d})")
        if self.cross_dataset_dupes:
            out.append(
                f"  [WARN] {len(self.cross_dataset_dupes)} near-duplicates across datasets "
                "(train-on-one/test-on-other is compromised for these)"
            )
            for a, b, d in self.cross_dataset_dupes[:5]:
                out.append(f"           {a} ~ {b}  (hamming {d})")
        for c in self.label_conflicts[:10]:
            out.append(f"  [FAIL] label conflict: {c}")
        if self.missing_files:
            out.append(f"  [FAIL] {len(self.missing_files)} manifest paths do not exist")
            for m in self.missing_files[:5]:
                out.append(f"           {m}")
        out.extend(f"  [note] {n}" for n in self.notes)
        return "\n".join(out)


# --------------------------------------------------------------------------
# identity disjointness
# --------------------------------------------------------------------------
def check_identity_disjoint(df: pd.DataFrame) -> dict[str, list[str]]:
    """No ``source_identity`` may appear in two splits of the same dataset.

    Checked per dataset: the same identity legitimately appears in both FF++
    and Celeb-DF (both scrape public video), and that is a cross-dataset
    duplicate question, handled separately below.
    """
    overlaps: dict[str, list[str]] = {}
    for ds, g in df.groupby("dataset"):
        by_split = {s: set(sg.source_identity) for s, sg in g.groupby("split")}
        splits = sorted(by_split)
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                shared = sorted(by_split[a] & by_split[b])
                if shared:
                    overlaps[f"{ds}:{a}|{b}"] = shared
    return overlaps


def check_label_consistency(df: pd.DataFrame) -> list[str]:
    """A ``video_id`` must not carry two labels, and real rows must say 'real'."""
    problems: list[str] = []
    for vid, g in df.groupby("video_id"):
        if g.label.nunique() > 1:
            problems.append(f"{vid} has labels {sorted(g.label.unique())}")
    bad = df[((df.label == 0) & (df.forgery_method != "real"))
             | ((df.label == 1) & (df.forgery_method == "real"))]
    problems.extend(
        f"{r.video_id}: label={r.label} method={r.forgery_method}"
        for r in bad.itertuples()
    )
    return problems


# --------------------------------------------------------------------------
# perceptual hashing
# --------------------------------------------------------------------------
def phash_frame(gray: np.ndarray, hash_size: int = 8) -> int:
    """64-bit DCT perceptual hash of a single grayscale frame.

    Implemented here rather than pulled from ``imagehash`` to avoid a Pillow
    version pin, and because we hash numpy arrays that are already in memory
    from the preprocessing pass.
    """
    import cv2

    size = hash_size * 4
    small = cv2.resize(gray.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)
    low = dct[:hash_size, :hash_size].flatten()
    # Exclude the DC term from the median: it dominates and flattens the hash.
    med = np.median(low[1:])
    bits = (low > med).astype(np.uint64)
    out = np.uint64(0)
    for b in bits:
        out = np.uint64((out << np.uint64(1)) | b)
    return int(out)


def video_phash(video_path: str | Path, n_frames: int = 8) -> list[int]:
    """Perceptual hashes of ``n_frames`` evenly spaced frames."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(total - 1, 0), num=min(n_frames, total), dtype=int)
    hashes: list[int] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        hashes.append(phash_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    cap.release()
    return hashes


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(
    hashes: dict[str, list[int]],
    threshold: int = 6,
    min_matching_frames: int = 3,
) -> list[tuple[str, str, int]]:
    """Pairwise near-duplicate search over per-video frame hashes.

    Two videos match when at least ``min_matching_frames`` frame hashes are
    within ``threshold`` bits. Requiring several frames avoids flagging two
    unrelated videos that happen to share one dark or blank frame -- a real
    problem on DFDC, which has many near-black opening frames.

    Blocked by the first frame's top bits to keep this near-linear; an O(n^2)
    sweep over ~100k videos is not viable.
    """
    buckets: dict[int, list[str]] = defaultdict(list)
    for vid, hs in hashes.items():
        if hs:
            buckets[hs[0] >> 48].append(vid)

    pairs: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    for group in buckets.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                ha, hb = hashes[a], hashes[b]
                dists = [
                    hamming(x, y)
                    for x, y in zip(ha, hb)
                    if hamming(x, y) <= threshold
                ]
                if len(dists) >= min_matching_frames:
                    pairs.append((*key, int(np.mean(dists))))
    return pairs


# --------------------------------------------------------------------------
# the sweep CI runs
# --------------------------------------------------------------------------
def audit(
    df: pd.DataFrame,
    check_paths: bool = True,
    phashes: dict[str, list[int]] | None = None,
) -> LeakReport:
    """Full integrity sweep over a manifest.

    ``phashes`` is optional because computing it means decoding every video;
    the preprocessing pass (F2) emits them as a by-product, so Stage 2 passes
    them in and the cheap structural checks run everywhere else.
    """
    rep = LeakReport()
    rep.identity_overlaps = check_identity_disjoint(df)
    rep.label_conflicts = check_label_consistency(df)

    if check_paths:
        rep.missing_files = [
            r.path for r in df.itertuples() if not Path(r.path).exists()
        ][:200]

    if phashes:
        split_of = dict(zip(df.video_id, df.split))
        ds_of = dict(zip(df.video_id, df.dataset))
        for a, b, d in find_near_duplicates(phashes):
            sa, sb = split_of.get(a), split_of.get(b)
            da, db = ds_of.get(a), ds_of.get(b)
            if da == db and sa != sb:
                rep.duplicate_pairs.append((a, b, d))   # same dataset, split leak -> FAIL
            elif da != db:
                rep.cross_dataset_dupes.append((a, b, d))  # cross-dataset -> WARN
    else:
        rep.notes.append("perceptual-hash sweep skipped (no hashes supplied)")

    return rep


def assert_clean(df: pd.DataFrame, **kw: object) -> LeakReport:
    """Raise unless the manifest is clean. This is what the CI test calls."""
    rep = audit(df, **kw)  # type: ignore[arg-type]
    if not rep.ok:
        raise AssertionError("manifest failed the integrity audit:\n" + rep.render())
    return rep
