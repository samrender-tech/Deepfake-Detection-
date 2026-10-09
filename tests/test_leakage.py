"""F5 - the leakage gate.

This is the most load-bearing test in the repository. The project's headline
claim is a cross-dataset accuracy DROP (Baseline B). If identities or
near-duplicate videos cross a split boundary, the drop shrinks for reasons
unrelated to generalisation and the central result is void -- while every
training curve still looks perfectly healthy.

The leakage rule: "if Baseline B does not drop, suspect leakage -- this is the most
likely silent failure in the whole project."

So these tests assert two things: that real manifests are clean, AND that the
detector actually catches planted leaks. A guard that never fires is
indistinguishable from a guard that cannot fire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ddetect.data.integrity import (
    assert_clean,
    audit,
    check_identity_disjoint,
    check_label_consistency,
    find_near_duplicates,
    hamming,
    phash_frame,
)
from ddetect.data.manifest_io import read_manifest


def _row(vid: str, split: str, ident: str, label: int = 0, ds: str = "ffpp") -> dict:
    return {
        "video_id": vid,
        "path": f"/nonexistent/{vid}.mp4",
        "label": label,
        "dataset": ds,
        "forgery_method": "real" if label == 0 else "Deepfakes",
        "split": split,
        "has_audio": False,
        "source_identity": ident,
        "compression": "c23",
        "fps": 25.0,
        "n_frames": 100,
        "duration_s": 4.0,
        "sha256": "0" * 64,
    }


# ==========================================================================
# identity disjointness
# ==========================================================================
def test_detects_identity_crossing_splits():
    df = pd.DataFrame(
        [
            _row("a", "train", "person_1"),
            _row("b", "test", "person_1"),  # <- the leak
            _row("c", "train", "person_2"),
        ]
    )
    overlaps = check_identity_disjoint(df)
    assert overlaps, "an identity in both train and test was not detected"
    assert "person_1" in next(iter(overlaps.values()))


def test_clean_manifest_passes():
    df = pd.DataFrame(
        [
            _row("a", "train", "person_1"),
            _row("b", "val", "person_2"),
            _row("c", "test", "person_3"),
        ]
    )
    assert not check_identity_disjoint(df)


def test_same_identity_across_datasets_is_not_a_split_leak():
    # FF++ and Celeb-DF both scrape public video, so the same public figure
    # legitimately appears in both. That is a cross-dataset duplicate question,
    # not a within-dataset split violation, and must not fail the build.
    df = pd.DataFrame(
        [
            _row("a", "train", "person_1", ds="ffpp"),
            _row("b", "test", "person_1", ds="celebdf"),
        ]
    )
    assert not check_identity_disjoint(df)


def test_assert_clean_raises_on_leak():
    df = pd.DataFrame([_row("a", "train", "p1"), _row("b", "test", "p1")])
    with pytest.raises(AssertionError, match="integrity audit"):
        assert_clean(df, check_paths=False)


# ==========================================================================
# label consistency
# ==========================================================================
def test_detects_conflicting_labels_for_one_video():
    df = pd.DataFrame([_row("a", "train", "p1", label=0), _row("a", "val", "p2", label=1)])
    assert check_label_consistency(df)


def test_detects_real_row_naming_a_generator():
    df = pd.DataFrame([_row("a", "train", "p1", label=0)])
    df.loc[0, "forgery_method"] = "Deepfakes"  # label 0 but a generator named
    assert check_label_consistency(df)


# ==========================================================================
# perceptual near-duplicates
# ==========================================================================
def test_phash_is_stable_under_recompression():
    import cv2

    rng = np.random.default_rng(0)
    img = (rng.random((128, 128)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # some structure to hash

    h1 = phash_frame(img)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    assert ok
    h2 = phash_frame(cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE))

    # The whole point: a re-encoded copy of the same video must still match,
    # or the duplicate sweep misses exactly the leaks that matter (the same
    # source video appearing at two compression levels).
    assert hamming(h1, h2) <= 6, f"phash moved {hamming(h1, h2)} bits under JPEG q40"


def test_phash_separates_different_images():
    rng = np.random.default_rng(1)
    a = (rng.random((128, 128)) * 255).astype(np.uint8)
    b = (rng.random((128, 128)) * 255).astype(np.uint8)
    assert hamming(phash_frame(a), phash_frame(b)) > 10


def _rand_hashes(n: int, seed: int) -> list[int]:
    """n independent 64-bit perceptual hashes.

    Real phashes of unrelated frames sit ~32 bits apart. Small integers do NOT
    model that -- hamming(1, 100) is only 4 bits, which is inside the match
    threshold and would make this test pass or fail for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)]


def test_near_duplicate_search_needs_several_matching_frames():
    # One shared frame must NOT be enough: DFDC has many near-black opening
    # frames, and matching on one would flag thousands of unrelated pairs.
    shared = phash_frame((np.random.default_rng(2).random((64, 64)) * 255).astype(np.uint8))
    x = [shared, *_rand_hashes(7, 10)]
    y = [shared, *_rand_hashes(7, 11)]
    # Same leading-bits bucket, so the blocking cannot be what rejects them.
    assert x[0] >> 48 == y[0] >> 48
    assert not find_near_duplicates({"x": x, "y": y}, min_matching_frames=3), (
        "a single shared frame was enough to flag a duplicate"
    )

    same = [shared, *_rand_hashes(7, 12)]
    assert find_near_duplicates({"x": same, "y": list(same)}, min_matching_frames=3), (
        "two identical videos were not flagged as duplicates"
    )


def test_audit_flags_duplicate_across_splits_but_warns_across_datasets():
    h = _rand_hashes(8, 20)

    # same dataset, different splits -> FAIL
    df = pd.DataFrame([_row("a", "train", "p1", ds="ffpp"), _row("b", "test", "p2", ds="ffpp")])
    rep = audit(df, check_paths=False, phashes={"a": h, "b": list(h)})
    assert rep.duplicate_pairs and not rep.ok

    # different datasets -> WARN only, build still passes
    df2 = pd.DataFrame([_row("a", "train", "p1", ds="ffpp"), _row("b", "test", "p2", ds="celebdf")])
    rep2 = audit(df2, check_paths=False, phashes={"a": h, "b": list(h)})
    assert rep2.cross_dataset_dupes and rep2.ok


def test_audit_reports_missing_files():
    df = pd.DataFrame([_row("a", "train", "p1")])
    rep = audit(df, check_paths=True)
    assert rep.missing_files and not rep.ok


# ==========================================================================
# the real manifest
# ==========================================================================
def test_fixture_manifest_is_clean(fixture_manifest):
    df = read_manifest(fixture_manifest)
    rep = audit(df, check_paths=True)
    assert rep.ok, rep.render()


def test_fixture_manifest_has_all_three_splits(fixture_manifest):
    df = read_manifest(fixture_manifest)
    assert set(df.split) == {"train", "val", "test"}, (
        "the trainer's val loop and the threshold-provenance rule both need a "
        "real val split in the fixture manifest"
    )
