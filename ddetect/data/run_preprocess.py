"""CLI: populate the preprocessing cache for a manifest.

    python -m ddetect.data.run_preprocess --manifest data/manifests/ffpp.parquet

Resumable: re-running skips videos already cached at the current schema
version, so an interrupted Colab session costs nothing. ``--phash-audit``
additionally runs the near-duplicate sweep (F5) using the hashes the pass emits
as a by-product -- the only cheap moment to do it, since the frames are already
decoded.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ddetect.data.facedet import PERMISSIVE_MTCNN_THRESHOLDS
from ddetect.data.manifest_io import read_manifest
from ddetect.data.preprocess import PreprocessConfig, cache_dir_for, is_cached, preprocess_video
from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)


def _one(args: tuple[dict, str, dict]) -> tuple[str, bool, str, list[int]]:
    row, cache_root, cfg_kw = args
    cfg = PreprocessConfig(**cfg_kw)
    cdir = cache_dir_for(cache_root, row["dataset"], row["video_id"])
    try:
        meta = preprocess_video(
            row["path"], cdir, cfg, video_id=row["video_id"], dataset=row["dataset"]
        )
        note = "" if meta.n_faces_found else "no-face"
        return row["video_id"], True, note, meta.phash
    except Exception as e:  # noqa: BLE001 - one bad video must not stop the pass
        return row["video_id"], False, f"{type(e).__name__}: {e}", []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache-root", default="processed")
    ap.add_argument("--split", default=None, help="restrict to one split")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--device", default="cpu", help="face detector device (cpu|mps|cuda)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-mouth", action="store_true")
    ap.add_argument(
        "--permissive-detector", action="store_true",
        help="lower MTCNN thresholds; for the synthetic fixtures only",
    )
    ap.add_argument("--phash-audit", action="store_true", help="run the F5 duplicate sweep")
    a = ap.parse_args(argv)
    setup_logging()

    df = read_manifest(a.manifest, split=a.split)
    if a.limit:
        df = df.head(a.limit)

    todo = df if a.overwrite else df[
        [
            not is_cached(cache_dir_for(a.cache_root, r.dataset, r.video_id))
            for r in df.itertuples()
        ]
    ]
    log.info("%d/%d videos need preprocessing -> %s", len(todo), len(df), a.cache_root)
    if len(todo) == 0 and not a.phash_audit:
        return 0

    cfg_kw = {
        "n_frames": a.n_frames,
        "detector": a.detector,
        "device": a.device,
        "extract_audio": not a.no_audio,
        "extract_mouth": not a.no_mouth,
        "overwrite": a.overwrite,
        "det_thresholds": (
            PERMISSIVE_MTCNN_THRESHOLDS if a.permissive_detector else (0.6, 0.7, 0.7)
        ),
    }

    jobs = [(r._asdict(), a.cache_root, cfg_kw) for r in todo.itertuples(index=False)]
    workers = a.workers or min(os.cpu_count() or 4, 8)

    phashes: dict[str, list[int]] = {}
    failures: list[tuple[str, str]] = []
    no_face = 0

    if workers <= 1:
        it = (_one(j) for j in jobs)
        results = list(tqdm(it, total=len(jobs), desc="preprocess", unit="vid"))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for f in tqdm(as_completed(futs), total=len(futs), desc="preprocess", unit="vid"):
                results.append(f.result())

    for vid, ok, note, ph in results:
        if not ok:
            failures.append((vid, note))
        else:
            if note == "no-face":
                no_face += 1
            if ph:
                phashes[vid] = ph

    log.info("done: %d ok, %d failed, %d with no face found",
             len(results) - len(failures), len(failures), no_face)
    if failures:
        Path("results").mkdir(exist_ok=True)
        p = Path("results") / "preprocess_failures.txt"
        p.write_text("".join(f"{v}\t{e}\n" for v, e in failures))
        log.warning("failures written to %s", p)

    if a.phash_audit:
        from ddetect.data.integrity import audit

        # Load hashes for already-cached videos too, so the sweep covers the
        # whole manifest rather than only what this invocation processed.
        import json

        for r in df.itertuples():
            if r.video_id in phashes:
                continue
            mp = cache_dir_for(a.cache_root, r.dataset, r.video_id) / "meta.json"
            if mp.exists():
                try:
                    phashes[r.video_id] = json.loads(mp.read_text()).get("phash", [])
                except json.JSONDecodeError:
                    pass
        print("\n" + audit(df, check_paths=False, phashes=phashes).render())

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
