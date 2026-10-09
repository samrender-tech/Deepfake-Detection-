#!/usr/bin/env python3
"""Offline degradation for the robustness grid.

Produces a genuinely re-encoded copy of a test set and a manifest pointing at
it, so the robustness table is a reporting axis rather than a retrain.

Why offline, when the training pipeline already applies JPEG compression
online: **they are not the same artefact.** Online `ImageCompression` is
intra-frame only -- it has no inter-frame prediction, no motion compensation
and no GOP structure. Claiming H.264 robustness from JPEG augmentation alone
would be dishonest, so the evaluation uses real video codecs.

SCOPE: every transformation here is a lossy operation on existing
real media. Nothing is generated.

    python scripts/degrade_dataset.py --manifest data/manifests/celebdf.parquet \
        --recipe crf40 --out-root data/degraded

The recipes come from the red-team agent, so a recipe added there is
immediately runnable here.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from agents.a8_redteam import RECIPES, apply_recipe
from ddetect.data.manifest_io import read_manifest, write_manifest
from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)

BY_NAME = {r.name: r for r in RECIPES}

#: CompressionTag values the manifest schema accepts (contracts.py). A recipe
#: outside this set is recorded as "unknown" rather than silently breaking
#: validation.
TAGS = {"crf23": "crf23", "crf32": "crf32", "crf40": "crf40"}


def _one(job: tuple[dict, str, str, str | None]) -> tuple[str, bool, str]:
    row, recipe_name, out_root, partner = job
    recipe = BY_NAME[recipe_name]
    src = Path(row["path"])
    dst = Path(out_root) / recipe_name / row["dataset"] / f"{row['video_id']}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return row["video_id"], True, str(dst)
    ok = apply_recipe(src, dst, recipe, Path(partner) if partner else None)
    return row["video_id"], ok, str(dst) if ok else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--recipe", required=True, choices=sorted(BY_NAME))
    ap.add_argument("--out-root", default="data/degraded")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--split", default=None, help="restrict to one split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    setup_logging()

    if not shutil.which("ffmpeg"):
        log.error("ffmpeg not found on PATH")
        return 2

    recipe = BY_NAME[a.recipe]
    df = read_manifest(a.manifest, split=a.split)
    if a.limit:
        df = df.head(a.limit)
    log.info("degrading %d videos with recipe %r (%s)", len(df), a.recipe, recipe.kind)
    log.info("  rationale: %s", recipe.rationale)
    log.info("  expected:  %s", recipe.expect)

    # audio_swap needs a second real clip; pair each video with the next one so
    # the pairing is deterministic and every clip keeps a real partner.
    partners: dict[str, str | None] = {}
    if recipe.needs_second_clip:
        paths = df.path.tolist()
        if len(paths) < 2:
            log.error("recipe %r needs at least two videos to pair", a.recipe)
            return 2
        partners = {r.video_id: paths[(i + 1) % len(paths)] for i, r in enumerate(df.itertuples())}

    jobs = [
        (r._asdict(), a.recipe, a.out_root, partners.get(r.video_id))
        for r in df.itertuples(index=False)
    ]
    workers = a.workers or min(os.cpu_count() or 4, 8)

    results: list[tuple[str, bool, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=a.recipe, unit="vid"):
            results.append(f.result())

    ok_paths = {vid: p for vid, ok, p in results if ok}
    failed = [vid for vid, ok, _ in results if not ok]
    log.info("%d succeeded, %d failed", len(ok_paths), len(failed))

    rows = []
    for r in df.itertuples(index=False):
        if r.video_id not in ok_paths:
            continue
        d = r._asdict()
        d["path"] = ok_paths[r.video_id]
        d["video_id"] = f"{r.video_id}__{a.recipe}"
        d["compression"] = TAGS.get(a.recipe, "unknown")
        # Degraded sets are evaluation-only. Marking them test prevents a
        # degraded copy from ever leaking into a training loader alongside its
        # own original -- which would be a textbook duplicate leak.
        d["split"] = "test"
        if a.recipe == "audio_strip":
            d["has_audio"] = False
        rows.append(d)

    if not rows:
        log.error("no videos were degraded successfully")
        return 1

    out = Path(a.out_manifest or f"data/manifests/{Path(a.manifest).stem}_{a.recipe}.parquet")
    write_manifest(rows, out)
    print(f"\nwrote {out} ({len(rows)} videos)")
    print(f"evaluate with:  ddetect eval --run <run> after preprocessing {out}")
    if failed:
        print(f"{len(failed)} failed (first 5): {failed[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
