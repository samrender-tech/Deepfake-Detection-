"""Unified command-line entry point.

    ddetect manifest --dataset ffpp --root ~/data/FaceForensics++
    ddetect preprocess --manifest data/manifests/ffpp.parquet
    ddetect train --exp baseline_a --model baseline
    ddetect eval --run runs/baseline_a/seed0
    ddetect predict video.mp4 --run runs/baseline_a/seed0 --report report.html
    ddetect batch clips/ --run runs/baseline_a/seed0 --out verdicts.csv
    ddetect audit --manifest data/manifests/ffpp.parquet

Each subcommand delegates to the module that owns it, so there is exactly one
implementation of every step and `python -m ddetect.train` and
`ddetect train` cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SUBCOMMANDS = {
    "manifest": ("ddetect.data.build_manifest", "build a dataset manifest"),
    "preprocess": ("ddetect.data.run_preprocess", "populate the preprocessing cache"),
    "train": ("ddetect.train", "train a model"),
    "eval": ("ddetect.evaluate", "evaluate a run's predictions"),
    "matrix": ("experiments.run_matrix", "show or run the experiment grid"),
    "aggregate": ("experiments.aggregate", "build results tables and LaTeX"),
}


def _predict(argv: list[str]) -> int:
    """Score one video and print the verdict."""
    import json

    from ddetect.inference import Detector
    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(prog="ddetect predict")
    ap.add_argument("video")
    ap.add_argument("--run", required=True, help="run directory with a checkpoint")
    ap.add_argument("--json", action="store_true", help="emit the full Result as JSON")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--report", metavar="HTML", help="also write a self-contained HTML report")
    ap.add_argument(
        "--stability",
        action="store_true",
        help="re-encode the clip at lower quality and check the verdict holds (slower)",
    )
    a = ap.parse_args(argv)
    setup_logging()

    det = Detector.from_run(a.run)
    res = det.predict(a.video, explain=not a.no_explain)
    stab = None
    if a.stability:
        from ddetect.stability import stability_check

        stab = stability_check(det, a.video, original=res)
    if a.report:
        _write_report(res, a.video, a.report, stab.to_dict() if stab else None)

    if a.json:
        d = res.to_dict()
        if stab is not None:
            d["stability"] = stab.to_dict()
        print(json.dumps(d, indent=2, default=float))
        return 0

    print(f"\n  {res.verdict.value.replace('_', ' ').upper()}")
    print(f"  probability of manipulation: {res.calibrated_prob:.1%}")
    print(
        f"  inconclusive band: {res.conformal_lo:.1%}-{res.conformal_hi:.1%}"
        f"   threshold: {res.threshold:.1%}"
    )
    if res.ood_flag:
        print("  [!] unlike the model's training distribution")
    print(f"  audio: {res.has_audio}   faces found: {res.n_faces_found}")
    for s in res.streams:
        score = f"{s.score:.3f}" if s.score is not None else "not run"
        print(f"    {s.name:7s} {score:>8s}  {s.note}")
    from ddetect.report import segments_for

    for seg in segments_for(res.to_dict()):
        print(
            f"  flagged {seg['start_s']:.2f}s-{seg['end_s']:.2f}s"
            f"  ({seg['n_frames']} frames, peak {seg['peak_score']:.3f})"
        )
    for w in res.warnings:
        print(f"  warning: {w}")
    if res.explanation:
        print(f"\n  {res.explanation}")
    if stab is not None:
        print(f"\n  {stab.summary()}")
        for v in stab.variants:
            detail = v.error or f"{v.verdict} ({v.calibrated_prob:.1%})"
            print(f"    {v.name:9s} {detail}")
    if a.report:
        print(f"\n  report written to {a.report}")
    return 0


def _write_report(
    res: Any, video: str | Path, out: str | Path, stability: dict[str, Any] | None = None
) -> Path:
    """Render ``res`` to a self-contained HTML file, with its heat maps inlined."""
    import base64

    from ddetect.report import render_html

    images = {k: base64.b64decode(v) for k, v in res._gradcam_png.items()}
    if res._spectrum_png:
        images["spectrum"] = base64.b64decode(res._spectrum_png)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        render_html(res.to_dict(), filename=Path(video).name, images=images, stability=stability)
    )
    return p


VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv")


def _collect_videos(paths: list[str], recursive: bool) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated video list."""
    found: set[Path] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            found.update(q for q in it if q.is_file() and q.suffix.lower() in VIDEO_SUFFIXES)
        elif p.is_file():
            found.add(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    return sorted(found)


def _batch(argv: list[str]) -> int:
    """Score many videos with one loaded model and write a summary table.

    A failure on one clip is recorded in its row and does not stop the batch:
    an unreadable file in a folder of 500 should cost one row, not the run.
    The exit status is non-zero when any clip failed, so a script notices.
    """
    import csv
    import json
    from collections import Counter

    from ddetect.inference import Detector
    from ddetect.report import render_batch_index, summary_row
    from ddetect.stability import stability_check
    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(prog="ddetect batch")
    ap.add_argument("paths", nargs="+", help="video files and/or directories")
    ap.add_argument("--run", required=True, help="run directory with a checkpoint")
    ap.add_argument("--out", required=True, help="summary file: .csv or .jsonl")
    ap.add_argument("--recursive", "-r", action="store_true", help="descend into directories")
    ap.add_argument("--report-dir", help="also write one HTML report per clip here")
    ap.add_argument("--explain", action="store_true", help="compute Grad-CAM (slower)")
    ap.add_argument(
        "--stability", action="store_true", help="also run the re-encode stability check"
    )
    a = ap.parse_args(argv)
    setup_logging()

    videos = _collect_videos(a.paths, a.recursive)
    if not videos:
        print("no video files found", file=sys.stderr)
        return 2
    out = Path(a.out)
    if out.suffix.lower() not in (".csv", ".jsonl"):
        print("--out must end in .csv or .jsonl", file=sys.stderr)
        return 2

    det = Detector.from_run(a.run)
    explain = a.explain or bool(a.report_dir)
    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for i, v in enumerate(videos, 1):
        row: dict[str, Any] = {"file": str(v)}
        try:
            res = det.predict(v, explain=explain)
            row.update(summary_row(res.to_dict()))
            stab = None
            if a.stability:
                stab = stability_check(det, v, original=res).to_dict()
                row["stable"] = stab["stable"]
                row["verdict_agreement"] = stab["verdict_agreement"]
                row["prob_spread"] = stab["prob_spread"]
            row["error"] = ""
            if a.report_dir:
                # Two clips can share a stem in different folders (-r); never
                # let the second report silently overwrite the first.
                name, k = f"{v.stem}.html", 1
                while name in used_names:
                    k += 1
                    name = f"{v.stem}-{k}.html"
                used_names.add(name)
                _write_report(res, v, Path(a.report_dir) / name, stab)
                row["report"] = name
        except Exception as e:  # recorded per row, see docstring
            row["verdict"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(f"  [{i}/{len(videos)}] {row['verdict']:20s} {v.name}", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".jsonl":
        out.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    else:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    if a.report_dir:
        index = Path(a.report_dir) / "index.html"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(render_batch_index(rows))
        print(f"\n  summary page: {index}")

    counts = Counter(r["verdict"] for r in rows)
    print(f"\n  {len(rows)} clips -> {out}")
    for verdict, n in sorted(counts.items()):
        print(f"    {verdict:20s} {n}")
    return 1 if counts.get("error") else 0


def _audit(argv: list[str]) -> int:
    """Run the F5 integrity sweep over a manifest."""
    from ddetect.data.integrity import audit
    from ddetect.data.manifest_io import read_manifest, summarise
    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(prog="ddetect audit")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--no-paths", action="store_true", help="skip the file-existence check")
    a = ap.parse_args(argv)
    setup_logging()

    df = read_manifest(a.manifest)
    print(summarise(df))
    rep = audit(df, check_paths=not a.no_paths)
    print("\n" + rep.render())
    if not rep.ok:
        print(
            "\n  The cross-dataset gap is this project's headline result. A leak "
            "shrinks that gap for reasons unrelated to generalisation, so this "
            "is a blocking failure, not a warning."
        )
    return 0 if rep.ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        for name, (_, help_) in SUBCOMMANDS.items():
            print(f"  {name:12s} {help_}")
        print(f"  {'predict':12s} score one video and print the verdict")
        print(f"  {'batch':12s} score many videos and write a CSV/JSONL summary")
        print(f"  {'audit':12s} run the leakage and integrity sweep")
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "predict":
        return _predict(rest)
    if cmd == "batch":
        return _batch(rest)
    if cmd == "audit":
        return _audit(rest)

    if cmd not in SUBCOMMANDS:
        print(f"unknown command {cmd!r}. Run `ddetect --help`.", file=sys.stderr)
        return 2

    import importlib

    module = importlib.import_module(SUBCOMMANDS[cmd][0])
    return int(module.main(rest))  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
