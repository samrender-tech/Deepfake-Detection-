"""Evaluation CLI: predictions -> metrics, cross-dataset matrix, tables.

    # evaluate one test set against a threshold derived on source val
    python -m ddetect.evaluate --run runs/baseline_b/seed0 --test celebdf

    # the final, sanctioned target-test evaluation
    make eval-final RUN=runs/baseline_b/seed0

Reads only ``preds_*.csv`` (contract 5.4), so it works on any run directory --
including ones produced by someone else, or hand-written during development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ddetect.metrics import (
    MetricReport,
    evaluate_predictions,
    load_preds,
    reports_to_frame,
    save_reports,
    threshold_from_val,
)
from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)


def evaluate_run(
    run_dir: str | Path,
    test_splits: list[str] | None = None,
    group_by: list[str] | None = None,
    n_boot: int = 0,
) -> dict[str, MetricReport]:
    """Evaluate every ``preds_*.csv`` in a run against the source-val threshold.

    The threshold ALWAYS comes from ``preds_val.csv`` in the same run. There is
    no flag to override it from a test set, because that is the single easiest
    way to inflate a cross-dataset number by several points.
    """
    run_dir = Path(run_dir)
    val_p = run_dir / "preds_val.csv"
    if not val_p.exists():
        raise FileNotFoundError(
            f"{val_p} not found. The decision threshold must be derived on the "
            "source validation split; without it this run cannot be evaluated."
        )

    val = load_preds(val_p)
    # Calibrated probabilities when available, raw scores otherwise -- the
    # threshold and the band must live in the same space (see calibrate.py).
    col = "calibrated_prob" if val.calibrated_prob.notna().all() else "video_score"
    thr = threshold_from_val(val.label.to_numpy(), val[col].to_numpy())
    src = f"EER on source val ({len(val)} videos, column={col})"
    log.info("threshold %.4f from %s", thr, src)

    files = sorted(run_dir.glob("preds_*.csv"))
    if test_splits:
        want = {f"preds_{s}.csv" for s in test_splits} | {"preds_val.csv"}
        files = [f for f in files if f.name in want]

    reports: dict[str, MetricReport] = {}
    for f in files:
        split_name = f.stem.replace("preds_", "")
        df = load_preds(f)
        reports[split_name] = evaluate_predictions(
            df.label.to_numpy(),
            df.video_score.to_numpy(),
            threshold=thr,
            threshold_source=src,
            name=split_name,
            calibrated=df[col].to_numpy(),
            abstain=df.abstain.to_numpy() if "abstain" in df else None,
            n_boot=n_boot,
        )
        for key in group_by or []:
            if key not in df.columns:
                continue
            for val_, g in df.groupby(key, observed=True):
                if len(g) < 10:
                    continue  # a breakdown on <10 videos is noise, not a result
                reports[f"{split_name}/{key}={val_}"] = evaluate_predictions(
                    g.label.to_numpy(),
                    g.video_score.to_numpy(),
                    threshold=thr,
                    threshold_source=src,
                    name=f"{split_name}/{key}={val_}",
                    calibrated=g[col].to_numpy(),
                    abstain=g.abstain.to_numpy() if "abstain" in g else None,
                )
    return reports


def cross_dataset_gap(reports: dict[str, MetricReport], source: str = "val") -> dict[str, float]:
    """The headline number: in-dataset AUC minus each cross-dataset AUC.

    This is what the whole project exists to measure (typically >95% within
    a dataset, ~65% across). Reported as a dict so the aggregator can put it in
    its own table rather than burying it among twenty other metrics.
    """
    base = reports.get(source)
    if base is None or not np.isfinite(base.auc):
        return {}
    return {
        name: round(base.auc - r.auc, 4)
        for name, r in reports.items()
        if name != source and "/" not in name and np.isfinite(r.auc)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--test", nargs="*", default=None, help="limit to these splits")
    ap.add_argument(
        "--group-by",
        nargs="*",
        default=["forgery_method", "compression", "has_audio"],
        help="breakdown columns",
    )
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap iterations for CIs")
    ap.add_argument("--out", default=None, help="write metrics json here")
    a = ap.parse_args(argv)
    setup_logging()

    reports = evaluate_run(a.run, a.test, a.group_by, a.bootstrap)

    print("\n" + "=" * 118)
    for name, r in reports.items():
        if "/" not in name:
            print(r.summary())
    print("=" * 118)

    breakdowns = {k: v for k, v in reports.items() if "/" in k}
    if breakdowns:
        print("\nbreakdowns:")
        for r in breakdowns.values():
            print("  " + r.summary())

    gap = cross_dataset_gap(reports)
    if gap:
        print("\nCROSS-DATASET GAP (in-dataset AUC minus cross-dataset AUC):")
        for name, g in sorted(gap.items(), key=lambda kv: -kv[1]):
            print(f"  {name:28s} {g:+.4f}")
        print(
            "\n  This gap IS the result. A small gap is "
            "either a genuine generalisation win or a leak -- run "
            "`pytest tests/test_leakage.py` before believing it."
        )

    out = Path(a.out or Path(a.run) / "metrics_report.json")
    save_reports(reports, out)
    reports_to_frame(reports).to_csv(out.with_suffix(".csv"), index=False)
    if gap:
        out.with_name("cross_dataset_gap.json").write_text(json.dumps(gap, indent=2))
    log.info("wrote %s and %s", out, out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
