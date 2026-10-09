"""F18 - generate every figure the paper uses.

Figures are generated, never drawn. A hand-made figure goes stale the moment
anything is re-run, and the reproducibility claim in the paper depends on
every number tracing to a committed run.

Five figures, each answering a question a reader will actually ask:

  roc              how well does it separate, in-dataset vs cross-dataset?
  reliability      can the probability be believed? (and how does that degrade
                   off-domain, which is a finding in itself)
  risk_coverage    does abstaining actually help, or is confidence noise?
  per_method       which forgery methods does it fail on?
  gap              the headline: how far does accuracy fall off-domain?

Styling is deliberately plain: IEEE two-column, greyscale-safe, no chartjunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE single column is 3.5in; double is 7.16in.
COL, DCOL = 3.5, 7.16
#: Colour-blind safe and distinguishable in greyscale. Printed proceedings are
#: still a thing, and a reviewer may be reading on paper.
PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#949494", "#ECE133"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )


# ==========================================================================
def roc_figure(preds: dict[str, Path], out: Path) -> Path:
    """ROC per test set. The in-dataset/cross-dataset separation is the story."""
    from sklearn.metrics import roc_curve

    from ddetect.metrics import delong_ci, load_preds

    _style()
    fig, ax = plt.subplots(figsize=(COL, COL * 0.85))
    for i, (name, p) in enumerate(sorted(preds.items())):
        df = load_preds(p)
        y, s = df.label.to_numpy(), df.video_score.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc, lo, hi = delong_ci(y, s)
        ax.plot(
            fpr,
            tpr,
            color=PALETTE[i % len(PALETTE)],
            label=f"{name} (AUC {auc:.3f} [{lo:.3f},{hi:.3f}])",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return out


def reliability_figure(preds: dict[str, Path], out: Path, n_bins: int = 10) -> Path:
    """Reliability diagram. Off-domain miscalibration is itself a result."""
    from ddetect.metrics import expected_calibration_error, load_preds, reliability_curve

    _style()
    fig, ax = plt.subplots(figsize=(COL, COL * 0.85))
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5, label="perfect")
    for i, (name, p) in enumerate(sorted(preds.items())):
        df = load_preds(p)
        y = df.label.to_numpy()
        prob = (df.calibrated_prob if "calibrated_prob" in df else df.video_score).to_numpy()
        c, obs, cnt = reliability_curve(y, prob, n_bins)
        m = cnt > 0
        ece = expected_calibration_error(y, prob)
        ax.plot(
            c[m],
            obs[m],
            "o-",
            ms=3,
            color=PALETTE[i % len(PALETTE)],
            label=f"{name} (ECE {ece:.3f})",
        )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    return out


def risk_coverage_figure(preds: dict[str, Path], out: Path) -> Path:
    """Does abstention buy anything? A flat curve means confidence is noise."""
    from ddetect.metrics import load_preds, risk_coverage_curve, threshold_from_val

    _style()
    fig, ax = plt.subplots(figsize=(COL, COL * 0.85))
    for i, (name, p) in enumerate(sorted(preds.items())):
        df = load_preds(p)
        y = df.label.to_numpy()
        prob = (df.calibrated_prob if "calibrated_prob" in df else df.video_score).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        thr = threshold_from_val(y, prob)
        cov, risk = risk_coverage_curve(y, prob, thr)
        ax.plot(cov, risk, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("coverage (fraction of videos answered)")
    ax.set_ylabel("selective risk (error rate on those)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    return out


def per_method_figure(breakdowns_csv: Path, out: Path) -> Path | None:
    """Per-forgery-method AUC. Shows which generators the model cannot see."""
    import pandas as pd

    if not breakdowns_csv.exists():
        return None
    df = pd.read_csv(breakdowns_csv)
    rows = df[df.name.astype(str).str.contains("forgery_method=", na=False)].copy()
    if rows.empty:
        return None
    rows["method"] = rows.name.str.split("forgery_method=").str[-1]
    rows["split"] = rows.name.str.split("/").str[0]

    _style()
    fig, ax = plt.subplots(figsize=(DCOL * 0.6, COL * 0.8))
    splits = sorted(rows.split.unique())
    methods = sorted(rows.method.unique())
    width = 0.8 / max(len(splits), 1)
    x = np.arange(len(methods))
    for i, sp in enumerate(splits):
        sub = rows[rows.split == sp].set_index("method").reindex(methods)
        ax.bar(x + i * width, sub.auc.fillna(0), width, label=sp, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0.5, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out


def gap_figure(gap_csv: Path, out: Path) -> Path | None:
    """The headline: in-dataset vs cross-dataset AUC per experiment."""
    import pandas as pd

    if not gap_csv.exists():
        return None
    df = pd.read_csv(gap_csv)
    if df.empty:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(DCOL * 0.6, COL * 0.8))
    labels = [f"{r.exp}→{r.test_set}" for r in df.itertuples()]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.in_dataset_auc, 0.4, label="in-dataset", color=PALETTE[0])
    ax.bar(x + 0.2, df.cross_dataset_auc, 0.4, label="cross-dataset", color=PALETTE[1])
    for i, r in enumerate(df.itertuples()):
        ax.annotate(
            f"-{r.gap:.3f}",
            (i, max(r.in_dataset_auc, r.cross_dataset_auc) + 0.02),
            ha="center",
            fontsize=6.5,
        )
    ax.axhline(0.5, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return out


# ==========================================================================
def discover_preds(run_root: Path) -> dict[str, Path]:
    """Collect preds CSVs, keyed ``exp/split``."""
    out: dict[str, Path] = {}
    for p in sorted(run_root.glob("*/seed*/preds_*.csv")):
        exp = p.parent.parent.name
        split = p.stem.replace("preds_", "")
        out.setdefault(f"{exp}/{split}", p)  # first seed is representative
    return out


def generate_all(
    run_root: str | Path = "runs",
    results: str | Path = "results",
    out_dir: str | Path = "paper/figs",
) -> dict[str, Path]:
    run_root, results, out = Path(run_root), Path(results), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = discover_preds(run_root)
    made: dict[str, Path] = {}
    if preds:
        made["roc"] = roc_figure(preds, out / "roc.pdf")
        made["reliability"] = reliability_figure(preds, out / "reliability.pdf")
        made["risk_coverage"] = risk_coverage_figure(preds, out / "risk_coverage.pdf")
    for name, fn, src in (
        ("per_method", per_method_figure, results / "breakdowns.csv"),
        ("gap", gap_figure, results / "cross_dataset_gap.csv"),
    ):
        p = fn(src, out / f"{name}.pdf")  # type: ignore[operator]
        if p:
            made[name] = p
    (out / "manifest.json").write_text(json.dumps({k: str(v) for k, v in made.items()}, indent=2))
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figs")
    a = ap.parse_args(argv)

    made = generate_all(a.runs, a.results, a.out)
    if not made:
        print("no figures generated -- train something, then run ddetect.evaluate")
        return 1
    for name, p in made.items():
        print(f"  {name:14s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
