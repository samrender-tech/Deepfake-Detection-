"""A4 - failure diagnosis.

Turns "accuracy is low" into a ranked list of testable fixes. It pulls the
worst-scoring SOURCE VALIDATION videos, clusters the failures by metadata, and
names the mechanism behind each cluster.

The firewall confines it to train/val of the source dataset. That
restriction is the point: an agent that looked at cross-dataset test failures
and then proposed a model change would be doing manual test-set optimisation
by proxy, and the headline number would stop meaning anything.

The clustering is deterministic. The model, when available, only ranks the
hypotheses and writes them up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.core.loop import Agent, AgentResult, Finding

#: Metadata axes to slice failures along. Each one maps to a concrete fix, which
#: is why they were chosen -- "fails on short clips" is actionable, "fails on
#: videos whose id starts with 7" is not.
AXES = ("forgery_method", "compression", "has_audio", "dataset")


class FailureDiagnosisAgent(Agent):
    name = "a4_failure"

    def register_tools(self) -> None:
        from agents.core.firewall import guarded_read_preds

        self.registry.add(
            "read_val_preds",
            lambda path: guarded_read_preds(self.name, path),
            "read source-validation predictions",
            reads_data=True,
            pure=False,
        )

    # ------------------------------------------------------------------
    def analyse(
        self, preds: str | Path, manifest: str | Path | None = None, top_k: int = 20
    ) -> AgentResult:
        from agents.core.firewall import guarded_read_preds

        # The firewall refuses anything that is not a source val file.
        df = guarded_read_preds(self.name, preds)
        res = AgentResult(agent=self.name)
        findings: list[Finding] = []

        if manifest:
            from ddetect.data.manifest_io import read_manifest

            meta = read_manifest(manifest)[
                ["video_id", "duration_s", "fps", "n_frames", "source_identity"]
            ]
            df = df.merge(meta, on="video_id", how="left")

        y = df.label.to_numpy()
        p = (df.calibrated_prob if "calibrated_prob" in df else df.video_score).to_numpy()

        if len(np.unique(y)) < 2:
            res.ok = False
            res.summary = "single-class predictions; nothing to diagnose"
            return res

        from ddetect.metrics import threshold_from_val

        thr = threshold_from_val(y, p)
        pred = (p >= thr).astype(int)
        df = df.assign(
            correct=(pred == y),
            # Distance past the threshold in the WRONG direction: a confident
            # mistake is far more informative than a borderline one.
            error_mag=np.where(pred == y, 0.0, np.abs(p - thr)),
        )
        overall = float(df.correct.mean())
        res.data["overall_accuracy"] = round(overall, 4)
        res.data["threshold"] = round(float(thr), 4)
        res.data["n"] = len(df)

        # ---- 1. per-axis breakdown -------------------------------------
        clusters: dict[str, Any] = {}
        for axis in AXES:
            if axis not in df.columns or df[axis].nunique() < 2:
                continue
            g = df.groupby(axis, observed=True).agg(n=("correct", "size"), acc=("correct", "mean"))
            g = g[g.n >= 5]
            if g.empty:
                continue
            clusters[axis] = {
                str(k): {"n": int(v.n), "acc": round(float(v.acc), 4)} for k, v in g.iterrows()
            }
            worst = g.acc.idxmin()
            gap = overall - float(g.acc.min())
            if gap > 0.10:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"{axis}={worst!r}: accuracy {g.acc.min():.3f} vs {overall:.3f} overall",
                        detail=(
                            f"Accuracy drops {gap:.3f} on this slice ({int(g.loc[worst].n)} "
                            f"videos). A gap this size on one axis usually means the model "
                            f"is relying on a cue that is absent or different there."
                        ),
                        evidence={
                            "axis": axis,
                            "value": str(worst),
                            "slice_acc": round(float(g.acc.min()), 4),
                            "overall_acc": round(overall, 4),
                            "n": int(g.loc[worst].n),
                        },
                        suggested_action=_action_for(axis, str(worst)),
                    )
                )
        res.data["clusters"] = clusters

        # ---- 2. continuous axes ----------------------------------------
        for col, label in (("duration_s", "clip duration"), ("fps", "frame rate")):
            if col not in df.columns or df[col].isna().all():
                continue
            try:
                q = pd.qcut(df[col], 4, duplicates="drop")
            except ValueError:
                continue
            g = df.groupby(q, observed=True).correct.agg(["size", "mean"])
            g = g[g["size"] >= 5]
            if len(g) < 2:
                continue
            res.data.setdefault("continuous", {})[col] = {
                str(k): {"n": int(v["size"]), "acc": round(float(v["mean"]), 4)}
                for k, v in g.iterrows()
            }
            if float(g["mean"].max() - g["mean"].min()) > 0.12:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"accuracy varies with {label}",
                        detail=(
                            f"Accuracy ranges {g['mean'].min():.3f}-{g['mean'].max():.3f} "
                            f"across quartiles of {col}. Worst quartile: {g['mean'].idxmin()}."
                        ),
                        evidence=res.data["continuous"][col],
                        suggested_action=(
                            f"Stratify training by {col}, or add augmentation that covers "
                            f"the weak range."
                        ),
                    )
                )

        # ---- 3. confident mistakes -------------------------------------
        wrong = df[~df.correct].nlargest(top_k, "error_mag")
        res.data["worst_videos"] = [
            {
                "video_id": r.video_id,
                "label": int(r.label),
                "prob": round(float(getattr(r, "calibrated_prob", r.video_score)), 4),
                "method": getattr(r, "forgery_method", "?"),
                "has_audio": bool(getattr(r, "has_audio", False)),
            }
            for r in wrong.itertuples()
        ]
        if len(wrong):
            fn = int(((df.label == 1) & (~df.correct)).sum())
            fp = int(((df.label == 0) & (~df.correct)).sum())
            findings.append(
                Finding(
                    severity="info",
                    title=f"{fn} missed fakes, {fp} false accusations",
                    detail=(
                        "A false negative launders a real forgery; a false positive "
                        "accuses a real person. The abstention band exists to convert "
                        "the worst of both into 'inconclusive' -- check the "
                        "risk-coverage curve before tuning the threshold."
                    ),
                    evidence={
                        "false_negatives": fn,
                        "false_positives": fp,
                        "worst": res.data["worst_videos"][:5],
                    },
                    suggested_action=(
                        "If false positives dominate, raise the threshold on SOURCE VAL "
                        "and report the new operating point -- never tune it on a test set."
                    ),
                )
            )

        # ---- 4. is the confidence signal real? -------------------------
        if "abstain" in df.columns and df.abstain.any():
            kept = df[~df.abstain]
            if len(kept) > 10:
                lift = float(kept.correct.mean()) - overall
                sev = "info" if lift > 0.01 else "warn"
                findings.append(
                    Finding(
                        severity=sev,
                        title=f"abstention changes accuracy by {lift:+.3f}",
                        detail=(
                            "Abstaining on the least confident cases should raise accuracy "
                            "on the rest. A lift near zero means the model's confidence "
                            "carries no information -- which is itself a publishable finding."
                        ),
                        evidence={
                            "all": round(overall, 4),
                            "non_abstained": round(float(kept.correct.mean()), 4),
                            "abstain_rate": round(float(df.abstain.mean()), 4),
                        },
                        suggested_action=(
                            "If the lift is ~0, report it honestly rather than widening "
                            "the band until the number looks good."
                        ),
                    )
                )

        findings.sort(key=lambda f: {"blocker": 0, "warn": 1, "info": 2}[f.severity])
        res.findings = findings
        res.summary = (
            f"val accuracy {overall:.3f} on {len(df)} videos at threshold {thr:.3f}; "
            f"{len(findings)} finding(s) across {len(clusters)} metadata axes."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        if not result.findings:
            return None
        return (
            "You are a machine-learning engineer reviewing a deepfake detector's "
            "validation failures. Below is an automated breakdown.\n\n"
            "Propose the THREE highest-value changes, ranked. For each: the "
            "mechanism you think is responsible, and the single experiment that "
            "would confirm or refute it. Be concrete (a config change, an "
            "augmentation, a sampler) -- not 'collect more data'.\n\n"
            "Use only the numbers given. If the evidence does not support three "
            "distinct hypotheses, give fewer and say why.\n\n"
            f"```json\n{json.dumps(result.data, indent=2, default=str)[:7000]}\n```"
        )


def _action_for(axis: str, value: str) -> str:
    """Map a weak slice to a concrete, testable change."""
    if axis == "compression":
        return (
            "Increase degradation augmentation strength (configs/aug/degrade.yaml: "
            "raise jpeg_p and widen downscale_range) and re-check this slice."
        )
    if axis == "forgery_method":
        return (
            f"Check the method-balanced sampler is actually running; if {value} is "
            "under-represented the model has specialised away from it. Also compare "
            "with SBI enabled -- blending-boundary supervision is method-agnostic."
        )
    if axis == "has_audio":
        return (
            "If the weak slice is audio-absent, modality dropout may be too low "
            "(the model is leaning on audio). If it is audio-present, the fusion "
            "gate may be over-trusting a noisy audio stream."
        )
    return f"Investigate the {axis}={value} slice; stratify training across this axis."


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="source VAL predictions CSV")
    ap.add_argument("--manifest", default=None, help="optional, for duration/fps axes")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = FailureDiagnosisAgent()
    res = agent.run(preds=a.preds, manifest=a.manifest)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "failure-diagnosis")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
