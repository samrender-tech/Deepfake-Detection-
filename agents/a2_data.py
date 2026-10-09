"""A2 - data curation.

Validates a dataset before anyone trains on it. The three things it checks are
the three that silently invalidate results:

1. **Audio presence.** Produces the evidence table behind the no-audio finding --
   that FF++ and Celeb-DF ship without usable audio, so the audio-visual
   objective cannot be evaluated on that pairing. This is run with ffprobe,
   not taken on reputation.
2. **Leakage.** Identity overlap and near-duplicate videos across splits. A
   leak shrinks the cross-dataset gap for reasons unrelated to generalisation
   while every training curve still looks healthy, so this is a blocker.
3. **Distribution skew.** Resolution, duration, face size and per-method
   balance -- the confounds that make a per-method breakdown uninterpretable.

All of it is deterministic Python. The model, if available, only writes the
dataset-card prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents.core.loop import Agent, AgentResult, Finding


class DataCurationAgent(Agent):
    name = "a2_data"

    def register_tools(self) -> None:
        from ddetect.data.integrity import audit
        from ddetect.data.manifest_io import read_manifest

        self.registry.add(
            "read_manifest",
            lambda path, split=None: read_manifest(path, split=split),
            "read a dataset manifest",
            reads_data=True,
            pure=False,
        )
        self.registry.add(
            "integrity_audit",
            lambda df, **kw: audit(df, **kw),
            "run the F5 leakage sweep",
            pure=False,
        )

    # ------------------------------------------------------------------
    def analyse(self, manifest: str | Path, phashes: dict | None = None) -> AgentResult:
        from ddetect.data.integrity import audit
        from ddetect.data.manifest_io import read_manifest

        df = read_manifest(manifest)
        res = AgentResult(agent=self.name)
        findings: list[Finding] = []

        # ---- 1. the audio evidence table --------------------
        audio = (
            df.groupby("dataset")
            .agg(n=("video_id", "count"), with_audio=("has_audio", "sum"))
            .assign(pct=lambda d: (d.with_audio / d.n * 100).round(1))
        )
        res.data["audio_table"] = audio.to_dict("index")

        for ds, row in audio.iterrows():
            if row.with_audio == 0:
                findings.append(
                    Finding(
                        severity="info",
                        title=f"{ds}: no usable audio in any of {int(row.n)} videos",
                        detail=(
                            f"ffprobe found no usable audio stream in {int(row.n)}/{int(row.n)} "
                            f"videos. The audio and lip-sync streams cannot be evaluated on "
                            f"this dataset at all, so it belongs in the visual-only protocol."
                        ),
                        evidence={"dataset": ds, "n": int(row.n), "with_audio": 0},
                        suggested_action=(
                            "Keep this dataset in protocol V. Evaluate Objective 3 on "
                            "DFDC -> FakeAVCeleb instead."
                        ),
                    )
                )
            elif row.pct < 100:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"{ds}: only {row.pct}% of videos carry audio",
                        detail=(
                            "A mixed manifest needs audio-stratified batching, or many "
                            "batches will be all-silent and the audio and sync branches "
                            "will get no gradient."
                        ),
                        evidence={"dataset": ds, "pct_with_audio": float(row.pct)},
                        suggested_action="Use sampler=audio_stratified for this manifest.",
                    )
                )

        # ---- 2. leakage (blocking) -------------------------------------
        rep = audit(df, check_paths=False, phashes=phashes)
        res.data["integrity"] = {
            "ok": rep.ok,
            "identity_overlaps": {k: v[:20] for k, v in rep.identity_overlaps.items()},
            "duplicate_pairs": rep.duplicate_pairs[:20],
            "cross_dataset_dupes": rep.cross_dataset_dupes[:20],
        }
        for pair, ids in rep.identity_overlaps.items():
            findings.append(
                Finding(
                    severity="blocker",
                    title=f"identity overlap across splits: {pair}",
                    detail=(
                        f"{len(ids)} source identities appear on both sides of this split "
                        f"boundary. The cross-dataset gap measured against this manifest "
                        f"would be shrunk by memorisation, not generalisation."
                    ),
                    evidence={"pair": pair, "identities": ids[:10], "count": len(ids)},
                    suggested_action="Fix the split before training. Do not report any number from this manifest.",
                )
            )
        if rep.duplicate_pairs:
            findings.append(
                Finding(
                    severity="blocker",
                    title=f"{len(rep.duplicate_pairs)} near-duplicate videos across splits",
                    detail="Perceptually near-identical videos appear in two splits of the same dataset.",
                    evidence={"pairs": rep.duplicate_pairs[:10]},
                    suggested_action="Drop one side of each pair, then re-run the sweep.",
                )
            )
        if rep.cross_dataset_dupes:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"{len(rep.cross_dataset_dupes)} near-duplicates ACROSS datasets",
                    detail=(
                        "These videos appear in both a training and a held-out dataset. "
                        "The train-on-one/test-on-other claim is compromised for them "
                        "specifically, though not for the dataset as a whole."
                    ),
                    evidence={"pairs": rep.cross_dataset_dupes[:10]},
                    suggested_action="Exclude them from the cross-dataset test set and say so in the paper.",
                )
            )

        # ---- 3. distribution skew --------------------------------------
        skew: dict[str, Any] = {}
        for ds, g in df.groupby("dataset"):
            real, fake = int((g.label == 0).sum()), int((g.label == 1).sum())
            ratio = fake / max(real, 1)
            methods = g[g.label == 1].forgery_method.value_counts().to_dict()
            skew[ds] = {
                "real": real,
                "fake": fake,
                "fake_per_real": round(ratio, 2),
                "methods": methods,
                "duration_s": {
                    "median": round(float(g.duration_s.median()), 1),
                    "p05": round(float(g.duration_s.quantile(0.05)), 1),
                    "p95": round(float(g.duration_s.quantile(0.95)), 1),
                },
                "fps_median": round(float(g.fps.median()), 1),
                "identities": int(g.source_identity.nunique()),
            }
            if ratio > 3:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"{ds}: {ratio:.1f} fakes per real video",
                        detail=(
                            "An unbalanced sampler would make the model method-specialised "
                            "-- over-fitting whichever generator is most numerous, which is "
                            "the exact failure mode the project exists to measure."
                        ),
                        evidence={"dataset": ds, "real": real, "fake": fake},
                        suggested_action="Use sampler=balanced (real/fake AND per-method).",
                    )
                )
            if len(methods) > 1:
                counts = np.array(list(methods.values()), dtype=float)
                if counts.max() / max(counts.min(), 1) > 2:
                    findings.append(
                        Finding(
                            severity="info",
                            title=f"{ds}: forgery methods are unevenly represented",
                            detail="Per-method breakdowns will have very different sample sizes; say so in captions.",
                            evidence={"dataset": ds, "methods": methods},
                        )
                    )
        res.data["skew"] = skew

        # ---- 4. split sanity -------------------------------------------
        for ds, g in df.groupby("dataset"):
            missing = {"train", "val", "test"} - set(g.split.unique())
            if missing and len(g) > 20:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"{ds}: missing split(s) {sorted(missing)}",
                        detail=(
                            "A missing val split means the decision threshold and the "
                            "calibration have nowhere legitimate to come from."
                        ),
                        evidence={"dataset": ds, "present": sorted(g.split.unique())},
                        suggested_action="Define a val split, or use this dataset as a test set only.",
                    )
                )

        res.findings = findings
        res.ok = not any(f.severity == "blocker" for f in findings)
        n_audio = int(df.has_audio.sum())
        res.summary = (
            f"{len(df)} videos across {df.dataset.nunique()} dataset(s); "
            f"{n_audio} ({100 * n_audio / max(len(df), 1):.0f}%) carry audio; "
            f"{len(res.blockers)} blocker(s), "
            f"{len([f for f in findings if f.severity == 'warn'])} warning(s)."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        import json

        if not result.findings:
            return None
        return (
            "You are writing the 'Known biases and caveats' section of a dataset "
            "card for a deepfake-detection project. Below is the output of an "
            "automated audit. Write 3-5 short paragraphs in plain prose.\n\n"
            "Rules: use only the numbers given; do not speculate about data you "
            "cannot see; do not reassure. If the audit found a blocker, say "
            "plainly that the dataset must not be trained on until it is fixed.\n\n"
            f"```json\n{json.dumps(result.data, indent=2, default=str)[:6000]}\n```"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--write", action="store_true", help="write a proposal file")
    a = ap.parse_args(argv)
    setup_logging()

    agent = DataCurationAgent()
    res = agent.run(manifest=a.manifest)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "dataset-audit")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
