"""F18 - expand the experiment grid into concrete runs.

    python -m experiments.run_matrix --dry-run          # show the plan
    python -m experiments.run_matrix --group headline   # actually run it

Two things it enforces:

* **A compute budget.** Section 8.4 estimates 250-350 GPU-hours for the full
  grid against roughly 90 free Kaggle hours a week. A grid that silently
  exceeds the budget is a grid that does not finish, so the estimate is checked
  before anything launches and the shortfall is reported in hours.
* **Seed discipline.** Headline rows get three seeds; ablations get one, and
  that is recorded so the table caption can say so.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)


@dataclass
class Run:
    exp: str
    model: str
    data: str
    aug: str
    seeds: tuple[int, ...]
    group: str
    note: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    #: Rough GPU-hours per seed, from the section 8.4 estimate
    #: (~25-35 min/epoch for EffNet-B4 at batch 32 on a T4, 12 epochs).
    hours_per_seed: float = 6.0

    @property
    def total_hours(self) -> float:
        return self.hours_per_seed * len(self.seeds)


HEADLINE_SEEDS = (0, 1, 2)
ABLATION_SEEDS = (0,)

MATRIX: list[Run] = [
    # ---- headline --------------------------------------
    Run(
        "baseline_a",
        "visual_xception",
        "ffpp",
        "none",
        HEADLINE_SEEDS,
        "headline",
        "E1: reproduce published in-dataset FF++ accuracy",
        hours_per_seed=4.0,
    ),
    Run(
        "baseline_b",
        "visual_xception",
        "ffpp",
        "none",
        HEADLINE_SEEDS,
        "headline",
        "E2: THE HEADLINE -- the cross-dataset drop",
        hours_per_seed=4.0,
    ),
    Run(
        "v_aug",
        "visual_effb4",
        "ffpp",
        "degrade",
        HEADLINE_SEEDS,
        "headline",
        "E3: + degradation augmentation (Objective 4)",
    ),
    Run(
        "v_sbi",
        "visual_effb4",
        "ffpp",
        "sbi",
        HEADLINE_SEEDS,
        "headline",
        "E4: + Self-Blended Images (Objective 2)",
    ),
    Run(
        "v_full",
        "visual_effb4",
        "ffpp",
        "degrade_sbi",
        HEADLINE_SEEDS,
        "headline",
        "E5: the full visual stack",
    ),
    Run(
        "av_proposed",
        "avforge",
        "dfdc",
        "degrade_sbi",
        HEADLINE_SEEDS,
        "headline",
        "E6: audio-visual (Objective 3) -- DFDC -> FakeAVCeleb per the no-audio finding",
        hours_per_seed=8.0,
    ),
    Run(
        "full_system",
        "avforge",
        "combined",
        "degrade_sbi",
        HEADLINE_SEEDS,
        "headline",
        "E7: one checkpoint, all four test sets",
        hours_per_seed=10.0,
    ),
    # ---- ablations -------------------------------------
    Run("abl_no_degrade", "avforge", "combined", "sbi", ABLATION_SEEDS, "ablation", "-degradation"),
    Run("abl_no_sbi", "avforge", "combined", "degrade", ABLATION_SEEDS, "ablation", "-SBI"),
    Run(
        "abl_no_freq",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-frequency branch",
        extra={"model.visual.use_freq": "false"},
    ),
    Run(
        "abl_no_blend",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-blending-boundary head",
        extra={"model.visual.use_blend_head": "false"},
    ),
    Run(
        "abl_mean_pool",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-temporal transformer (mean pool)",
        extra={"model.visual.temporal": "mean"},
    ),
    Run(
        "abl_no_audio",
        "visual_effb4",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-audio stream",
    ),
    Run(
        "abl_no_gate",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-reliability gate",
        extra={"model.fusion.use_gate": "false"},
    ),
    Run(
        "abl_concat_fusion",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "co-attention -> concat MLP",
        extra={"model.fusion.mode": "concat"},
    ),
    Run(
        "abl_score_fusion",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "co-attention -> score-level LR",
        extra={"model.fusion.mode": "score"},
    ),
    Run(
        "abl_no_modality_dropout",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "PREDICTED FAILURE: must collapse on silent video (the modality-dropout ablation)",
        extra={"model.fusion.modality_dropout": "0.0"},
    ),
    Run(
        "abl_no_swad",
        "avforge",
        "combined",
        "degrade_sbi",
        ABLATION_SEEDS,
        "ablation",
        "-SWAD",
        extra={"train": "finetune"},
    ),
]


def build_command(run: Run, seed: int, dry: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "ddetect.train",
        "--exp",
        run.exp,
        "--model",
        run.model,
        "--seed",
        str(seed),
    ]
    for k, v in run.extra.items():
        cmd += [f"--{k.replace('.', '-').replace('_', '-')}", v]
    if dry:
        cmd.append("--help")
    return cmd


def plan_table(runs: list[Run]) -> str:
    lines = [
        f"{'experiment':28s} {'model':16s} {'data':10s} {'aug':14s} {'seeds':6s} {'GPU-h':>6s}  note",
        "-" * 128,
    ]
    for r in runs:
        lines.append(
            f"{r.exp:28s} {r.model:16s} {r.data:10s} {r.aug:14s} "
            f"{len(r.seeds):<6d} {r.total_hours:6.1f}  {r.note}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default=None, choices=["headline", "ablation"])
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiment names")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument(
        "--budget-hours",
        type=float,
        default=350.0,
        help="GPU-hour budget (estimated 250-350)",
    )
    ap.add_argument("--out", default="results/matrix_plan.json")
    a = ap.parse_args(argv)
    setup_logging()

    runs = MATRIX
    if a.group:
        runs = [r for r in runs if r.group == a.group]
    if a.only:
        runs = [r for r in runs if r.exp in set(a.only)]
    if not runs:
        print("no runs match that filter")
        return 1

    total = sum(r.total_hours for r in runs)
    print(plan_table(runs))
    print(
        f"\n{len(runs)} experiments, {sum(len(r.seeds) for r in runs)} runs, "
        f"~{total:.0f} GPU-hours estimated."
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(
            [
                {
                    "exp": r.exp,
                    "model": r.model,
                    "data": r.data,
                    "aug": r.aug,
                    "seeds": list(r.seeds),
                    "group": r.group,
                    "note": r.note,
                    "hours": r.total_hours,
                }
                for r in runs
            ],
            indent=2,
        )
    )

    if total > a.budget_hours:
        print(
            f"\n  OVER BUDGET by {total - a.budget_hours:.0f} GPU-hours.\n"
            f"  The compute budget's mitigation: keep 3 seeds only on baseline_b and\n"
            f"  full_system, drop ablations to the documented 0.3x subset\n"
            f"  protocol, and say so in the table captions."
        )

    if a.dry_run:
        print("\n(dry run -- nothing launched)")
        return 0

    print(
        "\nSequential launch. Each run writes runs/<exp>/seed<N>/ and is resumable;\n"
        "interrupt safely with Ctrl-C.\n"
    )
    failures = []
    for r in runs:
        for seed in r.seeds:
            cmd = build_command(r, seed, dry=False)
            log.info("launching %s seed=%d", r.exp, seed)
            rc = subprocess.call(cmd)
            if rc != 0:
                failures.append(f"{r.exp}/seed{seed} (exit {rc})")
                log.error("%s seed=%d FAILED with exit %d", r.exp, seed, rc)

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nall runs completed. Aggregate with: python -m experiments.aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
