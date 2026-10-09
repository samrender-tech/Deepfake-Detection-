"""A3 - experiment orchestration and fine-tuning.

Schedules the experiment grid under a hard GPU-hour budget, searches
hyperparameters on the SOURCE VALIDATION split, and babysits running jobs --
killing a run that has diverged rather than letting it burn hours producing
NaNs.

Two constraints make this safe to let loose:

* **It cannot see a target test set.** The firewall confines it to source
  train/val. It optimises against validation data only, which is
  the same discipline a human would be held to.
* **It cannot write model code.** Its tools compose configs and launch jobs.
  It has no editor, so it cannot quietly change an architecture to make a
  number move.

Every launch is logged with the hypothesis it is testing, stated BEFORE the
run. That makes post-hoc rationalisation visible in the audit log.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.core.loop import Agent, AgentResult, Finding

#: Log signatures that mean a run is dead and should be killed now rather than
#: at the end of its epoch budget.
DIVERGENCE = (
    (re.compile(r"non-?finite loss", re.I), "non-finite loss"),
    (re.compile(r"\bnan\b", re.I), "NaN in the logs"),
    (re.compile(r"CUDA out of memory", re.I), "CUDA OOM"),
    (re.compile(r"Traceback", re.I), "unhandled exception"),
)


class ExperimentAgent(Agent):
    name = "a3_experiment"

    def register_tools(self) -> None:
        self.registry.add("list_runs", _list_runs, "enumerate existing run directories", pure=False)
        self.registry.add(
            "read_val_metrics",
            _read_val_metrics,
            "read a run's val metrics (source split only)",
            reads_data=True,
            pure=False,
        )
        self.registry.add(
            "estimate_budget",
            _estimate_budget,
            "GPU-hour cost of the grid",
        )

    # ------------------------------------------------------------------
    def analyse(
        self, run_root: str | Path = "runs", budget_hours: float | None = None
    ) -> AgentResult:
        from experiments.run_matrix import MATRIX

        res = AgentResult(agent=self.name)
        findings: list[Finding] = []
        budget_hours = budget_hours or self.budget.max_gpu_hours

        done, running, failed = _scan_runs(Path(run_root))
        res.data["completed"] = sorted(done)
        res.data["incomplete"] = sorted(running)
        res.data["failed"] = failed

        # ---- 1. what still needs running -------------------------------
        todo = []
        for run in MATRIX:
            for seed in run.seeds:
                tag = f"{run.exp}/seed{seed}"
                if tag not in done:
                    todo.append(
                        {
                            "tag": tag,
                            "exp": run.exp,
                            "model": run.model,
                            "group": run.group,
                            "hours": run.hours_per_seed,
                            "note": run.note,
                        }
                    )
        remaining = sum(t["hours"] for t in todo)
        res.data["todo"] = todo
        res.data["remaining_gpu_hours"] = round(remaining, 1)
        res.data["budget_gpu_hours"] = budget_hours

        if remaining > budget_hours > 0:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"grid needs {remaining:.0f} GPU-h, budget is {budget_hours:.0f}",
                    detail=(
                        f"{len(todo)} runs remain. The documented mitigation is "
                        f"to keep three seeds only on baseline_b and full_system, drop the "
                        f"ablations to the 0.3x subset protocol, and state that in the "
                        f"table captions."
                    ),
                    evidence={
                        "remaining": round(remaining, 1),
                        "budget": budget_hours,
                        "runs": len(todo),
                    },
                    suggested_action="Run --group headline first; ablations can use one seed.",
                )
            )

        # ---- 2. order the queue ----------------------------------------
        # Headline before ablations, and cheap before expensive inside each
        # group, so the earliest hours bought produce the most reportable
        # result. baseline_b first of all: it IS the paper.
        priority = {"headline": 0, "ablation": 1}
        todo.sort(
            key=lambda t: (
                priority.get(t["group"], 2),
                0 if t["exp"] == "baseline_b" else 1,
                t["hours"],
            )
        )
        res.data["suggested_order"] = [t["tag"] for t in todo[:12]]

        # ---- 3. babysitting running jobs -------------------------------
        for tag in sorted(running):
            log_p = Path(run_root) / tag.replace("/", "/") / "metrics.jsonl"
            diag = _diagnose(Path(run_root) / tag)
            if diag:
                findings.append(
                    Finding(
                        severity="blocker",
                        title=f"{tag}: {diag['reason']}",
                        detail=diag["detail"],
                        evidence=diag,
                        suggested_action="Kill this run; it will not produce a usable result.",
                    )
                )
            elif log_p.exists():
                trend = _val_trend(log_p)
                if trend and trend["stalled_epochs"] >= 4:
                    findings.append(
                        Finding(
                            severity="warn",
                            title=f"{tag}: val AUC flat for {trend['stalled_epochs']} epochs",
                            detail=(
                                f"Best val AUC {trend['best']:.4f} at epoch "
                                f"{trend['best_epoch']}; no improvement since. Early "
                                f"stopping should fire, but the remaining epochs are "
                                f"budget that could go to another experiment."
                            ),
                            evidence=trend,
                            suggested_action="Let early stopping end it, and reallocate the hours.",
                        )
                    )
                if trend and trend.get("suspiciously_high"):
                    findings.append(
                        Finding(
                            severity="blocker",
                            title=f"{tag}: val AUC {trend['best']:.4f} on epoch {trend['best_epoch']}",
                            detail=(
                                "Near-perfect validation accuracy this early is far more "
                                "likely to be leakage than learning. The project's whole "
                                "claim depends on this number being honest."
                            ),
                            evidence=trend,
                            suggested_action="Stop and run `pytest tests/test_leakage.py` before trusting this.",
                        )
                    )

        # ---- 4. hyperparameter search space ----------------------------
        res.data["search_space"] = _SEARCH_SPACE
        findings.append(
            Finding(
                severity="info",
                title="hyperparameter search is scored on SOURCE VAL only",
                detail=(
                    "Optuna/ASHA over the space below. Target test sets are never "
                    "read during search; they are evaluated once per "
                    "frozen config through `make eval-final`."
                ),
                evidence=_SEARCH_SPACE,
            )
        )

        res.findings = findings
        res.ok = not any(f.severity == "blocker" for f in findings)
        res.summary = (
            f"{len(done)} run(s) complete, {len(todo)} remaining "
            f"(~{remaining:.0f} GPU-h of {budget_hours:.0f} budget); "
            f"{len(res.blockers)} blocker(s)."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        if not result.data.get("todo"):
            return None
        return (
            "You are planning the next batch of training runs for a deepfake "
            "detection project with a limited GPU budget.\n\n"
            "Given the state below, recommend the next 3 runs to launch and say "
            "for each what hypothesis it tests and what result would make you "
            "change course. Be specific about why that ordering buys the most "
            "reportable result per GPU-hour.\n\n"
            "Do not propose reading any test set.\n\n"
            f"```json\n{json.dumps(result.data, indent=2, default=str)[:6000]}\n```"
        )


#: Searched on source val only. Ranges are deliberately narrow: with ~1k
#: training videos a wide search overfits the validation split itself.
_SEARCH_SPACE = {
    "lr": {"type": "loguniform", "low": 3e-5, "high": 3e-4},
    "head_lr": {"type": "loguniform", "low": 3e-4, "high": 3e-3},
    "weight_decay": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "focal_gamma": {"type": "categorical", "choices": [0.0, 1.0, 2.0, 3.0]},
    "label_smoothing": {"type": "uniform", "low": 0.0, "high": 0.15},
    "lambda_mask": {"type": "uniform", "low": 0.0, "high": 0.5},
    "lambda_supcon": {"type": "uniform", "low": 0.0, "high": 0.3},
    "n_frames": {"type": "categorical", "choices": [8, 16, 32]},
    "modality_dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
    "sbi_p": {"type": "uniform", "low": 0.3, "high": 0.7},
}


def _scan_runs(root: Path) -> tuple[set[str], set[str], list[dict]]:
    done, running, failed = set(), set(), []
    if not root.exists():
        return done, running, failed
    for d in sorted(root.glob("*/seed*")):
        tag = f"{d.parent.name}/{d.name}"
        if (d / "preds_val.csv").exists() and (d / "summary.json").exists():
            done.add(tag)
        elif (d / "metrics.jsonl").exists():
            running.add(tag)
        else:
            failed.append({"tag": tag, "reason": "no metrics written"})
    return done, running, failed


def _list_runs(root: str = "runs") -> dict:
    done, running, failed = _scan_runs(Path(root))
    return {"done": sorted(done), "running": sorted(running), "failed": failed}


def _read_val_metrics(path: str, split: str = "val") -> dict:  # noqa: ARG001
    """Read a run's val metrics.

    ``split`` is unused here but required: the tool registry reads it to decide
    what the firewall should check before this runs.
    """
    p = Path(path)
    if not p.exists():
        return {}
    rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return {"epochs": len(rows), "last": rows[-1] if rows else {}}


def _estimate_budget(group: str | None = None) -> dict:
    from experiments.run_matrix import MATRIX

    runs = [r for r in MATRIX if group is None or r.group == group]
    return {
        "runs": len(runs),
        "seeds": sum(len(r.seeds) for r in runs),
        "gpu_hours": round(sum(r.total_hours for r in runs), 1),
    }


def _diagnose(run_dir: Path) -> dict | None:
    """Look for a hard failure signature in a run's logs."""
    for name in ("metrics.jsonl", "train.log", "stderr.log"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text()[-20000:]
        except OSError:
            continue
        for pat, reason in DIVERGENCE:
            if pat.search(text):
                return {
                    "run": str(run_dir),
                    "reason": reason,
                    "source": name,
                    "detail": f"{reason} found in {name}; the run cannot recover.",
                }
    return None


def _val_trend(metrics_jsonl: Path) -> dict | None:
    rows = [json.loads(ln) for ln in metrics_jsonl.read_text().splitlines() if ln.strip()]
    vals = [(r["epoch"], r["val_auc"]) for r in rows if "val_auc" in r]
    if not vals:
        return None
    best_epoch, best = max(vals, key=lambda kv: kv[1])
    return {
        "epochs": len(vals),
        "best": round(float(best), 4),
        "best_epoch": int(best_epoch),
        "stalled_epochs": int(vals[-1][0] - best_epoch),
        # A detector that reaches near-perfect val AUC in the first couple of
        # epochs is almost always reading a leak, not learning a forgery cue.
        "suspiciously_high": bool(best > 0.995 and best_epoch <= 1 and len(vals) > 1),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = ExperimentAgent()
    res = agent.run(run_root=a.runs, budget_hours=a.budget_hours)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "experiment-plan")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
