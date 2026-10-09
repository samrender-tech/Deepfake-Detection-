"""A6 - defence preparation.

Builds the Q&A bank for a project review, grounded in results that actually
exist.

The rule that makes it useful rather than reassuring: **if no result supports
an answer, A6 does not invent one -- it reports the question as a gap to close
before the review.** A defence sheet full of confident answers to questions the
work cannot yet answer is worse than no sheet, because it is discovered live.

The question bank starts with four questions any reviewer of this work will
ask, and adds the hostile ones reviewers actually ask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agents.core.loop import Agent, AgentResult, Finding


@dataclass
class Question:
    q: str
    category: str
    #: Results that must exist for an honest answer. Empty = answerable from
    #: the design alone.
    needs: list[str] = field(default_factory=list)
    answer_sketch: str = ""
    hostile: bool = False


#: The first four are the questions any reviewer will ask. The rest are what
#: sceptical reviewers actually ask about work like this.
BANK: list[Question] = [
    Question(
        "Why not train on all the datasets together?",
        "method",
        answer_sketch=(
            "Because then nothing is unseen. The whole measurement is how far "
            "accuracy falls on forgery methods the model has never met; pooling "
            "the datasets destroys the only thing being measured. We do report a "
            "combined-training system (E7), but it is evaluated against held-out "
            "datasets, not against its own training distribution."
        ),
    ),
    Question(
        "What if the audio-visual fusion does not help?",
        "method",
        needs=["av_proposed"],
        answer_sketch=(
            "That is still a reportable result, and the ablation says which part "
            "was responsible. The streams are trained separately first, so the "
            "visual result stands on its own regardless."
        ),
    ),
    Question(
        "Isn't deepfake detection already solved?",
        "framing",
        answer_sketch=(
            "Within a dataset, largely. Across datasets, no -- and that gap is "
            "what the project measures. A detector that scores 95% on its own "
            "test set and 65% on a new forgery method is not solved, it is "
            "overfitted to a generator."
        ),
    ),
    Question(
        "What compute does this need?",
        "feasibility",
        answer_sketch=(
            "We sample frames rather than decoding full video, so a single T4 "
            "is sufficient. The full grid is roughly 198 GPU-hours, run across "
            "free Kaggle and Colab allocations."
        ),
    ),
    # ---- the hostile ones -------------------------------------------
    Question(
        "Your cross-dataset drop is smaller than the literature's. Did you leak?",
        "rigour",
        hostile=True,
        needs=["baseline_b"],
        answer_sketch=(
            "Fair challenge, and we test for it rather than assert it. Splits are "
            "identity-disjoint using each dataset's official lists, and a CI gate "
            "runs perceptual-hash near-duplicate detection within and across "
            "datasets. A label-shuffle control gives chance accuracy. If the drop "
            "were small because of a leak, those checks would fire."
        ),
    ),
    Question(
        "Self-blended images are published work. What is yours?",
        "novelty",
        hostile=True,
        answer_sketch=(
            "SBI is prior work and cited as such. The contributions are the "
            "evaluation protocol -- threshold fixed on source validation, each "
            "target evaluated once on a frozen config -- and the audio-visual "
            "arm with a modality mask, which lets one checkpoint serve silent and "
            "audio-bearing datasets. Most published AV detectors cannot run on "
            "silent video at all."
        ),
    ),
    Question(
        "Isn't test-time adaptation just cheating?",
        "rigour",
        hostile=True,
        answer_sketch=(
            "It uses unlabelled target data, so yes, it is transductive. That is "
            "why it is reported in a separate row and labelled as such, never "
            "mixed into the headline number."
        ),
    ),
    Question(
        "Why does your model abstain so often? Isn't that avoiding the problem?",
        "framing",
        hostile=True,
        needs=["full_system"],
        answer_sketch=(
            "At cross-dataset accuracy a binary verdict about whether a real "
            "person faked a video is not defensible. The abstention band is "
            "calibrated on source validation and we report the full "
            "risk-coverage curve, so the trade-off is visible rather than hidden "
            "behind a single accuracy number."
        ),
    ),
    Question(
        "How do you know the lip-sync stream learned synchrony and not the generator?",
        "method",
        hostile=True,
        answer_sketch=(
            "It is trained contrastively on real video only. Having never seen a "
            "forgery, it cannot have encoded a generator's artefacts -- that is "
            "the point of the design, and the ablation isolates its contribution."
        ),
    ),
    Question(
        "What happens on a forgery method released after your datasets?",
        "limitations",
        hostile=True,
        answer_sketch=(
            "We expect degradation, and the out-of-distribution flag exists to "
            "say so at inference rather than return a confident wrong answer. We "
            "validate the flag by treating each held-out dataset as novel and "
            "reporting the detector-of-novelty AUROC."
        ),
    ),
    Question(
        "Could this be used to falsely accuse someone?",
        "ethics",
        hostile=True,
        answer_sketch=(
            "Yes, which is why the system never emits a bare binary verdict, "
            "always reports a calibrated probability with an abstention band, and "
            "the model card lists legal, forensic, journalistic, employment and "
            "immigration use as out of scope."
        ),
    ),
    Question(
        "Why did you drop the audio-visual experiment on Celeb-DF?",
        "data",
        answer_sketch=(
            "We did not drop it; it was never possible. FF++ and Celeb-DF ship "
            "without usable audio tracks, verified with ffprobe across every "
            "video. The audio-visual arm runs DFDC to FakeAVCeleb instead."
        ),
    ),
]


class DefenceAgent(Agent):
    name = "a6_defence"

    def register_tools(self) -> None:
        self.registry.add(
            "read_results",
            _read_results,
            "read the aggregated results (test results permitted)",
            pure=False,
        )

    # ------------------------------------------------------------------
    def analyse(self, results_dir: str | Path = "results") -> AgentResult:
        res = AgentResult(agent=self.name)
        available = _available_experiments(Path(results_dir))
        res.data["experiments_with_results"] = sorted(available)

        answerable, gaps = [], []
        for q in BANK:
            missing = [n for n in q.needs if n not in available]
            entry = {
                "question": q.q,
                "category": q.category,
                "hostile": q.hostile,
                "answer": q.answer_sketch,
                "missing_results": missing,
            }
            (gaps if missing else answerable).append(entry)

        res.data["answerable"] = answerable
        res.data["gaps"] = gaps

        findings = [
            Finding(
                severity="info",
                title=f"{len(answerable)} of {len(BANK)} questions answerable now",
                detail=(
                    f"{sum(1 for q in answerable if q['hostile'])} of them are the "
                    f"hostile kind. Questions answerable from the design alone do not "
                    f"need a result; the rest do."
                ),
                evidence={"answerable": [q["question"] for q in answerable]},
            )
        ]

        if gaps:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"{len(gaps)} question(s) have no supporting result yet",
                    detail=(
                        "A6 will not write an answer it cannot ground. Each of these "
                        "is a gap to close before the review, not a sentence to "
                        "improvise on the day."
                    ),
                    evidence={
                        "gaps": [{"q": g["question"], "needs": g["missing_results"]} for g in gaps]
                    },
                    suggested_action="Run the named experiments, or prepare to say plainly that the result does not exist yet.",
                )
            )

        hostile_unanswered = [g for g in gaps if g["hostile"]]
        if hostile_unanswered:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"{len(hostile_unanswered)} HOSTILE question(s) unsupported",
                    detail=(
                        "These are the ones that are asked when a reviewer is sceptical, "
                        "and the ones most damaging to improvise."
                    ),
                    evidence={"questions": [g["question"] for g in hostile_unanswered]},
                    suggested_action="Prioritise the experiments these depend on.",
                )
            )

        res.findings = findings
        res.summary = (
            f"{len(answerable)}/{len(BANK)} questions answerable from current "
            f"results; {len(gaps)} blocked on experiments that have not run."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        return (
            "You are preparing the author to defend a deepfake-detection "
            "project to reviewers. Below are the questions an automated "
            "agent judges answerable, with answer sketches.\n\n"
            "Tighten each answer to two or three spoken sentences. Keep them "
            "honest: where the work has a limitation, say it plainly rather than "
            "deflecting -- reviewers respond well to that. Do not add any number "
            "that is not in the sketches.\n\n"
            f"```json\n{json.dumps(result.data.get('answerable', []), indent=2)[:7000]}\n```"
        )


def _available_experiments(results_dir: Path) -> set[str]:
    p = results_dir / "by_experiment.csv"
    if not p.exists():
        return set()
    import pandas as pd

    try:
        return set(pd.read_csv(p).exp.astype(str))
    except Exception:
        return set()


def _read_results(path: str) -> list[dict]:
    import pandas as pd

    p = Path(path)
    return pd.read_csv(p).to_dict("records") if p.exists() else []


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = DefenceAgent()
    res = agent.run(results_dir=a.results)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "defence-sheet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
