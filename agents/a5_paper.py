"""A5 - paper agent.

Keeps every quantitative sentence in the paper tied to a cell in ``results/``.

A5 is the one research agent permitted to read test results, and the reason is
structural rather than a matter of trust: by the time it runs the configuration
is frozen, and its tool registry contains no training tool, so it *cannot* act
on what it sees. It can only describe it. That is the difference between
reporting a cross-dataset number and optimising against one.

The claims check is deterministic: ``paper/claims.csv`` maps each claim to a
results file, column and row selector, and A5 resolves every one. A claim with
no backing number, or one that disagrees with the number it cites, is a
blocker. The model only drafts prose around claims that already check out.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from agents.core.loop import Agent, AgentResult, Finding

#: Words that promise more than a cross-dataset detector can deliver. "Do not
#: oversell" is a correctness criterion for this project, not a style
#: preference.
OVERCLAIM = re.compile(
    r"\b(solves?|solved|robust to all|state[- ]of[- ]the[- ]art|"
    r"significantly outperforms|proves?|guarantees?|eliminates?|"
    r"fully generaliz|completely)\b",
    re.I,
)


class PaperAgent(Agent):
    name = "a5_paper"

    def register_tools(self) -> None:
        # Test results ARE readable here -- and there is
        # deliberately no training or config tool to act on them with.
        self.registry.add(
            "read_results", _read_results, "read a results CSV (test results permitted)", pure=False
        )
        self.registry.add("read_claims", _read_claims, "read the claims table", pure=False)

    # ------------------------------------------------------------------
    def analyse(
        self,
        claims: str | Path = "paper/claims.csv",
        results_dir: str | Path = "results",  # noqa: ARG002 - paths come from claims.csv
        tex: str | Path = "paper/main.tex",
    ) -> AgentResult:
        res = AgentResult(agent=self.name)
        findings: list[Finding] = []
        rows = _read_claims(str(claims))
        res.data["n_claims"] = len(rows)

        verified, unbacked, mismatched = [], [], []
        for row in rows:
            cid = row.get("claim_id", "?")
            expected = (row.get("expected_value") or "").strip()
            rf = (row.get("results_file") or "").strip()
            col = (row.get("column") or "").strip()

            if not rf or expected.upper() == "PENDING":
                unbacked.append(
                    {
                        "claim_id": cid,
                        "reason": "no result recorded yet",
                        "fragment": row.get("sentence_fragment", ""),
                    }
                )
                continue

            actual = _resolve(Path(rf), col, row.get("row_selector", ""))
            if actual is None:
                unbacked.append(
                    {
                        "claim_id": cid,
                        "reason": f"{rf}[{col}] could not be resolved",
                        "fragment": row.get("sentence_fragment", ""),
                    }
                )
                continue

            try:
                if abs(float(actual) - float(expected)) > 1e-3:
                    mismatched.append(
                        {
                            "claim_id": cid,
                            "in_paper": expected,
                            "in_results": actual,
                            "source": f"{rf}[{col}]",
                        }
                    )
                else:
                    verified.append(cid)
            except (TypeError, ValueError):
                if str(actual).strip() != expected:
                    mismatched.append(
                        {
                            "claim_id": cid,
                            "in_paper": expected,
                            "in_results": actual,
                            "source": f"{rf}[{col}]",
                        }
                    )
                else:
                    verified.append(cid)

        res.data.update(verified=verified, unbacked=unbacked, mismatched=mismatched)

        if mismatched:
            findings.append(
                Finding(
                    severity="blocker",
                    title=f"{len(mismatched)} claim(s) disagree with results/",
                    detail=(
                        "The number written in the paper is not the number the "
                        "aggregator produced. This is what happens after a re-run: "
                        "the tables regenerate and the prose does not."
                    ),
                    evidence={"claims": mismatched[:10]},
                    suggested_action="Regenerate the prose from results/, or re-run the experiment if the results file is stale.",
                )
            )
        if unbacked:
            findings.append(
                Finding(
                    severity="blocker" if len(unbacked) < len(rows) else "warn",
                    title=f"{len(unbacked)} claim(s) have no backing number",
                    detail=(
                        "Every quantitative sentence must resolve to a cell in "
                        "results/. An unbacked claim is either a number someone typed "
                        "by hand, or an experiment that has not been run."
                    ),
                    evidence={"claims": unbacked[:10]},
                    suggested_action=(
                        "Run the experiment and `python -m experiments.aggregate`, or "
                        "remove the sentence. Do not type the number in."
                    ),
                )
            )

        # ---- overclaiming language --------------------------------------
        tex_p = Path(tex)
        if tex_p.exists():
            text = tex_p.read_text()
            hits = []
            for m in OVERCLAIM.finditer(text):
                line = text[: m.start()].count("\n") + 1
                hits.append(
                    {
                        "line": line,
                        "phrase": m.group(),
                        "context": text[max(0, m.start() - 70) : m.end() + 70].replace("\n", " "),
                    }
                )
            res.data["overclaims"] = hits
            if hits:
                findings.append(
                    Finding(
                        severity="warn",
                        title=f"{len(hits)} phrase(s) claim more than the results support",
                        detail=(
                            "The project's own guidance is 'do not oversell -- expect to "
                            "close part of the gap, not all of it'. Reviewers "
                            "respond well to a measured failure and badly to an "
                            "overstated success."
                        ),
                        evidence={"hits": hits[:8]},
                        suggested_action="Replace with a hedged form naming the measured effect size.",
                    )
                )

            todos = len(re.findall(r"\\todo\{", text))
            res.data["todo_count"] = todos
            if todos:
                findings.append(
                    Finding(
                        severity="info",
                        title=f"{todos} \\todo{{}} placeholder(s) remain",
                        detail="These render visibly in the PDF; none may survive submission.",
                        evidence={"count": todos},
                    )
                )

            if "\\input{../results/" not in text:
                findings.append(
                    Finding(
                        severity="warn",
                        title="no generated table is \\input into the paper",
                        detail=(
                            "Tables must be \\input from results/, not pasted. A pasted "
                            "table goes stale silently the next time anything is re-run."
                        ),
                        evidence={},
                        suggested_action="Use \\input{../results/table_main.tex}.",
                    )
                )

        res.findings = findings
        res.ok = not any(f.severity == "blocker" for f in findings)
        res.summary = (
            f"{len(verified)}/{len(rows)} claims verified against results/; "
            f"{len(unbacked)} unbacked, {len(mismatched)} mismatched."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        if not result.data.get("verified"):
            return None
        return (
            "You are drafting the Results section of a paper on cross-dataset "
            "deepfake detection. Below are the claims that have been VERIFIED "
            "against generated result files.\n\n"
            "Write the section using only these numbers. Lead with the "
            "cross-dataset gap -- the measured failure is the contribution, not "
            "an embarrassment. Do not use 'state-of-the-art', 'proves', "
            "'significantly outperforms' or similar. Report effect sizes with "
            "their confidence intervals.\n\n"
            f"```json\n{json.dumps(result.data, indent=2, default=str)[:7000]}\n```"
        )


def _read_claims(path: str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(lines))


def _read_results(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return pd.read_csv(p).to_dict("records")  # type: ignore[no-any-return]


def _resolve(results_file: Path, column: str, selector: str) -> Any:
    """Look up one cell. ``selector`` is ``col=value;col2=value2``."""
    if not results_file.exists():
        return None
    try:
        df = pd.read_csv(results_file)
    except Exception:
        return None
    if column not in df.columns:
        return None
    for clause in filter(None, (selector or "").split(";")):
        if "=" not in clause:
            continue
        k, v = clause.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in df.columns:
            df = df[df[k].astype(str) == v]
    if df.empty:
        return None
    return df[column].iloc[0]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="paper/claims.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tex", default="paper/main.tex")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = PaperAgent()
    res = agent.run(claims=a.claims, results_dir=a.results, tex=a.tex)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "paper-check")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
