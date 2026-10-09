"""Run the agent eval suites.

    python -m agents.evals.runner              # all suites
    python -m agents.evals.runner --blocking   # only the CI-blocking ones

Two suites block the build:

* **a1_citations** -- a hallucinated reference is the highest-cost error a
  paper can contain, and it survives review.
* **a7_grounding** -- A7 writes user-facing prose about whether a real person's
  video is fake.

None of these call a model. They replay recorded inputs through the
deterministic guards, which is what makes them cheap enough to run on every
commit and reliable enough to block on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent


@dataclass
class CaseResult:
    suite: str
    case_id: str
    passed: bool
    detail: str = ""


# ==========================================================================
def run_a7_grounding(spec: dict[str, Any]) -> list[CaseResult]:
    from api.explain_agent import check_grounding

    out = []
    for case in spec["cases"]:
        payload = {**spec["base_payload"], **case.get("payload_overrides", {})}
        problems = check_grounding(case["text"], payload)
        rejected = bool(problems)
        want_reject = case["expect"] == "reject"
        ok = rejected == want_reject
        detail = f"expected {case['expect']}, guard said {'reject' if rejected else 'pass'}" + (
            f" ({'; '.join(problems)})" if problems else ""
        )
        out.append(CaseResult(spec["name"], case["id"], ok, "" if ok else detail))
    return out


def run_a1_citations(spec: dict[str, Any]) -> list[CaseResult]:
    from agents.a1_literature import _verify

    out = []
    for case in spec["cases"]:
        v = _verify(entry=case["entry"])
        expect = case["expect"]
        if expect == "accept":
            ok = not v["missing_identifier"] and not v["missing_fields"]
        elif expect == "reject":
            ok = v["missing_identifier"]
        else:  # incomplete
            ok = bool(v["missing_fields"])
        out.append(
            CaseResult(
                spec["name"],
                case["id"],
                ok,
                "" if ok else f"expected {expect}, got {v}",
            )
        )
    return out


def run_a5_claims(spec: dict[str, Any]) -> list[CaseResult]:
    import tempfile

    from agents.a5_paper import _resolve

    out = []
    for case in spec["cases"]:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.csv"
            p.write_text(case["results_csv"])
            got = _resolve(p, case["column"], case["selector"])
            want = case["expect_value"]
            if want is None:
                ok = got is None
            else:
                ok = got is not None and abs(float(got) - float(want)) < 1e-9
            out.append(
                CaseResult(
                    spec["name"],
                    case["id"],
                    ok,
                    "" if ok else f"expected {want}, got {got}",
                )
            )
    return out


RUNNERS = {
    "a7_grounding": run_a7_grounding,
    "a1_citations": run_a1_citations,
    "a5_claims": run_a5_claims,
}


# ==========================================================================
def run_suite(path: Path) -> tuple[dict[str, Any], list[CaseResult]]:
    spec = yaml.safe_load(path.read_text())
    runner = RUNNERS.get(spec["name"])
    if runner is None:
        raise KeyError(f"no runner registered for suite {spec['name']!r}")
    return spec, runner(spec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocking", action="store_true", help="only CI-blocking suites")
    ap.add_argument("--suite", default=None, help="run one suite by name")
    a = ap.parse_args(argv)

    specs = sorted(EVAL_DIR.glob("*.yaml"))
    failures: list[CaseResult] = []
    blocking_failures: list[CaseResult] = []

    for p in specs:
        spec, results = run_suite(p)
        if a.suite and spec["name"] != a.suite:
            continue
        is_blocking = bool(spec.get("blocking"))
        if a.blocking and not is_blocking:
            continue

        passed = sum(r.passed for r in results)
        tag = "BLOCKING" if is_blocking else "advisory"
        print(f"\n{spec['name']}  ({tag})  {passed}/{len(results)} passed")
        for r in results:
            if not r.passed:
                print(f"  FAIL {r.case_id}: {r.detail}")
                failures.append(r)
                if is_blocking:
                    blocking_failures.append(r)

    print()
    if blocking_failures:
        print(
            f"{len(blocking_failures)} BLOCKING eval failure(s). These guard the "
            f"two things an agent must never do: fabricate a citation, or make an "
            f"ungrounded claim about a real person's video."
        )
        return 1
    if failures:
        print(f"{len(failures)} advisory failure(s); not blocking the build.")
        return 0
    print("all agent evals passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
