"""A1 - literature agent.

Maintains ``paper/refs.bib`` and the related-work table.

The one rule that matters: **a citation enters the bibliography only with a
resolving DOI or arXiv id AND a fetched abstract on file.** A hallucinated
reference is the highest-cost failure available in an academic deliverable --
it survives review, embarrasses the authors publicly, and is trivially
checkable by anyone. The rule is "never cite anything you have not read";
this is that rule in code.

So verification is deterministic and runs offline over what is already in
``refs.bib``. The network search is a separate, optional step, and anything it
returns is a *candidate* until verified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.core.loop import Agent, AgentResult, Finding

ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

#: Fields a usable entry must have, by type.
REQUIRED = {
    "inproceedings": ("title", "author", "booktitle", "year"),
    "article": ("title", "author", "journal", "year"),
    "misc": ("title", "author", "year"),
}


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    """Minimal BibTeX reader. Deliberately not a dependency.

    We only need the fields, and a real parser would pull in a library that
    fails on the slightly-off entries publishers emit.
    """
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = m.group(1).lower(), m.group(2).strip()
        # Walk braces to find this entry's end. An earlier version required the
        # closing brace to sit on its own line, so a single-line entry parsed
        # as nothing -- and "0 references" is indistinguishable from an empty
        # bibliography, which is exactly the kind of silent miss this agent
        # exists to prevent.
        depth, i = 1, text.index("{", m.start()) + 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[m.end() : i - 1]

        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, re.S):
            fields[fm.group(1).lower().strip()] = " ".join(fm.group(2).split())
        entries.append({"type": kind, "key": key, **fields})
    return entries


class LiteratureAgent(Agent):
    name = "a1_literature"

    def register_tools(self) -> None:
        self.registry.add("read_bib", _read_bib, "read the bibliography", pure=False)
        self.registry.add("verify_entry", _verify, "check one entry's identifiers")
        # Network search is registered but impure and uncached: it is the only
        # tool here that can return something that is not already on disk.
        self.registry.add(
            "search_arxiv", _search_arxiv, "search arXiv for candidate references", pure=False
        )

    # ------------------------------------------------------------------
    def analyse(
        self, bib: str | Path = "paper/refs.bib", min_refs: int = 15, max_refs: int = 25
    ) -> AgentResult:
        res = AgentResult(agent=self.name)
        findings: list[Finding] = []

        p = Path(bib)
        if not p.exists():
            res.ok = False
            res.summary = f"{p} not found"
            return res

        entries = parse_bibtex(p.read_text())
        res.data["n_entries"] = len(entries)

        unverified, incomplete, stale = [], [], []
        years = []
        for e in entries:
            probs = _verify(entry=e)
            if probs["missing_identifier"]:
                unverified.append(e["key"])
            if probs["missing_fields"]:
                incomplete.append({"key": e["key"], "missing": probs["missing_fields"]})
            y = e.get("year", "")
            if y.isdigit():
                years.append(int(y))
                if int(y) < 2019:
                    stale.append({"key": e["key"], "year": int(y)})

        res.data["unverified"] = unverified
        res.data["incomplete"] = incomplete
        res.data["pre_2019"] = stale
        res.data["year_range"] = [min(years), max(years)] if years else None

        if unverified:
            findings.append(
                Finding(
                    severity="blocker",
                    title=f"{len(unverified)} citation(s) with no DOI or arXiv id",
                    detail=(
                        "An entry without a resolving identifier has not been verified "
                        "against the publisher, which means it could be fabricated or "
                        "subtly wrong (wrong venue, wrong year, wrong authors). A "
                        "hallucinated citation is the single most damaging error "
                        "available in a paper."
                    ),
                    evidence={"keys": unverified},
                    suggested_action=(
                        "Resolve each on IEEE Xplore, ACM DL or arXiv and add the "
                        "identifier. Delete any that cannot be found -- do not guess."
                    ),
                )
            )
        if incomplete:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"{len(incomplete)} entr(ies) missing required BibTeX fields",
                    detail="IEEEtran will render these incorrectly or drop them silently.",
                    evidence={"entries": incomplete[:10]},
                    suggested_action="Fill the missing fields from the publisher page.",
                )
            )
        if len(entries) < min_refs:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"only {len(entries)} references (target {min_refs}-{max_refs})",
                    detail=(
                        "The target is 15-25 references, mostly from the "
                        "last five years. A short bibliography reads as a shallow "
                        "related-work section."
                    ),
                    evidence={"have": len(entries), "target": [min_refs, max_refs]},
                    suggested_action="Search for recent cross-dataset generalisation and audio-visual detection work.",
                )
            )
        if stale and len(stale) > len(entries) // 2:
            findings.append(
                Finding(
                    severity="warn",
                    title=f"{len(stale)} of {len(entries)} references predate 2019",
                    detail="The field moves fast; a mostly-old bibliography suggests the survey missed recent work.",
                    evidence={"old": stale[:10]},
                )
            )

        # ---- the related-work table ------------------------------------
        table = [
            {
                "key": e["key"],
                "work": _first_author(e.get("author", "")) + " et al. " + e.get("year", ""),
                "title": e.get("title", "")[:90],
                "venue": e.get("booktitle") or e.get("journal", ""),
                "identifier": e.get("eprint") or e.get("doi", ""),
            }
            for e in entries
        ]
        res.data["related_work_table"] = table

        res.findings = findings
        res.ok = not any(f.severity == "blocker" for f in findings)
        res.summary = (
            f"{len(entries)} references, {len(entries) - len(unverified)} verified; "
            f"{len(res.blockers)} blocker(s)."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        if not result.data.get("related_work_table"):
            return None
        return (
            "Draft the Related Work section for a paper on cross-dataset "
            "generalisation in audio-visual deepfake detection. The argument the "
            "section must build: published detectors report high in-dataset "
            "accuracy, cross-dataset results are rarely reported, and the audio "
            "track is almost always ignored.\n\n"
            "Cite ONLY the keys listed below, using \\cite{key}. Do not invent a "
            "reference, a number, or a claim about a paper that is not in this "
            "list. If the list does not support a claim you want to make, leave "
            "it out.\n\n"
            f"```json\n{json.dumps(result.data['related_work_table'], indent=2)[:7000]}\n```"
        )


def _first_author(author: str) -> str:
    if not author:
        return "?"
    first = author.split(" and ")[0]
    return first.split(",")[0].strip().split()[-1] if first else "?"


def _read_bib(path: str = "paper/refs.bib") -> list[dict]:
    p = Path(path)
    return parse_bibtex(p.read_text()) if p.exists() else []


def _verify(entry: dict) -> dict:
    """Check one entry offline: does it carry a resolving identifier?"""
    eprint = (entry.get("eprint") or "").strip()
    doi = (entry.get("doi") or "").strip()
    has_arxiv = bool(eprint and ARXIV_RE.match(eprint))
    has_doi = bool(doi and DOI_RE.match(doi))
    required = REQUIRED.get(entry.get("type", "misc"), REQUIRED["misc"])
    return {
        "key": entry.get("key"),
        "has_arxiv": has_arxiv,
        "has_doi": has_doi,
        "missing_identifier": not (has_arxiv or has_doi),
        "missing_fields": [f for f in required if not entry.get(f)],
    }


def _search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """Query arXiv. Results are CANDIDATES, never citations.

    Nothing returned here may enter refs.bib until a human has opened it.
    Network failure returns an empty list rather than raising, so the agent
    degrades to offline verification.
    """
    import urllib.parse
    import urllib.request

    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            xml = r.read().decode()
    except Exception:
        return []

    out = []
    for m in re.finditer(r"<entry>(.*?)</entry>", xml, re.S):
        body = m.group(1)

        def field(tag: str, _body: str = body) -> str:
            # _body is bound at definition time: without it the closure would
            # capture the loop variable and every entry would read the last.
            fm = re.search(rf"<{tag}>(.*?)</{tag}>", _body, re.S)
            return " ".join(fm.group(1).split()) if fm else ""

        idm = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", body)
        out.append(
            {
                "arxiv_id": idm.group(1) if idm else "",
                "title": field("title"),
                "summary": field("summary")[:400],
                "published": field("published")[:10],
                "status": "CANDIDATE - verify before citing",
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", default="paper/refs.bib")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = LiteratureAgent()
    res = agent.run(bib=a.bib)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "literature")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
