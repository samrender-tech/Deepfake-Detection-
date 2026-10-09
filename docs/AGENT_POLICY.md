# Agent policy

Eight agents support this project. Seven accelerate the research; one (**A7**)
ships inside the product. This document states what each may touch, and —
more importantly — what none of them may.

Every rule below is enforced in code and asserted by `tests/test_agents.py`. A
policy that lives only in a document is not a control.

---

## The test-set firewall

**The project's entire claim is an honest cross-dataset number.** An agent that
reads target-domain test results and then proposes a model change has performed
manual test-set optimisation by proxy. The headline number becomes worthless —
and every log still looks like diligent research.

So:

- `agents/core/firewall.py` mediates every data and results read.
- **A3** (experiment orchestration) and **A4** (failure diagnosis) may read
  **only** `train` and `val` splits of the **source** dataset. A target-split
  or target-dataset read raises `FirewallViolation` and is written to the audit
  log.
- **A5** (paper) and **A6** (defence prep) may read test results, because by
  then the configuration is frozen and neither has any training tool. A5
  cannot propose a model change; its tool registry contains no way to do so.
- Target test sets are evaluated **once per frozen configuration, by a human**,
  through `make eval-final`. That target refuses to run on a dirty working
  tree: a number stamped with a `-dirty` sha cannot be reproduced, so it is not
  a result.

---

## The agents

| | Agent | Reads | Writes | Notes |
|---|---|---|---|---|
| A1 | Literature | arXiv, Semantic Scholar, OpenAlex | `paper/refs.bib`, `proposals/` | A citation enters only with a resolving DOI/arXiv id **and** a fetched abstract. Evaluated against seeded fabricated references; it must reject all of them. |
| A2 | Data curation | manifests, `ffprobe`, file stats | `docs/DATASET_CARD.md`, `proposals/` | Produces the audio evidence table and the leakage sweep. |
| A3 | Experiment orchestration | **source train/val only** | configs, `docs/EXPERIMENT_LOG.md` | Hard GPU-hour budget. No delete tool. Cannot modify `ddetect/` — it composes configs, it does not write model code. Every launch logs its hypothesis, so post-hoc rationalisation is visible. |
| A4 | Failure diagnosis | **source val predictions only** | `proposals/` | Clusters failures and names concrete, testable fixes. |
| A5 | Paper | results incl. test, figures | `paper/` | Maintains `paper/claims.csv` mapping every quantitative sentence to a results cell; fails if a claim has no backing number. **No training tools.** |
| A6 | Defence prep | results, docs | `proposals/` | Refuses to answer a question no result supports; surfaces it as a gap instead. |
| **A7** | **Verdict explainer — ships** | one job's explainability payload | the API response | See below. |
| A8 | Red team | source val only | degraded manifests | Composes and degrades **existing real media only**. No generative tool, no network. Human review before any recipe enters training augmentation. |

---

## A7 — the one that ships

A7 writes user-facing prose about whether a real person's video is fake. It is
the most safety-critical component in the system, and it is defended three
ways:

**1. A deterministic template that is always available.** The product never
blocks on the language model. No API key, no network, a timeout, a refusal or
a failed guard all end at the same template — which is accurate by
construction and passes the same guards the model is held to
(`tests/test_agents.py` asserts this for every branch).

**2. Programmatic grounding checks on the generated text.** The prompt forbids
overclaiming; the code *verifies* it. Output is discarded entirely if it:

- cites any numeral not present in its input payload
- uses certainty language ("definitely", "proves", "conclusively")
- attributes identity, intent or blame to anyone
- fails to lead with the uncertainty when the result is inconclusive or OOD
- runs long, or is empty

There is no partial-credit path. An ungrounded claim about a real person is
replaced, not edited.

**3. An audit record of every explanation**, with a hash of its input payload,
so any served text can be reconstructed. The log stores hashes and a short
preview rather than the full payload: it contains per-frame scores about an
identifiable person's video, and the audit log is not the place to accumulate
that.

A7 runs at temperature 0 with a pinned prompt version, so the same payload
produces the same explanation.

---

## Implementation status

All eight agents exist. Seven run as batch tools (`python -m agents.a2_data
--manifest ...`); A7 ships inside the serving path.

Each is built the same way, and the shape is deliberate:

    deterministic analysis (Python)  ->  optional LLM synthesis  ->  proposal

The analysis is the valuable part and runs with **no API key**: A2 really does
probe every file and sweep for leakage, A4 really does cluster the failures,
A5 really does resolve each claim against `results/`. The model is a narrative
layer on top. So the agents are useful offline, testable without mocking a
model, and their *conclusions* are reproducible even when their prose is not.

`make agents` runs the eval suites; `make agent-audit` runs the research
agents over the current repository state.

## Shared rules

- **Human approval** for any write outside `proposals/`, and for any compute
  spend.
- **Token and wall-clock budgets** per agent, with hard stops.
- **Versioned prompts** under `agents/prompts/<agent>/vN.md`, referenced by
  version in every log line.
- **Response caching** keyed on `(prompt_version, tool_args_hash)`, so an
  agent-assisted result is reproducible.
- **Append-only audit log** at `audit/agent_actions.jsonl`.
- **CI gates**: A1's citation eval and A7's grounding eval block the build.

## Disclosure

Agents are a research and product accelerator, never an authority. No agent
output enters the paper or the product without a human signing off. The paper
contains a subsection stating how agents were used and what they were
forbidden from reading.
