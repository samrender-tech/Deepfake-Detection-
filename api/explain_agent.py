"""A7 - the verdict explainer. The only agent that ships in the product.

This is the most safety-critical component in the system: it writes
user-facing prose about whether a real person's video is fake.

Three defences, in order of importance:

1. **A deterministic template fallback that is always available.** The product
   NEVER blocks on the LLM. No key, no network, a timeout, a refusal, or a
   failed guard all end at the template.
2. **Programmatic grounding checks on the generated text.** The prompt forbids
   overclaiming; this code *verifies* it. Any numeral not present in the
   payload, any certainty word, any identity or intent language, and the output
   is discarded in favour of the template.
3. **An audit record of every explanation** with its input payload hash, so a
   disputed verdict can be reconstructed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from agents.core.audit import record
from ddetect.utils.log import get_logger

log = get_logger(__name__)

PROMPT_VERSION = "a7/v1"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "prompts" / "a7" / "v1.md"
if not PROMPT_PATH.exists():  # running from the repo root
    PROMPT_PATH = Path("agents/prompts/a7/v1.md")

Source = Literal["agent", "template", "none"]

# --------------------------------------------------------------------------
# grounding guards
# --------------------------------------------------------------------------
#: Words that assert more certainty than a calibrated probability ever licenses.
_CERTAINTY = re.compile(
    r"\b(definitely|certainly|clearly|undoubtedly|obviously|proves?|proven|"
    r"confirms?|confirmed|guarantee[sd]?|without doubt|100%|conclusive(ly)?)\b",
    re.I,
)
#: Language that attributes identity, intent or blame.
_ATTRIBUTION = re.compile(
    r"\b(this person|the person in|the subject is|he |she |his |her |"
    r"deliberately|intentionally|malicious|fraudster|scammer|created by|"
    r"in order to deceive|trying to)\b",
    re.I,
)


def _numbers_in(text: str) -> set[str]:
    """Numerals mentioned in the text, normalised for comparison."""
    out = set()
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        v = float(m.group())
        out.add(f"{v:.4f}".rstrip("0").rstrip("."))
    return out


def _numbers_available(payload: dict[str, Any]) -> set[str]:
    """Every numeral the payload licenses the agent to mention.

    Includes percentage renderings of each probability, because "78%" is the
    natural way to say 0.78 and refusing it would push the agent toward vaguer,
    less useful prose.
    """
    out: set[str] = set()

    def add(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            v = float(x)
            for cand in (v, round(v, 2), round(v, 1), round(v * 100), round(v * 100, 1)):
                out.add(f"{float(cand):.4f}".rstrip("0").rstrip("."))
        elif isinstance(x, dict):
            for vv in x.values():
                add(vv)
        elif isinstance(x, (list, tuple)):
            for vv in x:
                add(vv)

    add(payload)
    # Small integers are structural ("two of the three streams"), not claims.
    out.update(str(i) for i in range(0, 65))
    return out


def check_grounding(text: str, payload: dict[str, Any]) -> list[str]:
    """Return the reasons this text must be rejected. Empty list = acceptable.

    Run on EVERY agent output before it reaches a user, and replayed by
    ``agents/evals/a7_grounding.yaml`` against payloads with values removed,
    swapped and contradicted.
    """
    problems: list[str] = []

    if m := _CERTAINTY.search(text):
        problems.append(f"overclaims certainty: {m.group()!r}")
    if m := _ATTRIBUTION.search(text):
        problems.append(f"attributes identity or intent: {m.group().strip()!r}")

    allowed = _numbers_available(payload)
    invented = sorted(_numbers_in(text) - allowed)
    if invented:
        problems.append(f"cites numbers absent from the payload: {invented[:5]}")

    if payload.get("abstained") or payload.get("ood_flag"):
        # Must lead with the uncertainty, not bury it.
        head = text.strip().split(".")[0].lower()
        hedges = (
            "not confident",
            "inconclusive",
            "cannot",
            "can't",
            "could not",
            "couldn't",
            "unable",
            "uncertain",
            "unlike",
            "not reliable",
            "insufficient",
            "no face",
            "not enough",
            "should not be relied",
            "too",
            "no verdict",
        )
        if not any(w in head for w in hedges):
            problems.append("abstained/OOD result does not lead with the uncertainty")

    if len(text.split()) > 130:
        problems.append("too long for a verdict summary")
    if len(text.strip()) < 20:
        problems.append("empty or trivially short")
    return problems


# --------------------------------------------------------------------------
# the deterministic fallback -- always correct, never blocked
# --------------------------------------------------------------------------
def template_explanation(p: dict[str, Any]) -> str:
    """Build an explanation from the payload with no model involved.

    This is the baseline the product ships with. It is not a degraded mode: it
    is accurate, grounded by construction, and the thing users see whenever the
    agent is unavailable or its output fails the guards.
    """
    bits: list[str] = []
    prob = p.get("calibrated_probability")
    pct = f"{prob * 100:.0f}%" if isinstance(prob, (int, float)) else "unknown"

    if p.get("n_faces_found") == 0:
        return (
            "No face could be located in this video, so the detector had nothing "
            "to analyse and this result should not be relied on. Try a clip where "
            "a face is visible and unobstructed for at least a second."
        )

    if p.get("abstained") or p.get("ood_flag"):
        reason = (
            "this video's characteristics are unlike the data the model was trained on"
            if p.get("ood_flag")
            else "the score falls inside the band where the model is not reliable enough to call it"
        )
        bits.append(f"The system is not confident enough to give a verdict here, because {reason}.")
        bits.append(f"Its raw estimate was {pct}, which is within the inconclusive range.")
    elif p.get("verdict") == "likely_manipulated":
        bits.append(f"The detector estimates a {pct} probability that this video was manipulated.")
    else:
        bits.append(
            f"The detector estimates a {pct} probability of manipulation, which is below its decision threshold."
        )

    # Evidence, strictly from the payload.
    peak_t = p.get("peak_frame_time_s")
    hi = p.get("frame_score_max")
    if isinstance(peak_t, (int, float)) and isinstance(hi, (int, float)):
        bits.append(f"The strongest single-frame signal was {hi:.2f} at about {peak_t:.1f}s.")

    if not p.get("has_audio"):
        bits.append("This clip has no audio track, so the voice and lip-sync checks did not run.")
    else:
        streams = {s["name"]: s for s in p.get("streams", []) if isinstance(s, dict)}
        sync = streams.get("sync")
        if sync and sync.get("note"):
            bits.append(f"The lip-sync check was limited: {sync['note']}.")

    for w in p.get("warnings", [])[:1]:
        bits.append(str(w))

    return " ".join(bits)


# --------------------------------------------------------------------------
# the agent path
# --------------------------------------------------------------------------
def _call_claude(payload: dict[str, Any], timeout: float = 20.0) -> tuple[str | None, int]:
    """One grounded, deterministic call. Returns (text, tokens).

    temperature 0 and a pinned prompt version: the same payload must produce
    the same explanation, or the audit log cannot reconstruct what a user saw.
    """
    import json

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, 0
    try:
        from anthropic import Anthropic
    except ImportError:
        log.debug("anthropic SDK not installed; using the template explainer")
        return None, 0

    try:
        prompt = PROMPT_PATH.read_text().replace(
            "{payload}", json.dumps(payload, indent=2, default=str)
        )
    except OSError:
        return None, 0

    try:
        client = Anthropic(api_key=key, timeout=timeout)
        resp = client.messages.create(
            model=os.environ.get("AGENT_MODEL", "claude-opus-5-5"),
            max_tokens=400,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        tokens = resp.usage.input_tokens + resp.usage.output_tokens
        return text.strip(), tokens
    except Exception as e:
        log.warning("A7 call failed (%s); using the template explainer", type(e).__name__)
        return None, 0


def explain_result(payload: dict[str, Any]) -> tuple[str, Source]:
    """Produce a user-facing explanation. Never raises, never blocks.

    Returns ``(text, source)`` where source is "agent" or "template" so the UI
    can label it honestly.
    """
    fallback = template_explanation(payload)

    if os.environ.get("DDETECT_DISABLE_AGENT") == "1":
        return fallback, "template"

    text, tokens = _call_claude(payload)
    if not text:
        record(
            "a7_explainer",
            "fallback",
            prompt_version=PROMPT_VERSION,
            inputs=payload,
            output=fallback,
            ok=True,
            note="agent unavailable",
        )
        return fallback, "template"

    problems = check_grounding(text, payload)
    if problems:
        # The guard fired. Discard the generated text entirely -- there is no
        # partial-credit path for an ungrounded claim about a real person.
        record(
            "a7_explainer",
            "rejected",
            prompt_version=PROMPT_VERSION,
            inputs=payload,
            output=text,
            tokens=tokens,
            ok=False,
            note="; ".join(problems),
        )
        log.warning("A7 output rejected (%s); using the template", "; ".join(problems))
        return fallback, "template"

    record(
        "a7_explainer",
        "accepted",
        prompt_version=PROMPT_VERSION,
        inputs=payload,
        output=text,
        tokens=tokens,
        ok=True,
    )
    return text, "agent"
