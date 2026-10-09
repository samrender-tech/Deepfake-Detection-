"""Shareable analysis reports and suspicious-segment extraction.

Two things a user asks for once they have a verdict:

* **Where** in the clip the model saw something. The video-level number
  averages a forgery that is visible in a handful of frames away, so
  :func:`suspicious_segments` turns the per-frame scores into time ranges.
* **A copy they can keep.** :func:`render_html` writes one self-contained HTML
  file (inline SVG timeline, inline Grad-CAM PNGs, no external requests) that
  opens offline and prints cleanly.

Both read the ``Result.to_dict()`` shape -- the dict the CLI produces and the
API job store holds -- and nothing else, so a report cannot show a number the
detector did not produce. The disclaimer is rendered unconditionally: a report
is exactly the artefact that gets forwarded without its context, so it is the
last place the caveat may be dropped.
"""

from __future__ import annotations

import base64
import html
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from ddetect import __version__

#: Shown on every report. Kept textually identical to ``api.schemas.DISCLAIMER``
#: (asserted by the tests) without importing the API package, which needs the
#: optional serving dependencies.
REPORT_DISCLAIMER = (
    "This is an automated, probabilistic estimate from a research prototype. "
    "Accuracy drops substantially on manipulation methods the model has not "
    "seen. It is not evidence, and must not be used for legal, forensic, "
    "journalistic, employment or immigration decisions."
)

_VERDICT_LABEL = {
    "likely_authentic": "Likely authentic",
    "likely_manipulated": "Likely manipulated",
    "inconclusive": "Inconclusive",
}
_VERDICT_COLOUR = {
    "likely_authentic": "#15803d",
    "likely_manipulated": "#b91c1c",
    "inconclusive": "#a16207",
}


# --------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------
def suspicious_segments(
    frame_scores: Sequence[float],
    frame_timestamps: Sequence[float],
    threshold: float,
    merge_gap_s: float = 0.0,
) -> list[dict[str, float | int]]:
    """Contiguous runs of frames scoring at or above ``threshold``.

    Each segment is ``{start_s, end_s, peak_score, peak_time_s, n_frames}``.
    ``end_s`` extends to the next sampled frame's timestamp (or by the median
    frame spacing for the last frame), so a single flagged frame still has a
    visible width on a timeline. Runs separated by a gap no longer than
    ``merge_gap_s`` are merged.

    Frame scores are uncalibrated per-frame sigmoids, while the threshold is
    the run's decision threshold on the *calibrated* video probability. The
    comparison is the same one the UI's timeline draws as a dashed line; it
    marks where to look, not a second verdict.
    """
    scores = [float(s) for s in frame_scores]
    if not scores:
        return []
    if len(frame_timestamps) >= len(scores):
        ts = [float(t) for t in frame_timestamps[: len(scores)]]
    else:
        # A missing timestamp would silently shift every segment; fall back to
        # frame indices, which are at least monotone and honest about it.
        ts = [float(i) for i in range(len(scores))]
    if any(b < a for a, b in pairwise(ts)):
        raise ValueError("frame_timestamps must be non-decreasing")

    gaps = sorted(b - a for a, b in pairwise(ts) if b > a)
    step = gaps[len(gaps) // 2] if gaps else 1.0

    def end_of(i: int) -> float:
        return ts[i + 1] if i + 1 < len(ts) else ts[i] + step

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, s in enumerate(scores):
        if s >= threshold and start is None:
            start = i
        elif s < threshold and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(scores) - 1))

    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and ts[a] - end_of(merged[-1][1]) <= merge_gap_s:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    out: list[dict[str, float | int]] = []
    for a, b in merged:
        peak = max(range(a, b + 1), key=lambda i: scores[i])
        out.append(
            {
                "start_s": round(ts[a], 3),
                "end_s": round(end_of(b), 3),
                "peak_score": round(scores[peak], 4),
                "peak_time_s": round(ts[peak], 3),
                "n_frames": b - a + 1,
            }
        )
    return out


def segments_for(result: Mapping[str, Any]) -> list[dict[str, float | int]]:
    """:func:`suspicious_segments` applied to a ``Result.to_dict()``."""
    return suspicious_segments(
        result.get("frame_scores") or [],
        result.get("frame_timestamps") or [],
        float(result.get("threshold", 0.5)),
    )


# --------------------------------------------------------------------------
# summary row (batch CLI, JSON report)
# --------------------------------------------------------------------------
def summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """The flat, CSV-safe summary of one ``Result.to_dict()``."""
    segs = segments_for(result)
    return {
        "verdict": result["verdict"],
        "calibrated_prob": round(float(result["calibrated_prob"]), 4),
        "conformal_lo": float(result["conformal_lo"]),
        "conformal_hi": float(result["conformal_hi"]),
        "threshold": float(result["threshold"]),
        "ood_flag": bool(result["ood_flag"]),
        "calibration_reliable": not bool(result.get("calibration_degenerate", False)),
        "has_audio": bool(result["has_audio"]),
        "n_faces_found": int(result["n_faces_found"]),
        "n_suspicious_segments": len(segs),
        "peak_frame_score": max(result.get("frame_scores") or [0.0]),
        "n_warnings": len(result.get("warnings") or []),
        "model_version": result.get("model_version", ""),
    }


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
def _e(v: object) -> str:
    return html.escape(str(v), quote=True)


def _pct(p: float) -> str:
    return f"{100.0 * float(p):.1f}%"


def _timeline_svg(
    scores: Sequence[float],
    ts: Sequence[float],
    threshold: float,
    segments: Sequence[Mapping[str, float | int]],
) -> str:
    """Per-frame score as an inline SVG line chart, flagged segments shaded."""
    if not scores:
        return "<p class='muted'>No per-frame scores were produced for this clip.</p>"
    w, h, pad_l, pad_r, pad_t, pad_b = 640, 180, 34, 10, 10, 24
    t = list(ts[: len(scores)]) if len(ts) >= len(scores) else list(range(len(scores)))
    t0 = float(t[0])
    t1 = max(float(t[-1]), float(segments[-1]["end_s"]) if segments else 0.0, t0 + 1e-6)

    def x(v: float) -> float:
        return pad_l + (float(v) - t0) / (t1 - t0) * (w - pad_l - pad_r)

    def y(v: float) -> float:
        return pad_t + (1.0 - min(max(float(v), 0.0), 1.0)) * (h - pad_t - pad_b)

    parts = [
        f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='Per-frame manipulation score "
        f"over time' class='chart'>"
    ]
    for s in segments:
        x0, x1 = x(float(s["start_s"])), x(min(float(s["end_s"]), t1))
        parts.append(
            f"<rect x='{x0:.1f}' y='{pad_t}' width='{max(x1 - x0, 2):.1f}' "
            f"height='{h - pad_t - pad_b}' class='seg'/>"
        )
    for v in (0.0, 0.5, 1.0):
        parts.append(
            f"<line x1='{pad_l}' x2='{w - pad_r}' y1='{y(v):.1f}' y2='{y(v):.1f}' class='grid'/>"
            f"<text x='{pad_l - 6}' y='{y(v) + 4:.1f}' class='tick' text-anchor='end'>{v:g}</text>"
        )
    parts.append(
        f"<line x1='{pad_l}' x2='{w - pad_r}' y1='{y(threshold):.1f}' y2='{y(threshold):.1f}' "
        f"class='thr'/>"
    )
    pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(t, scores, strict=True))
    parts.append(f"<polyline points='{pts}' class='line'/>")
    for a, b in zip(t, scores, strict=True):
        parts.append(f"<circle cx='{x(a):.1f}' cy='{y(b):.1f}' r='2.5' class='dot'/>")
    for v in (t0, (t0 + t1) / 2, t1):
        parts.append(
            f"<text x='{x(v):.1f}' y='{h - 6}' class='tick' text-anchor='middle'>{v:.1f}s</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
:root{--fg:#18181b;--muted:#52525b;--line:#e4e4e7;--bg:#fff;--card:#fafafa;--accent:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px}
h1{font-size:22px;margin:4px 0 0}h2{font-size:15px;margin:28px 0 8px}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.verdict{border-radius:12px;padding:18px;color:#fff;margin-top:20px}
.verdict .big{font-size:26px;font-weight:600}.verdict .prob{font-size:30px;font-weight:600}
.row{display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px}
.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{font-weight:500;color:var(--muted)}.num{text-align:right;font-variant-numeric:tabular-nums}
.warn{background:#fef3c7;border-radius:10px;padding:10px 14px}
.disclaimer{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.chart{width:100%;height:auto}.chart .grid{stroke:var(--line)}
.chart .thr{stroke:#71717a;stroke-dasharray:4 3}.chart .line{fill:none;stroke:var(--accent);stroke-width:2}
.chart .dot{fill:var(--accent)}.chart .seg{fill:#f43f5e;opacity:.15}
.chart .tick{font-size:10px;fill:var(--muted)}
.cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.cams figure{margin:0}.cams img{width:100%;border-radius:8px}.cams figcaption{font-size:11px;color:var(--muted)}
footer{margin-top:32px;font-size:11px;color:var(--muted)}
@media print{.verdict{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def render_html(
    result: Mapping[str, Any],
    filename: str | None = None,
    images: Mapping[str, bytes] | None = None,
    generated_at: datetime | None = None,
    stability: Mapping[str, Any] | None = None,
) -> str:
    """One self-contained HTML report for one ``Result.to_dict()``.

    ``stability`` is an optional ``StabilityReport.to_dict()``; when given the
    report gains a section showing whether the verdict survives re-encoding.

    ``images`` maps an artefact key (``frame_3``, ``spectrum``) to PNG bytes;
    they are inlined as data URIs so the file makes no network request when
    opened. Every user-controlled string (the filename, warnings) is escaped.
    """
    verdict = str(result["verdict"])
    if verdict not in _VERDICT_LABEL:
        raise ValueError(f"unknown verdict {verdict!r}")
    images = images or {}
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    segs = segments_for(result)
    scores = list(result.get("frame_scores") or [])
    ts = list(result.get("frame_timestamps") or [])
    threshold = float(result.get("threshold", 0.5))

    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>AVFORGE analysis report</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<p class='eyebrow'>AVFORGE · research prototype · analysis report</p>",
        f"<h1>{_e(filename or result.get('video_id') or 'Uploaded clip')}</h1>",
        f"<p class='muted'>Generated {_e(when)}</p>",
        f"<section class='verdict' style='background:{_VERDICT_COLOUR[verdict]}'>",
        "<div class='row'><div><div class='eyebrow' style='color:#fff;opacity:.8'>Verdict</div>",
        f"<div class='big' data-verdict='{_e(verdict)}'>{_VERDICT_LABEL[verdict]}</div></div>",
        "<div style='text-align:right'><div class='eyebrow' style='color:#fff;opacity:.8'>"
        "Estimated probability of manipulation</div>",
        f"<div class='prob'>{_pct(result['calibrated_prob'])}</div></div></div>",
        f"<p style='margin:8px 0 0;opacity:.9'>Inconclusive band "
        f"{_pct(result['conformal_lo'])}–{_pct(result['conformal_hi'])} · decision threshold "
        f"{_pct(threshold)}</p></section>",
    ]

    flags: list[str] = []
    if result.get("ood_flag"):
        flags.append("This clip is unlike the videos the model was trained on.")
    if result.get("calibration_degenerate"):
        flags.append(
            "This model's calibration is not reliable; treat the percentage as indicative."
        )
    warnings = [str(w) for w in (result.get("warnings") or [])]
    if flags or warnings:
        out.append("<h2>Caveats for this clip</h2><div class='warn'><ul>")
        out.extend(f"<li>{_e(w)}</li>" for w in flags + warnings)
        out.append("</ul></div>")

    out.append("<h2>Per-frame score</h2>")
    out.append(_timeline_svg(scores, ts, threshold, segs))
    if segs:
        out.append(
            "<table><thead><tr><th>Flagged segment</th><th class='num'>Frames</th>"
            "<th class='num'>Peak score</th><th class='num'>Peak at</th></tr></thead><tbody>"
        )
        for s in segs:
            out.append(
                f"<tr><td>{float(s['start_s']):.2f}s – {float(s['end_s']):.2f}s</td>"
                f"<td class='num'>{int(s['n_frames'])}</td>"
                f"<td class='num'>{float(s['peak_score']):.3f}</td>"
                f"<td class='num'>{float(s['peak_time_s']):.2f}s</td></tr>"
            )
        out.append("</tbody></table>")
    elif scores:
        out.append("<p class='muted'>No frame scored above the decision threshold.</p>")

    streams = list(result.get("streams") or [])
    if streams:
        out.append(
            "<h2>What each check found</h2><table><thead><tr><th>Check</th>"
            "<th class='num'>Score</th><th class='num'>Reliability</th><th>Note</th>"
            "</tr></thead><tbody>"
        )
        for s in streams:
            sc = s.get("score")
            score = f"{float(sc):.3f}" if sc is not None and s.get("available") else "not run"
            out.append(
                f"<tr><td>{_e(s.get('name', ''))}</td><td class='num'>{score}</td>"
                f"<td class='num'>{float(s.get('reliability', 1.0)):.2f}</td>"
                f"<td class='muted'>{_e(s.get('note', ''))}</td></tr>"
            )
        out.append("</tbody></table>")

    cams = [(k, v) for k, v in images.items() if k != "spectrum" and v]
    if cams:
        out.append("<h2>Where the model looked</h2><div class='cams'>")
        for k, png in sorted(cams):
            out.append(
                f"<figure><img alt='attention overlay {_e(k)}' "
                f"src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
                f"<figcaption>{_e(k.replace('_', ' '))}</figcaption></figure>"
            )
        out.append("</div>")
    if images.get("spectrum"):
        spec = base64.b64encode(images["spectrum"]).decode()
        out.append(
            "<h2>Audio spectrum</h2>"
            f"<img alt='audio spectrum' style='width:100%' src='data:image/png;base64,{spec}'>"
        )

    if result.get("explanation"):
        src = (
            "generated summary"
            if result.get("explanation_source") == "agent"
            else "rule-based summary"
        )
        out.append(
            f"<h2>In plain language</h2><p>{_e(result['explanation'])}</p>"
            f"<p class='muted'>({src})</p>"
        )

    if stability is not None:
        out.append(_stability_html(stability))

    out.append(f"<h2>Limitations</h2><p class='disclaimer'>{_e(REPORT_DISCLAIMER)}</p>")
    out.append(
        "<table><tbody>"
        f"<tr><th>Model</th><td>{_e(result.get('model_version') or '—')}</td></tr>"
        f"<tr><th>Audio track</th><td>{'yes' if result.get('has_audio') else 'no'}</td></tr>"
        f"<tr><th>Faces found</th><td>{int(result.get('n_faces_found', 0))}</td></tr>"
        f"<tr><th>Report generator</th><td>ddetect {_e(__version__)}</td></tr>"
        "</tbody></table>"
    )
    out.append(
        "<footer>A detector, not a generator: this project creates no synthetic media.</footer>"
    )
    out.append("</main></body></html>")
    return "".join(out)


def _stability_html(st: Mapping[str, Any]) -> str:
    variants = list(st.get("variants") or [])
    scored = [v for v in variants if not v.get("error")]
    if not scored:
        head = "The re-encode check could not run on this clip."
    elif st.get("stable"):
        head = "The verdict held when the clip was re-compressed, as a messaging app would."
    else:
        head = (
            "The verdict changed or moved substantially when the clip was re-compressed. "
            "Treat it as resting on fragile evidence."
        )
    rows = "".join(
        f"<tr><td>{_e(v.get('name', ''))}</td>"
        f"<td>{_e(_VERDICT_LABEL.get(str(v.get('verdict')), v.get('error') or v.get('verdict')))}</td>"
        f"<td class='num'>{'' if v.get('error') else _pct(float(v.get('calibrated_prob', 0.0)))}</td></tr>"
        for v in variants
    )
    return (
        "<h2>Re-encode stability</h2>"
        f"<p data-stable='{str(bool(st.get('stable'))).lower()}'>{_e(head)}</p>"
        "<table><thead><tr><th>Variant</th><th>Verdict</th><th class='num'>Probability</th>"
        f"</tr></thead><tbody><tr><td>original</td>"
        f"<td>{_e(_VERDICT_LABEL.get(str(st.get('original_verdict')), ''))}</td>"
        f"<td class='num'>{_pct(float(st.get('original_prob', 0.0)))}</td></tr>{rows}"
        "</tbody></table>"
        f"<p class='muted'>Agreement {float(st.get('verdict_agreement', 0.0)):.0%} · "
        f"probability spread {_pct(float(st.get('prob_spread', 0.0)))}</p>"
    )


def render_batch_index(
    rows: Sequence[Mapping[str, Any]], generated_at: datetime | None = None
) -> str:
    """An index page for a batch run: counts per verdict and one row per clip.

    ``rows`` are the batch CLI's summary rows; a row with a ``report`` key
    links to that clip's report (a path relative to the index).
    """
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get("verdict"))] = counts.get(str(r.get("verdict")), 0) + 1
    order = [*_VERDICT_LABEL, "error"]
    tiles = "".join(
        f"<div class='tile'><div class='n'>{counts[k]}</div>"
        f"<div class='muted'>{_e(_VERDICT_LABEL.get(k, 'Failed'))}</div></div>"
        for k in order
        if counts.get(k)
    )
    body = []
    for r in rows:
        name = Path(str(r.get("file", ""))).name
        link = f"<a href='{_e(r['report'])}'>{_e(name)}</a>" if r.get("report") else _e(name)
        v = str(r.get("verdict"))
        label = _VERDICT_LABEL.get(v, "Failed")
        prob = "" if v == "error" else _pct(float(r.get("calibrated_prob", 0.0)))
        stab = r.get("stable")
        stab_txt = "" if stab in (None, "") else ("stable" if stab else "unstable")
        note = _e(r.get("error") or "")
        dot = _VERDICT_COLOUR.get(v, "#71717a")
        body.append(
            f"<tr><td><span class='dot' style='background:{dot}'></span>{link}</td>"
            f"<td>{_e(label)}</td><td class='num'>{prob}</td><td>{stab_txt}</td>"
            f"<td class='muted'>{note}</td></tr>"
        )
    css = _CSS + (
        ".tiles{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}"
        ".tile{background:var(--card);border:1px solid var(--line);border-radius:10px;"
        "padding:10px 14px;min-width:120px}.tile .n{font-size:22px;font-weight:600}"
        ".dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px}"
        ".scroll{overflow-x:auto}"
    )
    return "".join(
        [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>AVFORGE batch summary</title><style>{css}</style></head><body><main>",
            "<p class='eyebrow'>AVFORGE · research prototype · batch summary</p>",
            f"<h1>{len(rows)} clips analysed</h1><p class='muted'>Generated {_e(when)}</p>",
            f"<div class='tiles'>{tiles}</div>",
            "<h2>Clips</h2><div class='scroll'><table><thead><tr><th>Clip</th><th>Verdict</th>"
            "<th class='num'>Probability</th><th>Re-encode</th><th>Error</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>",
            f"<h2>Limitations</h2><p class='disclaimer'>{_e(REPORT_DISCLAIMER)}</p>",
            "</main></body></html>",
        ]
    )
