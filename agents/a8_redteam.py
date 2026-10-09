"""A8 - red team.

Searches for transformations that break the detector, so the paper can report
its own failure modes rather than wait for a reviewer to find them.

SCOPE: it composes and degrades **existing real media
only** -- re-encoding, frame-rate conversion, letterboxing, cropping, and
splicing real audio from one real video onto another. It has no generative
tool and no network access, so no synthetic media can be produced here even by
accident. ``scripts/scope_audit.py`` enforces that at the dependency level.

The audio-splice case is worth naming: taking real audio from clip A and
putting it on real clip B produces genuine audio-visual desynchronisation with
no forgery involved. It is a true-positive check for the sync stream -- the
detector *should* flag it -- and simultaneously a false-positive risk, because
the video itself is authentic. The paper needs to say which it treats as.

Every recipe is reviewed by a human before entering the training augmentation
set; the agent proposes, it does not apply.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.core.loop import Agent, AgentResult, Finding


@dataclass
class Recipe:
    """One stress transformation, expressed as ffmpeg arguments."""

    name: str
    kind: str  # compression | geometry | temporal | audio | combined
    args: list[str]
    rationale: str
    expect: str  # what SHOULD happen if the detector is sound
    severity: str = "warn"
    needs_second_clip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Everything below is a lossy transform of EXISTING media. No generation.
RECIPES: list[Recipe] = [
    # ---- compression chains ----------------------------------------
    Recipe(
        "crf40",
        "compression",
        ["-c:v", "libx264", "-crf", "40"],
        "Heavy single re-encode, at the edge of what social media applies.",
        "Accuracy drops but the ranking holds; the reliability gate should "
        "lower the visual weight.",
    ),
    Recipe(
        "double_encode",
        "compression",
        ["-c:v", "libx264", "-crf", "32"],
        "Applied twice. Real uploads are re-encoded by every platform they "
        "pass through, so a single encode understates the damage.",
        "Should degrade gracefully, not collapse.",
    ),
    Recipe(
        "low_bitrate",
        "compression",
        ["-c:v", "libx264", "-b:v", "150k", "-maxrate", "150k", "-bufsize", "300k"],
        "Bitrate-starved rather than CRF-targeted; produces blocking "
        "artefacts that can mimic blending boundaries.",
        "A spike in FALSE POSITIVES here is the thing to look for -- "
        "block edges are not blend seams.",
    ),
    # ---- geometry ----------------------------------------------------
    Recipe(
        "letterbox",
        "geometry",
        ["-vf", "scale=iw*0.7:ih*0.7,pad=iw/0.7:ih/0.7:(ow-iw)/2:(oh-ih)/2:black"],
        "Black bars, as added by every 'reposted' video.",
        "Face detection must still find the face; a drop here is a "
        "preprocessing bug, not a detector weakness.",
    ),
    Recipe(
        "offcentre_crop",
        "geometry",
        ["-vf", "crop=iw*0.6:ih*0.6:0:0"],
        "Pushes the face away from centre and may clip it.",
        "Tests whether the model depends on the face being centred.",
    ),
    Recipe(
        "downscale_240p",
        "geometry",
        ["-vf", "scale=-2:240"],
        "Small faces. Measured earlier as the weakest regime.",
        "Expected to be the worst slice; quantify it rather than hide it.",
    ),
    Recipe(
        "hflip",
        "geometry",
        ["-vf", "hflip"],
        "A transformation that changes no forensic evidence at all.",
        "The score must be essentially UNCHANGED. A large swing means the "
        "model learned an orientation artefact, which would be a serious finding.",
    ),
    # ---- temporal ------------------------------------------------------
    Recipe(
        "fps_15",
        "temporal",
        ["-r", "15"],
        "Frame-rate conversion duplicates and drops frames.",
        "Temporal head may degrade; the per-frame scores should not.",
    ),
    Recipe(
        "fps_60_interp",
        "temporal",
        ["-vf", "minterpolate=fps=60:mi_mode=blend"],
        "Blended frame interpolation invents inter-frame content.",
        "A rise in FALSE POSITIVES is plausible -- blended frames look like "
        "blended faces. Worth reporting either way.",
    ),
    # ---- audio ---------------------------------------------------------
    Recipe(
        "audio_strip",
        "audio",
        ["-an"],
        "Removes the audio track entirely.",
        "Must fall back cleanly to visual-only with audio/sync masked off "
        "and the verdict still produced.",
    ),
    Recipe(
        "audio_opus_low",
        "audio",
        ["-c:a", "libopus", "-b:a", "12k"],
        "Aggressive speech codec; destroys the high-frequency detail the audio stream relies on.",
        "The reliability gate should down-weight audio rather than trust it.",
    ),
    Recipe(
        "audio_music_bed",
        "audio",
        ["-af", "volume=0.3,aecho=0.8:0.9:1000:0.3"],
        "Audio present but speech-poor -- the common real-world case the "
        "SNR proxy exists to catch.",
        "audio_snr should fall and the gate should discount the stream.",
    ),
    Recipe(
        "audio_swap",
        "audio",
        [],
        "Real audio from a DIFFERENT real video spliced onto this one. No "
        "generation: both halves are authentic media.",
        "The sync stream SHOULD fire -- this is genuine desynchronisation. "
        "But the video is authentic, so the paper must state whether this "
        "counts as a true positive for 'manipulated' or a false accusation.",
        severity="blocker",
        needs_second_clip=True,
    ),
]


class RedTeamAgent(Agent):
    name = "a8_redteam"
    needs_human_review = True

    def register_tools(self) -> None:
        # Deliberately NO network tool and NO generation tool.
        self.registry.add(
            "list_recipes",
            lambda: [r.to_dict() for r in RECIPES],
            "enumerate stress transformations",
        )

    # ------------------------------------------------------------------
    def analyse(
        self,
        kinds: tuple[str, ...] | None = None,
        source_manifest: str | Path | None = None,  # noqa: ARG002 - reserved for per-dataset recipes
    ) -> AgentResult:
        res = AgentResult(agent=self.name)
        chosen = [r for r in RECIPES if kinds is None or r.kind in kinds]
        res.data["recipes"] = [r.to_dict() for r in chosen]
        res.data["scope"] = (
            "All transformations are lossy operations on existing real media. "
            "No generative model is used and none is available to this agent "
            "."
        )

        findings = [
            Finding(
                severity=r.severity,
                title=f"{r.name} ({r.kind})",
                detail=f"{r.rationale}\n\nExpected: {r.expect}",
                evidence={"ffmpeg_args": r.args, "needs_second_clip": r.needs_second_clip},
                suggested_action=(
                    f"Generate a degraded manifest with "
                    f"`python scripts/degrade_dataset.py --recipe {r.name}`, "
                    f"then evaluate and add the row to the robustness table."
                ),
            )
            for r in chosen
        ]

        findings.append(
            Finding(
                severity="info",
                title="invariance checks are the highest-value subset",
                detail=(
                    "hflip changes no forensic evidence, and audio_strip removes a "
                    "modality the model claims to handle. If either moves the score "
                    "materially, that is a model defect rather than a robustness "
                    "limitation -- and it is cheap to measure."
                ),
                evidence={"invariance_recipes": ["hflip", "audio_strip"]},
                suggested_action="Run these two first; they need no new data and have a clear pass/fail.",
            )
        )

        res.findings = findings
        res.ok = True
        res.summary = (
            f"{len(chosen)} stress recipes across "
            f"{len({r.kind for r in chosen})} categories, all derived from real "
            f"media only. Human review required before any enters training."
        )
        return res

    def narrative_prompt(self, result: AgentResult) -> str | None:
        return (
            "You are red-teaming a deepfake detector. Below are the stress "
            "transformations an automated agent proposes. All operate on real "
            "video only -- no synthetic media is generated.\n\n"
            "Identify the two transformations most likely to produce FALSE "
            "ACCUSATIONS (authentic video scored as manipulated), and explain "
            "the mechanism. False positives matter more than false negatives "
            "here because they defame a real person.\n\n"
            "Do not propose generating deepfakes.\n\n"
            f"```json\n{json.dumps(result.data['recipes'], indent=2)[:6000]}\n```"
        )


def apply_recipe(src: Path, dst: Path, recipe: Recipe, second: Path | None = None) -> bool:
    """Apply one recipe with ffmpeg. Returns True on success.

    Hardened the same way as the serving path: no shell, empty protocol
    whitelist, bounded runtime.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-nostdin", "-y", "-protocol_whitelist", "file", "-loglevel", "error"]

    if recipe.name == "audio_swap":
        if second is None:
            return False
        cmd = [
            *base,
            "-i",
            str(src),
            "-i",
            str(second),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-shortest",
            str(dst),
        ]
    elif recipe.name == "double_encode":
        tmp = dst.with_suffix(".pass1.mp4")
        try:
            subprocess.run(
                [*base, "-i", str(src), *recipe.args, str(tmp)],
                check=True,
                capture_output=True,
                timeout=300,
            )
            subprocess.run(
                [*base, "-i", str(tmp), *recipe.args, str(dst)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        finally:
            tmp.unlink(missing_ok=True)
        return dst.exists()
    else:
        cmd = [*base, "-i", str(src), *recipe.args, str(dst)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return dst.exists()


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        choices=["compression", "geometry", "temporal", "audio", "combined"],
    )
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()

    agent = RedTeamAgent()
    res = agent.run(kinds=tuple(a.kinds) if a.kinds else None)
    print("\n" + res.render())
    if a.write:
        agent.write_proposal(res, "red-team")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
