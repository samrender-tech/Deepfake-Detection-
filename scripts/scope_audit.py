#!/usr/bin/env python3
"""Enforce "we build a detector, not a generator" in CI.

The project's scope commitment is that no deepfakes are created here. That is a
claim about the software, so it is checked by the software: this script fails
the build if a generative-media dependency or import appears anywhere in the
tree.

It is deliberately conservative. Self-Blended Images (F4) and the red-team
agent (A8) manipulate *existing real media* -- image-space blending, codec
round-trips, audio splicing between two real clips. None of that needs a
generative model, so none of these packages should ever be required.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Packages whose presence would mean we can synthesise faces or voices.
BANNED_PACKAGES = {
    "diffusers",
    "stable-diffusion",
    "stablediffusion",
    "insightface",
    "simswap",
    "roop",
    "deepfacelab",
    "deepfacelive",
    "faceswap",
    "facefusion",
    "first-order-model",
    "sadtalker",
    "wav2lip",
    "tortoise-tts",
    "coqui-tts",
    "TTS",
    "bark",
    "rvc",
    "so-vits-svc",
    "openvoice",
    "elevenlabs",
    "real-esrgan",
    "gfpgan",
    "codeformer",
}

#: Import names that would indicate the same thing in source.
BANNED_IMPORTS = {
    "diffusers",
    "insightface",
    "roop",
    "deepfacelab",
    "facefusion",
    "wav2lip",
    "sadtalker",
    "gfpgan",
    "codeformer",
    "elevenlabs",
}

SKIP_DIRS = {".git", ".venv", "node_modules", "web/dist", "runs", "data", "processed", "paper"}


def check_installed() -> list[str]:
    """Scan the installed environment."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    installed = {line.split("==")[0].strip().lower() for line in out.splitlines()}
    return sorted(installed & {b.lower() for b in BANNED_PACKAGES})


def check_declared() -> list[str]:
    """Scan dependency declarations."""
    hits = []
    for name in ("pyproject.toml", "requirements.txt", "web/package.json"):
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text().lower()
        hits += [f"{name}: {b}" for b in BANNED_PACKAGES if b.lower() in txt]
    return sorted(hits)


def check_imports() -> list[str]:
    """Scan source for imports of generative libraries."""
    pat = re.compile(
        r"^\s*(?:import|from)\s+(" + "|".join(re.escape(b) for b in BANNED_IMPORTS) + r")\b",
        re.M,
    )
    hits = []
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or rel.name == Path(__file__).name:
            continue
        for m in pat.finditer(p.read_text()):
            hits.append(f"{rel}: imports {m.group(1)}")
    return sorted(hits)


def main() -> int:
    problems = (
        [f"installed: {h}" for h in check_installed()]
        + [f"declared: {h}" for h in check_declared()]
        + [f"source: {h}" for h in check_imports()]
    )
    if problems:
        print("SCOPE AUDIT FAILED -- this project builds a detector, not a generator.")
        print("Scope commitment: no deepfakes are created here.\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nIf a generative dependency is genuinely needed, that is a change of "
            "project scope and must be discussed and documented, not slipped in."
        )
        return 1
    print("scope audit: clean (no generative-media dependency or import found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
