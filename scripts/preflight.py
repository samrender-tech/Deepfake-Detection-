#!/usr/bin/env python3
"""Is this machine ready to run a real experiment?

Checks the things that waste the most time when they are wrong, in the order
they bite:

* a training run that dies at epoch 9 because the disk filled,
* an afternoon of preprocessing on CPU because MPS/CUDA was not picked up,
* numbers that cannot be reported because they came from a non-CUDA device,
* a dataset whose manifest exists but whose videos do not,
* a dirty working tree, which makes any result unreproducible.

Exits non-zero on a blocker. Warnings do not fail: "no GPU" is fine for
authoring and fatal only for a reportable run, and only this script knows
which you are about to do.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Rough per-dataset cache cost. Measured on FF++ c23 at 32 frames/video:
#: ~250 KB of JPEG + wav per video. Scaled by video count.
CACHE_GB = {
    "ffpp": 6.0,
    "celebdf": 4.0,
    "dfdc": 25.0,
    "fakeavceleb": 5.0,
    "deeperforensics": 10.0,
    "lavdf": 8.0,
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocker: bool = False
    fix: str = ""


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            cmd, text=True, stderr=subprocess.DEVNULL, timeout=20
        ).strip()
    except Exception:
        return None


# ==========================================================================
def check_binaries() -> list[Check]:
    out = []
    for name, cmd, blocker, fix in (
        ("ffmpeg", ["ffmpeg", "-version"], True, "brew install ffmpeg  /  apt install ffmpeg"),
        ("ffprobe", ["ffprobe", "-version"], True, "ships with ffmpeg"),
        ("git", ["git", "--version"], False, "needed to stamp runs with a commit"),
    ):
        path = shutil.which(name)
        ver = (_run(cmd) or "").splitlines()[:1]
        out.append(Check(name, bool(path), (ver[0][:60] if ver else "not found"), blocker, fix))
    return out


def check_python_deps() -> list[Check]:
    out = []
    required = [
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
        "numpy",
        "pandas",
        "cv2",
        "albumentations",
        "librosa",
        "soundfile",
        "pyarrow",
        "sklearn",
    ]
    optional = {
        "facenet_pytorch": "face detection quality (install the 'face' extra)",
        "mediapipe": "lip landmarks for the mouth crop ('face' extra)",
        "transformers": "the WavLM audio tier ('audio' extra)",
        "fastapi": "the serving layer ('serve' extra)",
        "onnxruntime": "ONNX export parity checks ('export' extra)",
        "mlflow": "experiment tracking ('track' extra)",
    }
    missing = []
    for m in required:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    out.append(
        Check(
            "core dependencies",
            not missing,
            "all present" if not missing else f"missing {missing}",
            blocker=True,
            fix='pip install -e ".[dev,face]"',
        )
    )

    for m, why in optional.items():
        try:
            __import__(m)
            out.append(Check(f"optional: {m}", True, "present"))
        except ImportError:
            out.append(Check(f"optional: {m}", False, f"absent -- {why}"))
    return out


def check_device() -> list[Check]:
    from ddetect.utils.device import device_report, pick_device

    rep = device_report(pick_device())
    dev = rep["device"]
    authoritative = bool(rep.get("authoritative"))
    detail = f"{dev}" + (
        f" ({rep.get('gpu_name')}, {rep.get('vram_gb')} GB)" if dev == "cuda" else ""
    )
    checks = [Check("compute device", True, detail)]
    checks.append(
        Check(
            "authoritative for reported numbers",
            authoritative,
            "CUDA"
            if authoritative
            else f"{dev} is for authoring and smoke tests only; reported numbers must "
            f"come from CUDA",
            blocker=False,
            fix="run the experiment grid on Kaggle/Colab/a lab GPU",
        )
    )
    if dev == "cuda":
        vram = float(rep.get("vram_gb", 0))
        checks.append(
            Check(
                "VRAM",
                vram >= 10,
                f"{vram} GB"
                + (
                    ""
                    if vram >= 10
                    else " -- EfficientNet-B4 at batch 16 needs ~10 GB; "
                    "lower batch_size or n_frames"
                ),
            )
        )
    return checks


def check_disk(datasets: list[str], cache_root: Path) -> list[Check]:
    free_gb = shutil.disk_usage(cache_root if cache_root.exists() else ROOT).free / 1e9
    needed = sum(CACHE_GB.get(d, 5.0) for d in datasets) if datasets else 10.0
    # Checkpoints and runs add up faster than people expect: ~0.5 GB per run
    # across 32 runs in the grid.
    needed += 16.0
    return [
        Check(
            "free disk",
            free_gb > needed,
            f"{free_gb:.0f} GB free, ~{needed:.0f} GB needed for "
            f"{datasets or 'a typical run'} (cache + checkpoints)",
            blocker=free_gb < needed * 0.5,
            fix="free space, or point --cache-root at a larger volume",
        )
    ]


def check_datasets(roots: dict[str, Path]) -> list[Check]:
    out = []
    for name, root in roots.items():
        if not root.exists():
            out.append(
                Check(
                    f"dataset: {name}",
                    False,
                    f"not found at {root}",
                    fix=f"see docs/DATASET_CARD.md for {name} access",
                )
            )
            continue
        n = sum(1 for _ in root.rglob("*.mp4"))
        out.append(
            Check(
                f"dataset: {name}",
                n > 0,
                f"{n} mp4 files under {root}",
                fix="" if n else "the directory exists but holds no video",
            )
        )
    return out


def check_manifests_and_cache(cache_root: Path) -> list[Check]:
    out = []
    man_dir = ROOT / "data" / "manifests"
    mans = sorted(p for p in man_dir.glob("*.parquet")) if man_dir.exists() else []
    real = [m for m in mans if "fixture" not in m.name]
    out.append(
        Check(
            "manifests",
            bool(real),
            f"{[m.stem for m in real]}" if real else "only the synthetic fixture manifest exists",
            fix="ddetect manifest --dataset <name> --root <path>",
        )
    )

    for m in real:
        try:
            from ddetect.data.manifest_io import read_manifest
            from ddetect.data.preprocess import cache_dir_for, is_cached

            df = read_manifest(m)
            cached = sum(
                is_cached(cache_dir_for(cache_root, r.dataset, r.video_id))
                for r in df.head(200).itertuples()
            )
            frac = cached / min(len(df), 200)
            out.append(
                Check(
                    f"cache: {m.stem}",
                    frac > 0.95,
                    f"{frac:.0%} of a 200-video sample is preprocessed",
                    fix="ddetect preprocess --manifest " + str(m),
                )
            )
            missing = sum(1 for r in df.head(200).itertuples() if not Path(r.path).exists())
            if missing:
                out.append(
                    Check(
                        f"files: {m.stem}",
                        False,
                        f"{missing}/200 sampled paths do not exist on disk",
                        blocker=True,
                        fix="the manifest points at a moved or deleted dataset; rebuild it",
                    )
                )
        except Exception as e:
            out.append(Check(f"cache: {m.stem}", False, f"could not read: {e}"))
    return out


def check_repro() -> list[Check]:
    from ddetect.utils.log import git_sha, is_tree_dirty

    sha = git_sha()
    dirty = is_tree_dirty()
    return [
        Check(
            "working tree",
            not dirty,
            f"git {sha}" + (" -- UNCOMMITTED CHANGES" if dirty else ""),
            blocker=False,
            fix="commit before a reportable run; `make eval-final` refuses a dirty tree",
        )
    ]


def check_leakage_gate() -> list[Check]:
    """The gate that protects the headline number."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_leakage.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [
        Check(
            "leakage gate",
            r.returncode == 0,
            "passing" if r.returncode == 0 else "FAILING",
            blocker=r.returncode != 0,
            fix="fix the splits before training; a leak voids the headline result",
        )
    ]


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="dataset names you intend to use (for the disk estimate)",
    )
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="dataset root to verify, repeatable",
    )
    ap.add_argument("--cache-root", default="processed")
    ap.add_argument(
        "--skip-tests", action="store_true", help="skip the leakage gate (it takes ~20s)"
    )
    a = ap.parse_args(argv)

    roots = {}
    for spec in a.root:
        if "=" in spec:
            k, v = spec.split("=", 1)
            roots[k] = Path(v).expanduser()

    checks: list[Check] = []
    checks += check_binaries()
    checks += check_python_deps()
    checks += check_device()
    checks += check_disk(a.datasets, Path(a.cache_root))
    if roots:
        checks += check_datasets(roots)
    checks += check_manifests_and_cache(Path(a.cache_root))
    checks += check_repro()
    if not a.skip_tests:
        checks += check_leakage_gate()

    print("\nPREFLIGHT\n" + "=" * 78)
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.blocker else "warn")
        print(f"  [{mark}] {c.name:34s} {c.detail}")
        if not c.ok and c.fix:
            print(f"         -> {c.fix}")

    blockers = [c for c in checks if not c.ok and c.blocker]
    warns = [c for c in checks if not c.ok and not c.blocker]
    print("=" * 78)
    if blockers:
        print(f"{len(blockers)} blocker(s). Fix these before running an experiment.")
        return 1
    print(f"ready; {len(warns)} warning(s).")
    if any("authoritative" in c.name for c in warns):
        print(
            "\n  NOTE: this device is not CUDA. Use it for authoring and smoke\n"
            "  tests; run the experiment grid somewhere with a GPU, or the\n"
            "  numbers cannot go in the paper."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
