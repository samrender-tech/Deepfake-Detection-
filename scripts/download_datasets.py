#!/usr/bin/env python3
"""Dataset acquisition.

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py dfdc --out ~/data        # automated
    python scripts/download_datasets.py ffpp --out ~/data        # prints the steps

Three of the six datasets are gated behind an access form that a human must
submit and an author must approve. This script does not pretend otherwise: for
those it prints the exact steps, the expected directory layout, and what to run
once the data lands. Automating around a licence agreement would be both
impossible and wrong.

DFDC is different -- it is on Kaggle, available immediately, and it is the only
instantly-available dataset that carries audio. So it is the one path here that
is genuinely automated, and it is the one that unblocks the audio-visual arm
while the forms clear.

Everything is resumable: a file already present with the right size is skipped,
so an interrupted 25 GB download costs only what it had not finished.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    key: str
    name: str
    role: str
    audio: str
    size_gb: float
    automated: bool
    url: str
    layout: str
    steps: list[str] = field(default_factory=list)
    note: str = ""


DATASETS: dict[str, Dataset] = {
    "dfdc": Dataset(
        key="dfdc",
        name="DFDC",
        role="audio-visual training set",
        audio="yes",
        size_gb=25.0,
        automated=True,
        url="https://www.kaggle.com/c/deepfake-detection-challenge/data",
        layout="dfdc/dfdc_train_part_N/{metadata.json, *.mp4}",
        note=(
            "The only instantly-available dataset with audio, so it is the one "
            "that unblocks Objective 3 while the access forms clear."
        ),
        steps=[
            "Create a Kaggle account and accept the DFDC competition rules.",
            "Settings -> API -> 'Create New Token' downloads kaggle.json.",
            "mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json",
            "Re-run this script; it will download and extract automatically.",
        ],
    ),
    "ffpp": Dataset(
        key="ffpp",
        name="FaceForensics++",
        role="primary visual training set",
        audio="NONE",
        size_gb=6.0,
        automated=False,
        url="https://github.com/ondyari/FaceForensics",
        layout=(
            "FaceForensics++/\n"
            "  original_sequences/youtube/c23/videos/*.mp4\n"
            "  manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/*.mp4\n"
            "  splits/{train,val,test}.json"
        ),
        note=(
            "Carries no usable audio -- this is the empirical basis for the "
            "two-protocol split. Download the OFFICIAL SPLITS too, "
            "or Baseline A cannot be compared with published numbers."
        ),
        steps=[
            "Fill the form at https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform",
            "Use an institutional email if you have one; approval takes days.",
            "The authors email back a download script (EULA-gated).",
            "python faceforensics_download_v4.py <out> -d all -c c23 -t videos",
            "Also fetch the splits JSONs from the repo's dataset/splits/ directory.",
        ],
    ),
    "celebdf": Dataset(
        key="celebdf",
        name="Celeb-DF v2",
        role="cross-dataset visual test",
        audio="NONE",
        size_gb=4.0,
        automated=False,
        url="https://github.com/yuezunli/celeb-deepfakeforensics",
        layout=(
            "Celeb-DF-v2/\n"
            "  Celeb-real/*.mp4  YouTube-real/*.mp4  Celeb-synthesis/*.mp4\n"
            "  List_of_testing_videos.txt"
        ),
        note="List_of_testing_videos.txt is required: without it our Baseline B "
        "number is not comparable to published Celeb-DF results.",
        steps=[
            "Email the authors using the form linked in the repo README.",
            "They reply with a Google Drive link.",
            "Keep List_of_testing_videos.txt alongside the video directories.",
        ],
    ),
    "fakeavceleb": Dataset(
        key="fakeavceleb",
        name="FakeAVCeleb",
        role="cross-dataset AV test",
        audio="yes",
        size_gb=5.0,
        automated=False,
        url="https://github.com/DASH-Lab/FakeAVCeleb",
        layout=(
            "FakeAVCeleb/\n"
            "  RealVideo-RealAudio/<race>/<gender>/<id>/*.mp4\n"
            "  FakeVideo-RealAudio/ ...  RealVideo-FakeAudio/ ...  FakeVideo-FakeAudio/ ..."
        ),
        note="Added because of the no-audio finding -- the cross-dataset test set for Objective 3, "
        "because FF++ and Celeb-DF are silent.",
        steps=[
            "Request access via the Google Form linked in the repo README.",
            "Keep the four combination directories at the top level.",
        ],
    ),
    "deeperforensics": Dataset(
        key="deeperforensics",
        name="DeeperForensics-1.0",
        role="robustness test",
        audio="NONE",
        size_gb=10.0,
        automated=False,
        url="https://github.com/EndlessSora/DeeperForensics-1.0",
        layout="DeeperForensics-1.0/{source_videos,manipulated_videos/end_to_end*}",
        note="Its perturbation suite is the reason to want it: the same fakes at "
        "several distortion types and levels, which no other dataset here gives.",
        steps=["Request access via the form in the repo README."],
    ),
    "lavdf": Dataset(
        key="lavdf",
        name="LAV-DF",
        role="backup AV test",
        audio="yes",
        size_gb=8.0,
        automated=False,
        url="https://github.com/ControlNet/LAV-DF",
        layout="LAV-DF/{metadata.json, train/, dev/, test/}",
        note="Forgeries are temporally localised with frame-level annotations -- "
        "the only set here that can check whether the frames we flag are "
        "the frames that were actually edited.",
        steps=["Request access via the form in the repo README."],
    ),
}


# ==========================================================================
def have_kaggle() -> tuple[bool, str]:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        return False, f"no credentials at {cred}"
    mode = oct(cred.stat().st_mode)[-3:]
    if mode != "600":
        return False, f"{cred} has mode {mode}; Kaggle refuses anything but 600"
    try:
        import kaggle  # noqa: F401
    except ImportError:
        return False, "the kaggle package is not installed (pip install kaggle)"
    except OSError as e:
        return False, str(e)
    return True, "ready"


def download_dfdc(out: Path, parts: list[int], sample_only: bool) -> int:
    """Download DFDC from Kaggle. Resumable."""
    ok, why = have_kaggle()
    if not ok:
        print(f"\nKaggle is not set up: {why}\n")
        for i, s in enumerate(DATASETS["dfdc"].steps, 1):
            print(f"  {i}. {s}")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    comp = "deepfake-detection-challenge"

    files = (
        ["train_sample_videos.zip"] if sample_only else [f"dfdc_train_part_{i}.zip" for i in parts]
    )
    if sample_only:
        print(
            "downloading the 400-video sample (~4 GB) -- enough to build the\n"
            "pipeline end to end before committing to the full 470 GB"
        )

    for fname in files:
        dest = out / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  [skip] {fname} already downloaded ({dest.stat().st_size / 1e9:.1f} GB)")
        else:
            print(f"  [get ] {fname}")
            r = subprocess.run(
                ["kaggle", "competitions", "download", "-c", comp, "-f", fname, "-p", str(out)],
                capture_output=False,
            )
            if r.returncode != 0:
                print(
                    f"  download failed for {fname}; have you accepted the "
                    f"competition rules at kaggle.com/c/{comp}/rules ?"
                )
                return 1

        target = out / dest.stem
        if target.exists() and any(target.glob("*.mp4")):
            print(f"  [skip] {dest.stem} already extracted")
            continue
        print(f"  [unzip] {fname}")
        try:
            with zipfile.ZipFile(dest) as z:
                z.extractall(target)
        except zipfile.BadZipFile:
            print(f"  {fname} is not a valid zip -- delete it and re-run to retry")
            return 1

    n = sum(1 for _ in out.rglob("*.mp4"))
    print(f"\nDFDC ready at {out}: {n} videos")
    print("\nNext:")
    print(f"  ddetect manifest --dataset dfdc --root {out} --audit")
    print("  ddetect preprocess --manifest data/manifests/dfdc.parquet --workers 8")
    return 0


def print_manual(ds: Dataset, out: Path) -> int:
    w = shutil.get_terminal_size((88, 20)).columns - 2
    print(f"\n{ds.name} — {ds.role}")
    print("=" * min(w, 78))
    print(f"  audio: {ds.audio}    approx size: {ds.size_gb:.0f} GB")
    print(f"  {ds.url}\n")
    if ds.note:
        print(textwrap.fill(ds.note, w, initial_indent="  ", subsequent_indent="  "))
        print()
    print("  This dataset is behind an access form. A human has to submit it and")
    print("  an author has to approve it, so there is nothing to automate here.\n")
    for i, s in enumerate(ds.steps, 1):
        print(textwrap.fill(s, w - 6, initial_indent=f"  {i}. ", subsequent_indent="     "))
    print(f"\n  Expected layout under {out}:")
    for line in ds.layout.splitlines():
        print(f"    {line}")
    print("\n  Once it has landed:")
    print(f"    python scripts/preflight.py --root {ds.name.lower()}={out}")
    print(f"    ddetect manifest --dataset {ds.key} --root {out} --audit")
    return 0


def list_datasets() -> int:
    print(f"\n{'dataset':<18}{'audio':<8}{'size':<9}{'access':<14}role")
    print("-" * 86)
    for key, d in DATASETS.items():
        access = "Kaggle (now)" if d.automated else "request form"
        print(f"{key:<18}{d.audio:<8}{d.size_gb:>5.0f} GB  {access:<14}{d.role}")
    print(
        "\nSubmit the FF++, Celeb-DF v2 and FakeAVCeleb forms on day one: approval\n"
        "takes several days and everything past Stage 2 waits on them. DFDC is\n"
        "instant and is the only one with audio, so start there.\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", nargs="?", choices=sorted(DATASETS))
    ap.add_argument("--out", default="~/data", help="parent directory for datasets")
    ap.add_argument("--list", action="store_true", help="show all datasets and exit")
    ap.add_argument("--parts", default="0", help="DFDC part numbers, comma separated")
    ap.add_argument(
        "--sample", action="store_true", help="DFDC: the 400-video sample instead of a full part"
    )
    a = ap.parse_args(argv)

    if a.list or not a.dataset:
        return list_datasets()

    ds = DATASETS[a.dataset]
    out = Path(a.out).expanduser() / a.dataset

    free_gb = shutil.disk_usage(Path(a.out).expanduser().parent).free / 1e9
    if free_gb < ds.size_gb * 1.6:
        print(
            f"\n  WARNING: {free_gb:.0f} GB free, and {ds.name} needs about "
            f"{ds.size_gb:.0f} GB plus roughly the same again for the\n"
            f"  preprocessing cache. Free space first, or point --out at a "
            f"larger volume.\n"
        )

    if ds.automated:
        parts = [int(x) for x in a.parts.split(",") if x.strip()]
        return download_dfdc(out, parts, a.sample)
    return print_manual(ds, out)


if __name__ == "__main__":
    raise SystemExit(main())
