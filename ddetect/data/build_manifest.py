"""CLI: build a dataset manifest.

    python -m ddetect.data.build_manifest --dataset fixture \
        --root tests/fixtures/videos --out data/manifests/fixture.parquet

``--audit`` additionally runs the integrity sweep (F5) and prints the audio
evidence table that settles the no-audio finding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ddetect.data.integrity import audit
from ddetect.data.manifest_io import read_manifest, summarise, write_manifest
from ddetect.data.manifests import PARSERS, get_parser
from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(PARSERS))
    ap.add_argument("--root", required=True, help="dataset root directory")
    ap.add_argument("--out", default=None, help="output parquet (default data/manifests/<ds>.parquet)")
    ap.add_argument("--compression", default="c23", help="FF++ only: raw|c23|c40")
    ap.add_argument("--parts", default=None, help="DFDC only: comma-separated part numbers")
    ap.add_argument("--test-only", action="store_true", help="force every row to split=test")
    ap.add_argument("--limit", type=int, default=None, help="probe at most N videos (smoke tests)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--audit", action="store_true", help="run the F5 integrity sweep")
    a = ap.parse_args(argv)
    setup_logging()

    out = Path(a.out or f"data/manifests/{a.dataset}.parquet")

    kw: dict[str, object] = {}
    if a.dataset == "ffpp":
        kw["compression"] = a.compression
    if a.dataset == "dfdc" and a.parts:
        kw["parts"] = tuple(int(x) for x in a.parts.split(","))
    if a.dataset in ("celebdf", "fakeavceleb") and a.test_only:
        kw["test_only"] = True

    parser = get_parser(a.dataset, a.root, **kw)
    log.info("parser=%s  expected audio: %s", parser.name, parser.expected_audio)

    rows = parser.build(workers=a.workers, limit=a.limit)
    write_manifest(rows, out)

    df = read_manifest(out)
    print("\n" + summarise(df))

    # --- the audio evidence table -----------------------------
    n_audio = int(df.has_audio.sum())
    print(
        f"\nAUDIO AUDIT  {a.dataset}: {n_audio}/{len(df)} videos carry a usable "
        f"audio stream ({100 * n_audio / max(len(df), 1):.1f}%)"
    )
    print(f"  parser expectation: {parser.expected_audio}")
    if n_audio == 0:
        print(
            "  => CONFIRMED silent. The audio-visual objective cannot be\n"
            "     evaluated on this dataset; use DFDC -> FakeAVCeleb."
        )
    elif n_audio < len(df):
        print("  => mixed; the has_audio mask and modality dropout handle this.")

    if a.audit:
        print("\n" + audit(df).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
