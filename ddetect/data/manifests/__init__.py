"""Dataset parser registry.

Adding a forgery dataset means adding a parser here and nothing else: the
cache, sampler, models, metrics and serving path all read the manifest
contract. This is what makes the pipeline dataset-agnostic.
"""

from __future__ import annotations

from ddetect.data.manifests.base import DatasetParser, split_by_identity
from ddetect.data.manifests.celebdf import CelebDFParser
from ddetect.data.manifests.deeperforensics import DeeperForensicsParser
from ddetect.data.manifests.dfdc import DFDCParser
from ddetect.data.manifests.fakeavceleb import FakeAVCelebParser
from ddetect.data.manifests.ffpp import FFPPParser
from ddetect.data.manifests.lavdf import LAVDFParser
from ddetect.data.manifests.fixture import FixtureParser

PARSERS: dict[str, type[DatasetParser]] = {
    "ffpp": FFPPParser,
    "celebdf": CelebDFParser,
    "dfdc": DFDCParser,
    "fakeavceleb": FakeAVCelebParser,
    "deeperforensics": DeeperForensicsParser,
    "lavdf": LAVDFParser,
    "fixture": FixtureParser,
}

__all__ = [
    "PARSERS", "DatasetParser", "split_by_identity",
    "FFPPParser", "CelebDFParser", "DFDCParser", "FakeAVCelebParser",
    "DeeperForensicsParser", "LAVDFParser", "FixtureParser",
]


def get_parser(name: str, root: str, **kw: object) -> DatasetParser:
    if name not in PARSERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(PARSERS)}")
    return PARSERS[name](root, **kw)  # type: ignore[arg-type]
