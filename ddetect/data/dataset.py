"""F6 - Dataset and samplers. Emits the frozen batch dict (contract 5.3).

Reads only the preprocessing cache, never the source videos, so an epoch is
JPEG decodes rather than H.264 seeks. Missing audio is normal, not an error:
``has_audio=False`` masks the audio and sync streams off, which is what lets
one checkpoint serve silent FF++/Celeb-DF and audio-bearing DFDC/FakeAVCeleb.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from ddetect.contracts import RELIABILITY_FIELDS
from ddetect.data.augment import AudioAugConfig, AugConfig, ClipAugmentor, augment_waveform
from ddetect.data.preprocess import AUDIO_SR, MOUTH_SIZE, cache_dir_for, is_cached
from ddetect.data.sbi import SBIConfig, boundary_target, self_blend
from ddetect.utils.log import get_logger
from ddetect.utils.seed import frame_sample_seed

log = get_logger(__name__)

MEL_BINS = 80
SYNC_WINDOW_FRAMES = 16          # SyncNet's temporal receptive field
SYNC_WINDOW_SECONDS = 0.64       # 16 frames at 25 fps


@dataclass
class DataConfig:
    manifest: str = "data/manifests/fixture.parquet"
    cache_root: str = "processed"
    split: str = "train"
    n_frames: int = 32
    image_size: int = 299
    # audio
    use_audio: bool = True
    use_sync: bool = True
    mel_bins: int = MEL_BINS
    audio_seconds: float = 4.0
    max_sync_windows: int = 6
    # augmentation
    aug: AugConfig = field(default_factory=AugConfig)
    audio_aug: AudioAugConfig = field(default_factory=AudioAugConfig)
    sbi: SBIConfig = field(default_factory=SBIConfig)
    # behaviour
    drop_faceless: bool = True
    require_cache: bool = True


class DeepfakeClipDataset(Dataset):
    """One item = one video, as the batch-dict contract describes.

    ``__getitem__`` never raises on bad data: a missing cache entry or an
    unreadable frame yields a zero clip with ``reliab`` driven to 0 so the
    reliability gate discounts it. A dataloader that dies 3 hours into an epoch
    because one of 100k videos is corrupt is not usable.
    """

    def __init__(self, cfg: DataConfig, df: pd.DataFrame | None = None) -> None:
        self.cfg = cfg
        from ddetect.data.manifest_io import read_manifest

        self.df = (
            df if df is not None
            else read_manifest(cfg.manifest, split=cfg.split)
        ).reset_index(drop=True)

        if cfg.require_cache:
            before = len(self.df)
            keep = [
                is_cached(cache_dir_for(cfg.cache_root, r.dataset, r.video_id))
                for r in self.df.itertuples()
            ]
            self.df = self.df[keep].reset_index(drop=True)
            if len(self.df) < before:
                log.warning(
                    "%s split: %d/%d videos have no cache entry and were dropped. "
                    "Run: python -m ddetect.data.run_preprocess --manifest %s",
                    cfg.split, before - len(self.df), before, cfg.manifest,
                )
        if len(self.df) == 0:
            raise RuntimeError(
                f"dataset is empty for split={cfg.split!r} of {cfg.manifest}"
            )

        self.train = cfg.split == "train"
        self.augmentor = ClipAugmentor(cfg.aug, cfg.image_size, self.train)
        self._labels = self.df.label.to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    # -- helpers -----------------------------------------------------------
    def _cache(self, row: Any) -> Path:
        return cache_dir_for(self.cfg.cache_root, row.dataset, row.video_id)

    def _load_meta(self, cdir: Path) -> dict:
        try:
            return json.loads((cdir / "meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_faces(self, cdir: Path, n: int) -> tuple[list[np.ndarray], list[int]]:
        import cv2

        paths = sorted((cdir / "faces").glob("*.jpg"))
        if not paths:
            return [], []
        # Evenly subsample/repeat to exactly n frames so the batch is rectangular.
        idx = np.linspace(0, len(paths) - 1, num=n, dtype=int)
        frames, used = [], []
        for i in idx:
            img = cv2.imread(str(paths[i]))
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            used.append(int(i))
        return frames, used

    def _load_mouth_windows(self, cdir: Path, n_win: int) -> np.ndarray | None:
        """-> (n_win, 16, 1, 96, 96) or None."""
        import cv2

        paths = sorted((cdir / "mouth").glob("*.jpg"))
        if len(paths) < SYNC_WINDOW_FRAMES // 2:
            return None

        imgs = []
        for p in paths:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (MOUTH_SIZE, MOUTH_SIZE):
                m = cv2.resize(m, (MOUTH_SIZE, MOUTH_SIZE))
            imgs.append(m)
        if not imgs:
            return None

        arr = np.stack(imgs).astype(np.float32) / 255.0      # (K, 96, 96)
        # Tile into overlapping 16-frame windows, padding by edge repetition.
        if arr.shape[0] < SYNC_WINDOW_FRAMES:
            pad = SYNC_WINDOW_FRAMES - arr.shape[0]
            arr = np.concatenate([arr, np.repeat(arr[-1:], pad, axis=0)], axis=0)

        starts = np.linspace(
            0, max(arr.shape[0] - SYNC_WINDOW_FRAMES, 0), num=n_win, dtype=int
        )
        wins = np.stack([arr[s : s + SYNC_WINDOW_FRAMES] for s in starts])  # (n_win,16,96,96)
        return wins[:, :, None, :, :]                                        # add channel

    def _load_audio(self, cdir: Path, seed: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """-> (mel (1,80,L), waveform (N,)) or (None, None)."""
        wav_p = cdir / "audio.wav"
        if not wav_p.exists():
            return None, None
        try:
            import soundfile as sf

            y, sr = sf.read(str(wav_p), dtype="float32")
        except Exception:  # noqa: BLE001
            return None, None
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size == 0:
            return None, None

        want = int(self.cfg.audio_seconds * AUDIO_SR)
        if y.size < want:
            y = np.pad(y, (0, want - y.size))
        else:
            # Centre crop at eval for determinism; random crop while training.
            if self.train:
                rng = np.random.default_rng(seed)
                off = int(rng.integers(0, y.size - want + 1))
            else:
                off = (y.size - want) // 2
            y = y[off : off + want]

        if self.train and self.cfg.audio_aug.enabled:
            y = augment_waveform(y, AUDIO_SR, self.cfg.audio_aug, np.random.default_rng(seed))
            if y.size < want:
                y = np.pad(y, (0, want - y.size))
            y = y[:want]

        import librosa

        mel = librosa.feature.melspectrogram(
            y=y, sr=AUDIO_SR, n_fft=400, hop_length=160, n_mels=self.cfg.mel_bins
        )
        logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        # Per-utterance standardisation: absolute loudness is a recording
        # artefact, not forgery evidence, and datasets differ wildly in level.
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-5)
        return logmel[None], y.astype(np.float32)

    # -- the contract ------------------------------------------------------
    def __getitem__(self, i: int) -> dict[str, Any]:
        cfg = self.cfg
        row = self.df.iloc[i]
        cdir = self._cache(row)
        seed = frame_sample_seed(str(row.video_id))
        meta = self._load_meta(cdir)

        label = float(row.label)
        frames, _ = self._load_faces(cdir, cfg.n_frames)

        # ---- F4: SBI turns a REAL clip into a fake with a known mask ----
        mask_target: np.ndarray | None = None
        rng = np.random.default_rng(seed)
        if (
            self.train and cfg.sbi.enabled and label == 0.0
            and frames and rng.random() < cfg.sbi.p
        ):
            blended, masks = [], []
            base = int(rng.integers(0, 2**31))
            for k, f in enumerate(frames):
                b, a = self_blend(f, cfg.sbi, seed=base + k // 4)
                blended.append(b)
                masks.append(boundary_target(a))
            frames = blended
            mask_target = np.stack(masks)
            label = 1.0          # it is now a fake, with boundary supervision

        if not frames:
            # An empty clip means the cache is missing for this video. With
            # require_cache=True these rows are dropped up front; reaching here
            # means someone bypassed that, and a silently-zero clip would train
            # or score on nothing. Warn every time -- it is never normal.
            log.warning(
                "no cached frames for %s at %s; returning a ZERO clip. "
                "Run ddetect.data.run_preprocess for this manifest.",
                row.video_id, cdir,
            )
            faces = torch.zeros(cfg.n_frames, 3, cfg.image_size, cfg.image_size)
            reliab = torch.zeros(len(RELIABILITY_FIELDS))
        else:
            aug = self.augmentor(frames)
            if len(aug) < cfg.n_frames:
                aug = aug + [aug[-1]] * (cfg.n_frames - len(aug))
            faces = torch.from_numpy(np.stack(aug[: cfg.n_frames]))
            rel = meta.get("reliability", {})
            reliab = torch.tensor(
                [float(rel.get(k, 0.0)) for k in RELIABILITY_FIELDS], dtype=torch.float32
            )

        item: dict[str, Any] = {
            "faces": faces,
            "label": torch.tensor(label, dtype=torch.float32),
            "reliab": reliab,
            "video_id": str(row.video_id),
            "dataset": str(row.dataset),
            "forgery_method": str(row.forgery_method),
        }

        # ---- mask target for the boundary head (F7c) --------------------
        if mask_target is not None:
            import cv2

            small = np.stack(
                [cv2.resize(m, (cfg.image_size // 8, cfg.image_size // 8)) for m in mask_target]
            )
            if small.shape[0] < cfg.n_frames:
                small = np.concatenate(
                    [small, np.repeat(small[-1:], cfg.n_frames - small.shape[0], 0)]
                )
            item["mask"] = torch.from_numpy(small[: cfg.n_frames, None])
        else:
            # NaN, per the contract: "SBI blend mask, NaN if n/a". The loss
            # masks these out rather than training the head toward zero, which
            # would teach it that real faces have no boundary anywhere -- true,
            # but it would also swamp the positive signal.
            s = cfg.image_size // 8
            item["mask"] = torch.full((cfg.n_frames, 1, s, s), float("nan"))

        # ---- audio + sync ------------------------------------------------
        has_audio = bool(row.has_audio) and cfg.use_audio
        mel, wav = (self._load_audio(cdir, seed) if has_audio else (None, None))
        if mel is None:
            has_audio = False
            n_mel_frames = int(cfg.audio_seconds * AUDIO_SR / 160) + 1
            mel = np.zeros((1, cfg.mel_bins, n_mel_frames), dtype=np.float32)
            wav = np.zeros(int(cfg.audio_seconds * AUDIO_SR), dtype=np.float32)
            item["reliab"] = item["reliab"].clone()
            item["reliab"][0] = 0.0          # audio_snr -> 0

        item["mel"] = torch.from_numpy(mel)
        item["wav"] = torch.from_numpy(wav)
        item["has_audio"] = torch.tensor(has_audio)

        mouth = (
            self._load_mouth_windows(cdir, cfg.max_sync_windows)
            if (cfg.use_sync and has_audio and meta.get("mouth_available", False))
            else None
        )
        if mouth is None:
            item["mouth"] = torch.zeros(
                cfg.max_sync_windows, SYNC_WINDOW_FRAMES, 1, MOUTH_SIZE, MOUTH_SIZE
            )
            item["reliab"] = item["reliab"].clone()
            item["reliab"][2] = 0.0          # mouth_vis -> 0
        else:
            item["mouth"] = torch.from_numpy(mouth.astype(np.float32))

        return item


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------
def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack into the batch dict. Meta keys stay python lists."""
    from ddetect.contracts import BATCH_META_KEYS

    out: dict[str, Any] = {}
    for k in batch[0]:
        if k in BATCH_META_KEYS:
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# --------------------------------------------------------------------------
# samplers
# --------------------------------------------------------------------------
class BalancedMethodSampler(Sampler[int]):
    """Balances real/fake AND the forgery methods within the fake half.

    FF++ has 1 real video per 4 fakes, and an unbalanced sampler makes the
    model method-specialised -- it over-fits whichever generator is most
    numerous, which is the exact failure mode the project exists to measure.
    Equalising per method also keeps the per-method breakdown
    interpretable.
    """

    def __init__(self, df: pd.DataFrame, n_samples: int | None = None, seed: int = 0) -> None:
        self.df = df.reset_index(drop=True)
        self.seed = seed
        self.n_samples = n_samples or len(df)
        self.epoch = 0

        reals = self.df.index[self.df.label == 0].to_numpy()
        fakes = self.df[self.df.label == 1]
        self._real_pool = reals
        self._fake_pools = [
            g.index.to_numpy() for _, g in fakes.groupby("forgery_method")
        ]
        if len(self._real_pool) == 0 or not self._fake_pools:
            log.warning(
                "BalancedMethodSampler: one class is empty (real=%d, methods=%d); "
                "falling back to uniform sampling",
                len(self._real_pool), len(self._fake_pools),
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(self.seed + self.epoch)
        if len(self._real_pool) == 0 or not self._fake_pools:
            yield from rng.permutation(len(self.df))[: self.n_samples].tolist()
            return

        half = self.n_samples // 2
        idx = list(rng.choice(self._real_pool, size=half, replace=True))
        per = max(half // len(self._fake_pools), 1)
        for pool in self._fake_pools:
            idx.extend(rng.choice(pool, size=per, replace=True).tolist())
        idx = idx[: self.n_samples]
        rng.shuffle(idx)
        yield from (int(i) for i in idx)


class AudioStratifiedSampler(Sampler[int]):
    """Guarantees every batch contains some audio-bearing videos.

    With a mixed manifest (FF++ silent + DFDC with audio), random batching
    produces all-silent batches where the audio and sync branches get no
    gradient at all, and the fusion head learns to ignore them. This interleaves
    so each batch has at least ``min_audio_frac`` audio items.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        min_audio_frac: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.k = max(int(batch_size * min_audio_frac), 1)
        self.seed = seed
        self.epoch = 0
        self._with = self.df.index[self.df.has_audio].to_numpy()
        self._without = self.df.index[~self.df.has_audio].to_numpy()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(self.seed + self.epoch)
        if len(self._with) == 0 or len(self._without) == 0:
            yield from rng.permutation(len(self.df)).tolist()
            return

        n_batches = len(self.df) // self.batch_size
        order: list[int] = []
        for _ in range(n_batches):
            a = rng.choice(self._with, size=self.k, replace=len(self._with) < self.k)
            b = rng.choice(self._without, size=self.batch_size - self.k, replace=True)
            batch = np.concatenate([a, b])
            rng.shuffle(batch)
            order.extend(int(x) for x in batch)
        yield from order
