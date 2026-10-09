"""F3 - degradation augmentation.

Project objective: "improve robustness to compression by training with
compression, blur and resolution augmentation". The mechanism argument is that degradation *forces* the model off generator-specific
high-frequency artefacts, which compression destroys anyway, and onto coarse
structural cues that survive a social-media re-encode.

Two kinds:

  online   albumentations, applied per frame in the dataloader.
  offline  a real H.264 re-encode of a whole test set at a fixed CRF, written
           as its own manifest. Online JPEG compression is NOT the same thing:
           it has no inter-frame or motion-compensation artefacts, so claiming
           H.264 robustness from JPEG augmentation alone would be dishonest.
           ``scripts/degrade_dataset.py`` produces the offline grid.

The clip-level consistency rule below matters: a per-frame random quality makes
the *variation* across frames the signal, which is an artefact of our own
pipeline that no real video has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AugConfig:
    enabled: bool = True
    #: Geometric / photometric
    hflip_p: float = 0.5
    rotate_limit: int = 10
    rotate_p: float = 0.3
    color_jitter_p: float = 0.3
    #: Degradation (the Objective-4 group)
    jpeg_p: float = 0.5
    jpeg_quality: tuple[int, int] = (30, 95)
    blur_p: float = 0.3
    blur_limit: tuple[int, int] = (3, 7)
    downscale_p: float = 0.3
    downscale_range: tuple[float, float] = (0.25, 0.75)
    noise_p: float = 0.2
    #: Occlusion: forces the model to spread evidence over the whole face
    #: rather than fixate on one region (a known Xception failure mode).
    cutout_p: float = 0.2
    cutout_holes: int = 3
    cutout_size: float = 0.12
    #: MixStyle-like domain perturbation (Zhou et al. ICLR 2021), applied as a
    #: channel-statistics shuffle. Cheap domain generalisation.
    mixstyle_p: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def assert_albumentations_api() -> None:
    """Verify the kwargs we pass still exist.

    albumentations 2.0 renamed ``quality_lower/upper`` -> ``quality_range``,
    ``scale_min/max`` -> ``scale_range`` and the CoarseDropout arguments, and
    accepted the old names with a *warning* rather than an error. The
    degradation group therefore ran as a no-op while every log looked normal.
    Objective 4 depends on this actually applying, so it is asserted.
    """
    import inspect

    import albumentations as A

    required = {
        A.ImageCompression: {"quality_range"},
        A.Downscale: {"scale_range"},
        A.CoarseDropout: {"num_holes_range", "hole_height_range", "fill"},
        A.GaussNoise: {"std_range"},
    }
    for cls, params in required.items():
        have = set(inspect.signature(cls.__init__).parameters)
        missing = params - have
        if missing:
            raise RuntimeError(
                f"albumentations {A.__version__}: {cls.__name__} no longer accepts "
                f"{sorted(missing)}. The degradation augmentation would silently "
                f"become a no-op. Update build_transform() in ddetect/data/augment.py."
            )


def build_transform(
    cfg: AugConfig, image_size: int, train: bool, include_degradation: bool = True
) -> Any:
    """Build the albumentations pipeline.

    Returns a callable ``t(image=...)`` -> dict. At eval time only the
    deterministic resize+normalise remains, so an evaluation is never
    accidentally stochastic.

    ``include_degradation=False`` drops the compression/blur/downscale group.
    ``ClipAugmentor`` uses that: it applies those ONCE per clip through a
    replayed transform, and a second per-frame application would re-randomise
    them and destroy the clip-level consistency it exists to guarantee.
    """
    import albumentations as A

    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    resize = A.Resize(image_size, image_size)

    if not train or not cfg.enabled:
        return A.Compose([resize, norm])

    degradation = [
        A.ImageCompression(quality_range=cfg.jpeg_quality, p=cfg.jpeg_p),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=cfg.blur_limit),
                A.MotionBlur(blur_limit=cfg.blur_limit[1]),
            ],
            p=cfg.blur_p,
        ),
        A.Downscale(scale_range=cfg.downscale_range, p=cfg.downscale_p),
    ] if include_degradation else []

    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.hflip_p),
            A.Affine(
                translate_percent=(-0.02, 0.02),
                scale=(0.95, 1.05),
                rotate=(-cfg.rotate_limit, cfg.rotate_limit),
                border_mode=0,
                fill=0,
                p=cfg.rotate_p,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.2, 0.2),
                    A.HueSaturationValue(10, 15, 10),
                    A.RandomGamma((80, 120)),
                ],
                p=cfg.color_jitter_p,
            ),
            # --- the degradation group (empty when ClipAugmentor owns it) ---
            *degradation,
            A.GaussNoise(std_range=(0.02, 0.12), p=cfg.noise_p),
            A.CoarseDropout(
                num_holes_range=(1, cfg.cutout_holes),
                hole_height_range=(0.04, cfg.cutout_size),
                hole_width_range=(0.04, cfg.cutout_size),
                fill=0,
                p=cfg.cutout_p,
            ),
            resize,
            norm,
        ]
    )


class ClipAugmentor:
    """Applies ONE sampled degradation consistently across a clip's frames.

    A real video has a single encode: every frame shares its compression level,
    blur and resolution. Sampling per frame would make inter-frame quality
    variance a dataset fingerprint the temporal head would happily latch onto,
    inflating in-dataset accuracy and generalising to nothing.

    Geometric/photometric jitter stays per frame (it is cheap regularisation and
    does not create a cross-frame artefact signature).
    """

    def __init__(self, cfg: AugConfig, image_size: int, train: bool) -> None:
        self.cfg = cfg
        self.image_size = image_size
        self.train = train
        clip_owns_degradation = train and cfg.enabled
        self._frame_t = build_transform(
            cfg, image_size, train, include_degradation=not clip_owns_degradation
        )
        self._clip_t = self._build_clip_level(cfg) if clip_owns_degradation else None

    @staticmethod
    def _build_clip_level(cfg: AugConfig) -> Any:
        import albumentations as A

        # Replay-mode: the first call records the sampled parameters, and every
        # later frame of the same clip replays them identically.
        return A.ReplayCompose(
            [
                A.ImageCompression(quality_range=cfg.jpeg_quality, p=cfg.jpeg_p),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=cfg.blur_limit),
                        A.MotionBlur(blur_limit=cfg.blur_limit[1]),
                    ],
                    p=cfg.blur_p,
                ),
                A.Downscale(scale_range=cfg.downscale_range, p=cfg.downscale_p),
            ]
        )

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """RGB uint8 frames -> normalised float32 CHW arrays."""
        import albumentations as A

        if self._clip_t is not None and frames:
            first = self._clip_t(image=frames[0])
            replay = first["replay"]
            frames = [first["image"]] + [
                A.ReplayCompose.replay(replay, image=f)["image"] for f in frames[1:]
            ]

        out: list[np.ndarray] = []
        for f in frames:
            img = self._frame_t(image=f)["image"]
            out.append(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return out


# --------------------------------------------------------------------------
# audio augmentation
# --------------------------------------------------------------------------
@dataclass
class AudioAugConfig:
    enabled: bool = True
    noise_p: float = 0.3
    snr_db: tuple[float, float] = (10.0, 30.0)
    gain_p: float = 0.3
    gain_db: tuple[float, float] = (-6.0, 6.0)
    #: Mild speed change. Kept small: a large shift would decorrelate the
    #: lips from the speech and manufacture the very desynchronisation the
    #: sync stream is supposed to detect, i.e. label noise.
    speed_p: float = 0.2
    speed_range: tuple[float, float] = (0.95, 1.05)
    codec_p: float = 0.3


def augment_waveform(
    y: np.ndarray, sr: int, cfg: AudioAugConfig, rng: np.random.Generator
) -> np.ndarray:
    """Waveform-domain audio augmentation (F3, audio half)."""
    if not cfg.enabled or y.size == 0:
        return y
    y = y.astype(np.float32, copy=True)

    if rng.random() < cfg.gain_p:
        y *= float(10 ** (rng.uniform(*cfg.gain_db) / 20))

    if rng.random() < cfg.speed_p:
        rate = float(rng.uniform(*cfg.speed_range))
        n = int(y.size / rate)
        y = np.interp(np.linspace(0, y.size - 1, n), np.arange(y.size), y).astype(np.float32)

    if rng.random() < cfg.noise_p:
        snr = float(rng.uniform(*cfg.snr_db))
        sig_p = float(np.mean(y**2)) + 1e-12
        noise_p = sig_p / (10 ** (snr / 10))
        y = y + rng.standard_normal(y.size).astype(np.float32) * np.sqrt(noise_p)

    if rng.random() < cfg.codec_p:
        # Codec proxy: 8-bit mu-law round trip. Cheap stand-in for the
        # quantisation a lossy codec applies. Real Opus/AAC round-trips are
        # done offline by scripts/degrade_dataset.py for the robustness grid.
        mu = 255.0
        comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
        q = np.round(comp * 127) / 127
        y = (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(np.float32)

    return np.clip(y, -1.0, 1.0)
