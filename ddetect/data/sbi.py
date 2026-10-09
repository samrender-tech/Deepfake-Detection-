"""F4 - Self-Blended Images (Shiohara & Yamasaki, CVPR 2022).

The direct answer to the core problem: "detectors learn one forger's fingerprint, not
what a fake actually looks like."

SBI makes a training fake from ONE REAL FRAME. Take two copies of the same
face, perturb one slightly (small affine warp, colour/frequency jitter), then
blend it back through a deformed mask of the face region. The result has a
blending boundary -- the artefact essentially every face-swap pipeline leaves
behind -- and nothing else. Since no generator is ever involved, the model
cannot learn a generator's fingerprint from it, which is precisely the property
the research gap demands.

It also yields the blend mask for free, which supervises the Face X-ray-style
boundary head (F7c) and gives the model a localisation signal rather than a
single global label.

SCOPE: this is image-space self-blending of real video. No
generative model is trained or run; nothing here can synthesise a person
saying something they did not say. ``scripts/scope_audit.py`` enforces that
no generative dependency enters the tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SBIConfig:
    enabled: bool = False
    #: Probability that a REAL training frame is converted into an SBI fake.
    #: 0.5 keeps the batch balanced when SBI is the only fake source.
    p: float = 0.5
    #: Source-copy perturbations. Deliberately mild: the artefact should be the
    #: blend boundary, not a visibly mangled face.
    translate: float = 0.03      # fraction of face width
    scale: tuple[float, float] = (0.95, 1.05)
    rotate_deg: float = 5.0
    #: Photometric difference between the two copies -- what makes the seam
    #: visible at all.
    brightness: float = 0.12
    contrast: float = 0.12
    hue_shift: int = 6
    sharpen_p: float = 0.3
    #: Mask geometry
    mask_blur: tuple[int, int] = (7, 25)
    mask_erode: tuple[int, int] = (2, 12)
    elastic_p: float = 0.5
    #: Downscale-then-upscale the source copy: mimics the resolution mismatch
    #: between a generated face and its host frame, a very common real artefact.
    resolution_mismatch_p: float = 0.4
    resolution_scale: tuple[float, float] = (0.4, 0.9)


# --------------------------------------------------------------------------
def _convex_hull_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A face-shaped mask when no landmarks are available.

    An ellipse over the central face region with a randomly jittered hull.
    Landmark-driven masks are better and are used when the cache has lip
    points, but SBI must work on any cached crop.
    """
    import cv2

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, int(h * 0.52)
    ax = int(w * rng.uniform(0.30, 0.40))
    ay = int(h * rng.uniform(0.38, 0.48))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    # Jitter the hull so the boundary is not a perfect ellipse every time --
    # otherwise the model learns "ellipse edge" instead of "blend seam".
    pts = []
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        r = rng.uniform(0.88, 1.12)
        pts.append([cx + ax * r * np.cos(ang), cy + ay * r * np.sin(ang)])
    hull = cv2.convexHull(np.array(pts, dtype=np.float32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


def _elastic_deform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency elastic warp of the mask boundary."""
    import cv2

    h, w = mask.shape
    grid = 6
    dx = rng.uniform(-1, 1, (grid, grid)).astype(np.float32) * (w * 0.04)
    dy = rng.uniform(-1, 1, (grid, grid)).astype(np.float32) * (h * 0.04)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        mask, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def _perturb_source(img: np.ndarray, cfg: SBIConfig, rng: np.random.Generator) -> np.ndarray:
    """Build the 'source' copy: the same face, slightly different."""
    import cv2

    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # photometric
    out = out * (1 + rng.uniform(-cfg.contrast, cfg.contrast))
    out = out + 255 * rng.uniform(-cfg.brightness, cfg.brightness)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if cfg.hue_shift > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-cfg.hue_shift, cfg.hue_shift + 1)) % 180
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    # resolution mismatch -- the generated-face-is-blurrier artefact
    if rng.random() < cfg.resolution_mismatch_p:
        s = float(rng.uniform(*cfg.resolution_scale))
        small = cv2.resize(out, (max(int(w * s), 8), max(int(h * s), 8)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif rng.random() < cfg.sharpen_p:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, k)

    # geometric
    tx = rng.uniform(-cfg.translate, cfg.translate) * w
    ty = rng.uniform(-cfg.translate, cfg.translate) * h
    ang = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
    sc = float(rng.uniform(*cfg.scale))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def self_blend(
    img: np.ndarray, cfg: SBIConfig | None = None, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Self-blend one RGB uint8 frame.

    Returns ``(blended_rgb_uint8, mask_float32)`` where the mask is the blend
    alpha in [0, 1] -- the Face X-ray target for the boundary head.
    """
    import cv2

    cfg = cfg or SBIConfig()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    source = _perturb_source(img, cfg, rng)

    mask = _convex_hull_mask(h, w, rng)
    if rng.random() < cfg.elastic_p:
        mask = _elastic_deform(mask, rng)

    er = int(rng.integers(cfg.mask_erode[0], cfg.mask_erode[1] + 1))
    if er > 0:
        mask = cv2.erode(mask, np.ones((er, er), np.uint8))

    k = int(rng.integers(cfg.mask_blur[0], cfg.mask_blur[1] + 1)) | 1   # must be odd
    alpha = (cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0)[..., None]

    blended = (source.astype(np.float32) * alpha + img.astype(np.float32) * (1 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8), alpha[..., 0]


def boundary_target(alpha: np.ndarray) -> np.ndarray:
    """Face X-ray target: 4*a*(1-a), peaking where the blend is half-and-half.

    Li et al. (CVPR 2020)'s formulation. It is 0 both inside and outside the
    swapped region and 1 exactly on the seam, so the head is trained to find
    the *boundary* rather than to segment the face -- a face segmenter would
    fire identically on a real face and learn nothing.
    """
    return (4.0 * alpha * (1.0 - alpha)).astype(np.float32)


def self_blend_clip(
    frames: list[np.ndarray], cfg: SBIConfig | None = None, seed: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Self-blend a whole clip with ONE consistent mask geometry.

    A real face swap is applied by the same pipeline on every frame, so the
    mask drifts smoothly rather than being redrawn. Re-sampling per frame would
    create a flickering seam -- a temporal artefact of our own making that the
    temporal head would detect trivially and that would not transfer.
    """
    cfg = cfg or SBIConfig()
    rng = np.random.default_rng(seed)
    base = int(rng.integers(0, 2**31))

    out_f: list[np.ndarray] = []
    out_m: list[np.ndarray] = []
    for i, f in enumerate(frames):
        # Same base seed + slow drift: coherent geometry across the clip.
        b, a = self_blend(f, cfg, seed=base + i // 4)
        out_f.append(b)
        out_m.append(boundary_target(a))
    return out_f, out_m
