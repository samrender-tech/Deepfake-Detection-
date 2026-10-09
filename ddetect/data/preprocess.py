"""F2 - the preprocessing cache.

One function, ``preprocess_video``, is called by BOTH training and serving.
That is deliberate and load-bearing: it is what makes the demo incapable of
silently drifting from the reported numbers, and what ``test_detector_parity``
asserts.

Cache layout per video:

    processed/{dataset}/{video_id}/
        faces/000.jpg ... 031.jpg    face crops, 320px, JPEG q95
        mouth/000.jpg ...            96x96 grey mouth crops
        audio.wav                    16 kHz mono
        meta.json                    bboxes, frame indices, reliability, versions

Resumable and idempotent: a complete directory is skipped unless the
``cache_schema_version`` in ``meta.json`` is stale.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ddetect.contracts import CACHE_SCHEMA_VERSION, RELIABILITY_FIELDS
from ddetect.data.facedet import (
    Detection,
    get_detector,
    pick_main_track,
    warn_if_fallback,
)
from ddetect.utils.log import get_logger
from ddetect.utils.seed import frame_sample_seed

log = get_logger(__name__)

FFMPEG = "ffmpeg"

# --- cache geometry. Changing any of these must bump CACHE_SCHEMA_VERSION. ---
STORE_SIZE = 320          # face crops stored at 320px, resized at load time so a
                          # backbone change (299 Xception / 224 EffNet) needs no re-run
MOUTH_SIZE = 96           # SyncNet's input
FACE_MARGIN = 1.3         # crop 1.3x the detected box
AUDIO_SR = 16_000
JPEG_QUALITY = 95


@dataclass
class PreprocessConfig:
    n_frames: int = 32
    sampling: str = "uniform"          # uniform | dense | random
    detector: str = "auto"
    device: str = "cpu"
    #: MTCNN cascade thresholds. The synthetic fixtures need permissive values
    #: (see facedet.PERMISSIVE_MTCNN_THRESHOLDS); real data uses the defaults.
    det_thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7)
    extract_audio: bool = True
    extract_mouth: bool = True
    min_face_px: int = 48
    detect_every: int = 1              # detect on every Nth sampled frame, interpolate between
    overwrite: bool = False


@dataclass
class VideoMeta:
    """What lands in ``meta.json``. Also returned in-memory for serving."""

    cache_schema_version: int = CACHE_SCHEMA_VERSION
    video_id: str = ""
    dataset: str = ""
    source_path: str = ""
    detector: str = ""
    fps: float = 0.0
    n_frames_total: int = 0
    frame_indices: list[int] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    bboxes: list[list[float] | None] = field(default_factory=list)
    det_confidences: list[float] = field(default_factory=list)
    n_faces_found: int = 0
    has_audio: bool = False
    audio_duration_s: float = 0.0
    mouth_available: bool = False
    reliability: dict[str, float] = field(default_factory=dict)
    phash: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# frame sampling
# --------------------------------------------------------------------------
def sample_indices(
    n_total: int, n_want: int, strategy: str = "uniform", seed: int = 0
) -> list[int]:
    """Choose which frames to read.

    Deterministic by design: ``random`` still derives from a per-video seed
    (``frame_sample_seed``) so training, evaluation and serving all land on the
    same frames for the same video.
    """
    n_total = max(int(n_total), 1)
    if n_total <= n_want:
        return list(range(n_total))

    if strategy == "uniform":
        return np.linspace(0, n_total - 1, num=n_want, dtype=int).tolist()
    if strategy == "dense":
        # A contiguous window from the middle: preserves temporal continuity,
        # which the sync stream needs.
        start = max((n_total - n_want) // 2, 0)
        return list(range(start, start + n_want))
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n_total, size=n_want, replace=False).tolist())
    raise ValueError(f"unknown sampling strategy {strategy!r}")


def read_frames(video_path: str | Path, indices: list[int]) -> tuple[list[np.ndarray], float, int]:
    """Decode specific frames as RGB uint8.

    Sequential decode with a skip, not ``CAP_PROP_POS_FRAMES`` per frame: seeking
    is unreliable on the variable-GOP H.264 in DFDC and silently returns the
    wrong frame, which would desynchronise the mouth crops from the audio.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    wanted = set(int(i) for i in indices)
    frames: dict[int, np.ndarray] = {}
    i = 0
    last_wanted = max(wanted) if wanted else -1
    while i <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

    ordered = [frames[i] for i in sorted(frames)]
    if not ordered:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return ordered, fps, total or i


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------
def crop_face(
    frame: np.ndarray, det: Detection | None, size: int = STORE_SIZE, margin: float = FACE_MARGIN
) -> np.ndarray:
    """Square crop around a detection, reflect-padded at the frame edge.

    Reflect rather than zero padding: a black bar is itself a strong artefact
    and the model would happily learn "black bar => this dataset => this label".
    """
    import cv2

    h, w = frame.shape[:2]
    if det is None:
        # No face: centre crop. Flagged in meta so the sample can be dropped.
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        patch = frame[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    x1, y1, x2, y2 = det.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    half = side / 2

    px1, py1 = int(round(cx - half)), int(round(cy - half))
    px2, py2 = int(round(cx + half)), int(round(cy + half))

    pad_l, pad_t = max(0, -px1), max(0, -py1)
    pad_r, pad_b = max(0, px2 - w), max(0, py2 - h)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)

    patch = frame[py1:py2, px1:px2]
    if patch.size == 0:
        side_i = min(h, w)
        patch = frame[: side_i, : side_i]
    elif any((pad_l, pad_t, pad_r, pad_b)):
        patch = cv2.copyMakeBorder(
            patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
        )
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def crop_mouth(
    frame: np.ndarray, det: Detection | None, size: int = MOUTH_SIZE
) -> np.ndarray | None:
    """96x96 grey mouth crop.

    Uses the lip landmark ring when the detector provided one; otherwise falls
    back to the lower-middle third of the face box, which is where the mouth is
    for a roughly frontal face. The fallback is recorded as lower
    ``mouth_vis`` reliability so the gate (F10) can discount it.
    """
    import cv2

    if det is None:
        return None

    if det.landmarks is not None and len(det.landmarks) >= 4:
        pts = det.landmarks
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
        half = max(span * 1.4, 24.0) / 2
    else:
        x1, y1, x2, y2 = det.bbox
        fw, fh = x2 - x1, y2 - y1
        cx, cy = x1 + fw / 2, y1 + fh * 0.75
        half = max(fw * 0.35, 24.0) / 2

    h, w = frame.shape[:2]
    mx1, my1 = int(max(0, cx - half)), int(max(0, cy - half))
    mx2, my2 = int(min(w, cx + half)), int(min(h, cy + half))
    patch = frame[my1:my2, mx1:mx2]
    if patch.size == 0:
        return None
    grey = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def extract_audio(video_path: str | Path, out_wav: Path, sr: int = AUDIO_SR) -> float:
    """Decode to 16 kHz mono wav. Returns duration, 0.0 if there is no audio.

    ffmpeg is invoked with the network protocol whitelist emptied and no shell:
    this same function runs on user uploads in the API, so it is
    hardened here rather than in a second copy.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-nostdin", "-y",
        "-protocol_whitelist", "file",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        out_wav.unlink(missing_ok=True)
        return 0.0

    if not out_wav.exists() or out_wav.stat().st_size <= 44:   # wav header only
        out_wav.unlink(missing_ok=True)
        return 0.0

    import soundfile as sf

    info = sf.info(str(out_wav))
    if info.frames == 0:
        out_wav.unlink(missing_ok=True)
        return 0.0
    return float(info.frames / info.samplerate)


def audio_snr_estimate(wav_path: Path) -> float:
    """Crude speech-presence proxy in [0, 1], for the reliability gate.

    Ratio of the 90th-percentile frame energy to the 10th: high for clean
    speech, near zero for a silent or constant-noise track. Not a calibrated
    SNR, and not claimed to be -- it only needs to order inputs so the gate can
    discount a track with no usable speech.
    """
    import soundfile as sf

    try:
        y, sr = sf.read(str(wav_path), dtype="float32")
    except Exception:  # noqa: BLE001
        return 0.0
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < sr // 10:
        return 0.0

    win = max(sr // 50, 1)
    n = y.size // win
    if n < 4:
        return 0.0
    energy = (y[: n * win].reshape(n, win) ** 2).mean(axis=1)
    hi, lo = np.percentile(energy, 90), np.percentile(energy, 10)
    if hi <= 1e-12:
        return 0.0
    ratio = float(10 * np.log10((hi + 1e-12) / (lo + 1e-12)))
    return float(np.clip(ratio / 40.0, 0.0, 1.0))


def blur_estimate(gray: np.ndarray) -> float:
    """Laplacian-variance sharpness mapped to [0, 1]; 1 = sharp.

    Feeds the reliability gate and the per-face-size breakdown. Heavily
    compressed social-media video sits low here, which is exactly the regime
    Objective 4 is about.
    """
    import cv2

    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(np.log1p(v) / np.log1p(1000.0), 0.0, 1.0))


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def cache_dir_for(cache_root: str | Path, dataset: str, video_id: str) -> Path:
    return Path(cache_root) / dataset / video_id


def is_cached(cache_dir: Path) -> bool:
    meta_p = cache_dir / "meta.json"
    if not meta_p.exists():
        return False
    try:
        meta = json.loads(meta_p.read_text())
    except json.JSONDecodeError:
        return False
    if meta.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return False
    return len(list((cache_dir / "faces").glob("*.jpg"))) > 0


def preprocess_video(
    video_path: str | Path,
    cache_dir: str | Path,
    cfg: PreprocessConfig | None = None,
    *,
    video_id: str = "",
    dataset: str = "custom",
) -> VideoMeta:
    """Decode, detect, crop and cache one video. Idempotent.

    This is THE shared entry point (contract 5.2). The API worker calls it on an
    upload with a temp ``cache_dir``; training calls it over a manifest.
    """
    import cv2

    cfg = cfg or PreprocessConfig()
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    video_id = video_id or video_path.stem

    if not cfg.overwrite and is_cached(cache_dir):
        return VideoMeta(**json.loads((cache_dir / "meta.json").read_text()))

    faces_dir, mouth_dir = cache_dir / "faces", cache_dir / "mouth"
    faces_dir.mkdir(parents=True, exist_ok=True)
    if cfg.extract_mouth:
        mouth_dir.mkdir(parents=True, exist_ok=True)

    meta = VideoMeta(
        video_id=video_id, dataset=dataset, source_path=str(video_path)
    )

    # ---- 1. sample + decode -------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if total <= 0:
        total = cfg.n_frames * 4   # unknown length; read_frames stops at EOF anyway

    idxs = sample_indices(
        total, cfg.n_frames, cfg.sampling, seed=frame_sample_seed(video_id)
    )
    frames, fps, total_real = read_frames(video_path, idxs)
    idxs = idxs[: len(frames)]

    meta.fps = fps
    meta.n_frames_total = total_real
    meta.frame_indices = [int(i) for i in idxs]
    meta.frame_timestamps = [round(i / fps, 4) for i in idxs]

    # ---- 2. detect ------------------------------------------------------
    det = get_detector(
        cfg.detector, cfg.device, tuple(cfg.det_thresholds)  # type: ignore[arg-type]
    )
    warn_if_fallback(det)
    meta.detector = det.name
    if tuple(cfg.det_thresholds) != (0.6, 0.7, 0.7):
        meta.warnings.append(f"non-default detector thresholds {tuple(cfg.det_thresholds)}")

    per_frame = det.detect(frames)
    h, w = frames[0].shape[:2]
    track = pick_main_track(per_frame, (w, h))

    # Drop boxes that are too small to carry any artefact signal; a 20px face
    # upsampled to 299px is noise, and keeping it dilutes the training set.
    for i, d in enumerate(track):
        if d is not None and min(d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) < cfg.min_face_px:
            track[i] = None

    meta.n_faces_found = sum(d is not None for d in track)
    meta.bboxes = [list(map(float, d.bbox)) if d else None for d in track]
    meta.det_confidences = [float(d.confidence) if d else 0.0 for d in track]

    if meta.n_faces_found == 0:
        meta.warnings.append(
            "no face detected in any sampled frame; centre crops stored instead"
        )

    # ---- 3. crop + write ------------------------------------------------
    from ddetect.data.integrity import phash_frame

    blurs: list[float] = []
    mouth_ok = 0
    for i, (frame, d) in enumerate(zip(frames, track)):
        face = crop_face(frame, d)
        cv2.imwrite(
            str(faces_dir / f"{i:03d}.jpg"),
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        grey = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        blurs.append(blur_estimate(grey))
        meta.phash.append(phash_frame(grey))

        if cfg.extract_mouth:
            m = crop_mouth(frame, d)
            if m is not None:
                cv2.imwrite(
                    str(mouth_dir / f"{i:03d}.jpg"), m,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                mouth_ok += 1

    meta.mouth_available = mouth_ok >= max(4, len(frames) // 4)

    # ---- 4. audio -------------------------------------------------------
    snr = 0.0
    if cfg.extract_audio:
        wav = cache_dir / "audio.wav"
        meta.audio_duration_s = extract_audio(video_path, wav)
        meta.has_audio = meta.audio_duration_s > 0.05
        if meta.has_audio:
            snr = audio_snr_estimate(wav)
        else:
            meta.warnings.append("no usable audio stream; audio+sync streams masked off")

    # ---- 5. reliability (contract: RELIABILITY_FIELDS order) ------------
    face_conf = float(np.mean([c for c in meta.det_confidences if c > 0] or [0.0]))
    meta.reliability = dict(
        zip(
            RELIABILITY_FIELDS,
            [
                round(snr, 4),
                round(face_conf, 4),
                round(mouth_ok / max(len(frames), 1), 4),
                round(float(np.mean(blurs)) if blurs else 0.0, 4),
            ],
        )
    )

    (cache_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    return meta
