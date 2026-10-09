"""Face detection with a graceful fallback chain.

Wheels for MTCNN / MediaPipe are the most fragile part of this stack on Apple
Silicon, and the walking skeleton (Stage 1) must run on the Mac before any
cloud training happens. So detection is an interface with three backends:

    mtcnn      facenet-pytorch. Best quality; what the reported numbers use.
    mediapipe  Google BlazeFace. Fast, CPU-friendly, gives lip landmarks.
    haar       OpenCV cascade. Always available; smoke tests only.

The backend actually used is written into ``meta.json`` (``detector``), because
a cache built with one detector is not comparable to a cache built with
another -- crops differ, and a model fine-tuned on one will mis-score the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from ddetect.utils.log import get_logger

log = get_logger(__name__)

Backend = Literal["mtcnn", "mediapipe", "haar", "auto"]

# MediaPipe FaceMesh outer-lip ring, used for the mouth crop (SyncNet input).
LIP_LANDMARKS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
)


@dataclass
class Detection:
    """One face. ``bbox`` is (x1, y1, x2, y2) in pixels."""

    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: np.ndarray | None = None   # (N, 2) pixel coords, if available

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class FaceDetector(Protocol):
    name: str

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """RGB uint8 frames -> per-frame detections."""
        ...


# --------------------------------------------------------------------------
#: MTCNN's published cascade thresholds. Real faces clear these comfortably;
#: the synthetic test fixtures do not, which is why they are configurable.
#: Lowering them on real data costs precision (more false boxes for
#: ``pick_main_track`` to reject), so only the fixtures do it.
DEFAULT_MTCNN_THRESHOLDS = (0.6, 0.7, 0.7)
PERMISSIVE_MTCNN_THRESHOLDS = (0.2, 0.2, 0.2)


class MTCNNDetector:
    name = "mtcnn"

    def __init__(
        self,
        device: str = "cpu",
        min_face: int = 40,
        thresholds: tuple[float, float, float] = DEFAULT_MTCNN_THRESHOLDS,
    ) -> None:
        from facenet_pytorch import MTCNN

        self._m = MTCNN(
            keep_all=True,
            min_face_size=min_face,
            device=device,
            post_process=False,
            select_largest=False,
            thresholds=list(thresholds),
        )

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        from PIL import Image

        pils = [Image.fromarray(f) for f in frames]
        # MTCNN batches only equal-sized images; videos are uniform, but a
        # mixed list (e.g. an API upload re-encoded mid-stream) would throw.
        try:
            boxes, probs, points = self._m.detect(pils, landmarks=True)
        except (ValueError, RuntimeError):
            boxes, probs, points = [], [], []
            for p in pils:
                b, pr, pt = self._m.detect(p, landmarks=True)
                boxes.append(b); probs.append(pr); points.append(pt)

        out: list[list[Detection]] = []
        for b, pr, pt in zip(boxes, probs, points):
            dets: list[Detection] = []
            if b is not None and len(b):
                for i in range(len(b)):
                    conf = float(pr[i]) if pr is not None and pr[i] is not None else 0.0
                    lm = np.asarray(pt[i], dtype=np.float32) if pt is not None else None
                    dets.append(Detection(tuple(map(float, b[i])), conf, lm))
            out.append(dets)
        return out


class MediaPipeDetector:
    """BlazeFace detection + FaceMesh landmarks (for the lip ring)."""

    name = "mediapipe"

    def __init__(self, min_conf: float = 0.5) -> None:
        import mediapipe as mp

        self._det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf
        )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=min_conf,
        )

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for f in frames:
            h, w = f.shape[:2]
            res = self._det.process(f)
            dets: list[Detection] = []
            for d in res.detections or []:
                bb = d.location_data.relative_bounding_box
                x1, y1 = bb.xmin * w, bb.ymin * h
                dets.append(
                    Detection(
                        (x1, y1, x1 + bb.width * w, y1 + bb.height * h),
                        float(d.score[0]),
                    )
                )
            if dets:
                mesh = self._mesh.process(f)
                if mesh.multi_face_landmarks:
                    lm = mesh.multi_face_landmarks[0].landmark
                    pts = np.array(
                        [[lm[i].x * w, lm[i].y * h] for i in LIP_LANDMARKS],
                        dtype=np.float32,
                    )
                    # Attach to the largest box; FaceMesh only returned one face.
                    dets.sort(key=lambda d: d.area, reverse=True)
                    dets[0].landmarks = pts
            out.append(dets)
        return out


class HaarDetector:
    """Always-available fallback. Smoke tests only -- do not report numbers."""

    name = "haar"

    def __init__(self) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._c = cv2.CascadeClassifier(path)
        if self._c.empty():
            raise RuntimeError(f"could not load Haar cascade from {path}")

    def detect(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        import cv2

        out: list[list[Detection]] = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            boxes = self._c.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            out.append(
                [
                    Detection((float(x), float(y), float(x + w), float(y + h)), 0.5)
                    for x, y, w, h in boxes
                ]
            )
        return out


# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_detector(
    backend: Backend = "auto",
    device: str = "cpu",
    mtcnn_thresholds: tuple[float, float, float] = DEFAULT_MTCNN_THRESHOLDS,
) -> FaceDetector:
    """Build a detector, falling back down the chain with a loud warning.

    Cached: constructing MTCNN loads weights, and preprocessing calls this per
    worker, not per video.
    """
    order = ["mtcnn", "mediapipe", "haar"] if backend == "auto" else [backend]
    errors: list[str] = []
    for name in order:
        try:
            if name == "mtcnn":
                return MTCNNDetector(device=device, thresholds=mtcnn_thresholds)
            if name == "mediapipe":
                return MediaPipeDetector()
            if name == "haar":
                return HaarDetector()
        except Exception as e:  # noqa: BLE001 - any import/weight failure falls through
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError("no face detector available:\n  " + "\n  ".join(errors))


def warn_if_fallback(det: FaceDetector) -> None:
    if det.name != "mtcnn":
        log.warning(
            "face detector is %r, not 'mtcnn'. Fine for smoke tests; "
            "do NOT report metrics from a cache built this way "
            "(install the 'face' extra: pip install -e '.[face]')",
            det.name,
        )


# --------------------------------------------------------------------------
def pick_main_track(
    per_frame: list[list[Detection]], image_wh: tuple[int, int]
) -> list[Detection | None]:
    """Reduce multi-face detections to one consistent subject per frame.

    Naive "largest box per frame" jumps between people in multi-speaker clips
    (common in DFDC), which destroys the temporal and lip-sync signals. We pick
    the largest face in the most confident frame as the anchor, then follow the
    nearest box frame to frame, allowing a gap of a few frames before re-anchoring.
    """
    w, h = image_wh
    diag = float(np.hypot(w, h))
    max_jump = 0.25 * diag   # a real face does not teleport a quarter-diagonal

    anchor_idx = max(
        range(len(per_frame)),
        key=lambda i: max((d.area for d in per_frame[i]), default=0.0),
        default=0,
    )
    if not per_frame or not per_frame[anchor_idx]:
        return [None] * len(per_frame)

    anchor = max(per_frame[anchor_idx], key=lambda d: d.area)
    track: list[Detection | None] = [None] * len(per_frame)
    track[anchor_idx] = anchor

    def walk(indices: range, start: Detection) -> None:
        prev = start
        for i in indices:
            cands = per_frame[i]
            if not cands:
                continue
            px, py = prev.center
            best = min(cands, key=lambda d: np.hypot(d.center[0] - px, d.center[1] - py))
            if np.hypot(best.center[0] - px, best.center[1] - py) <= max_jump:
                track[i] = best
                prev = best
            else:
                # Subject left frame / cut. Re-anchor on the largest face.
                best = max(cands, key=lambda d: d.area)
                track[i] = best
                prev = best

    walk(range(anchor_idx + 1, len(per_frame)), anchor)
    walk(range(anchor_idx - 1, -1, -1), anchor)
    return track
