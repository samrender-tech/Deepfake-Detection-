"""F15 - ``Detector``: the single inference entry point.

Used by ``evaluate.py``, the CLI, the API worker and the parity test. It calls
the SAME ``preprocess_video`` as training, which is what makes the demo
incapable of silently diverging from the reported numbers (contract 5.2).

It also applies the run's own calibration, conformal band and OOD threshold, so
a verdict shown to a user is the verdict the paper's numbers describe -- not a
raw sigmoid that happens to look confident.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ddetect.calibrate import Calibration
from ddetect.contracts import RELIABILITY_FIELDS, Result, StreamScore, Verdict
from ddetect.data.dataset import DataConfig, DeepfakeClipDataset
from ddetect.data.preprocess import (
    PreprocessConfig,
    VideoMeta,
    cache_dir_for,
    preprocess_video,
)
from ddetect.models.registry import build_model
from ddetect.ood import MahalanobisOOD, energy_score
from ddetect.utils.device import pick_device
from ddetect.utils.log import get_logger

log = get_logger(__name__)


class Detector:
    """Loads one run and predicts on arbitrary video files."""

    def __init__(
        self,
        model: torch.nn.Module,
        calibration: Calibration,
        data_cfg: DataConfig,
        preprocess_cfg: PreprocessConfig,
        device: torch.device | None = None,
        model_version: str = "",
        config_hash: str = "",
        run_dir: str = "",
        ood: MahalanobisOOD | None = None,
        train_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.eval()
        self.cal = calibration
        self.data_cfg = data_cfg
        self.pre_cfg = preprocess_cfg
        self.device = device or pick_device()
        self.model.to(self.device)
        self.model_version = model_version
        self.config_hash = config_hash
        self.run_dir = run_dir
        self.ood = ood
        #: The training config read out of the checkpoint; surfaced by the
        #: API's model-info endpoint, never used to change inference.
        self.train_cfg: dict[str, Any] = dict(train_cfg or {})
        #: One Detector is shared by every API worker thread. Grad-CAM registers
        #: hooks on the shared model, so a forward pass from another thread
        #: while they are attached overwrites the captured activations: the
        #: explanation then fails or, worse, shows another clip's heat map.
        #: Every use of the model holds this lock.
        self._model_lock = threading.RLock()

    # ------------------------------------------------------------------
    @classmethod
    def from_run(
        cls, run_dir: str | Path, device: torch.device | None = None, prefer: str = "best"
    ) -> Detector:
        """Rebuild a detector from a run directory.

        Reads the training config out of the checkpoint rather than taking it
        from the caller, so inference geometry (frame count, image size, audio
        length) can never drift from what the model was trained on.
        """
        run_dir = Path(run_dir)
        ckpt_p = run_dir / (f"ckpt_{prefer}.pt")
        if not ckpt_p.exists():
            alt = run_dir / ("ckpt_last.pt" if prefer == "best" else "ckpt_best.pt")
            if not alt.exists():
                raise FileNotFoundError(f"no checkpoint in {run_dir}")
            log.warning("%s missing, using %s", ckpt_p.name, alt.name)
            ckpt_p = alt

        ckpt = torch.load(ckpt_p, map_location="cpu")
        tcfg = ckpt.get("cfg", {})

        model = build_model(tcfg.get("model", "smoke"))
        state = ckpt.get("model") or ckpt.get("raw_model")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("checkpoint missing %d tensors (e.g. %s)", len(missing), list(missing)[:3])
        if unexpected:
            log.warning("checkpoint has %d unexpected tensors", len(unexpected))

        cal = Calibration.load(run_dir / "calibration.json")
        ood = MahalanobisOOD.load(run_dir / "ood_mahalanobis.json")

        data_cfg = DataConfig(
            n_frames=int(tcfg.get("n_frames", 32)),
            image_size=int(tcfg.get("image_size", 299)),
            audio_seconds=float(tcfg.get("audio_seconds", 4.0)),
            max_sync_windows=int(tcfg.get("max_sync_windows", 6)),
            split="test",
            require_cache=False,
        )
        pre_cfg = PreprocessConfig(
            n_frames=int(tcfg.get("n_frames", 32)),
            detector=tcfg.get("detector", "auto"),
            device="cpu",
        )
        return cls(
            model,
            cal,
            data_cfg,
            pre_cfg,
            device,
            model_version=f"{tcfg.get('exp', '?')}/seed{tcfg.get('seed', '?')}@{ckpt.get('git_sha', '?')}",
            config_hash=str(ckpt.get("git_sha", "")),
            run_dir=str(run_dir),
            ood=ood,
            train_cfg=tcfg,
        )

    # ------------------------------------------------------------------
    def _build_batch(self, cache_dir: Path, video_id: str) -> tuple[dict[str, Any], VideoMeta]:
        """Assemble a 1-item batch from the cache using the training Dataset.

        Reusing ``DeepfakeClipDataset.__getitem__`` rather than reimplementing
        the loading is the whole point of contract 5.2.
        """
        import pandas as pd

        from ddetect.data.dataset import collate

        meta = VideoMeta(**json.loads((cache_dir / "meta.json").read_text()))
        row = {
            "video_id": video_id,
            "path": meta.source_path,
            "label": 0,
            "dataset": meta.dataset or "custom",
            "forgery_method": "real",
            "split": "test",
            "has_audio": bool(meta.has_audio),
            "source_identity": video_id,
            "compression": "unknown",
            "fps": float(meta.fps or 25.0),
            "n_frames": int(meta.n_frames_total or 1),
            "duration_s": 1.0,
            "sha256": "0" * 64,
        }
        # The Dataset resolves its own path as cache_root/<dataset>/<video_id>
        # (cache_dir_for), so cache_root must be TWO levels above cache_dir --
        # not one. Getting this wrong does not raise: the loader simply finds no
        # frames and no audio, silently scores an all-zero clip, and every video
        # receives an identical verdict. Assert the round trip instead of
        # trusting the arithmetic.
        dataset_name = row["dataset"]
        cache_root = cache_dir.parent.parent
        resolved = cache_dir_for(cache_root, dataset_name, video_id)
        if resolved.resolve() != cache_dir.resolve():
            raise RuntimeError(
                f"cache layout mismatch: Dataset would read {resolved} but the "
                f"cache was written to {cache_dir}. Expected the cache at "
                f"<root>/{dataset_name}/{video_id}."
            )

        cfg = DataConfig(**{**self.data_cfg.__dict__, "cache_root": str(cache_root)})
        cfg.require_cache = False
        ds = DeepfakeClipDataset(cfg, df=pd.DataFrame([row]))
        item = ds[0]

        # A blank clip means the cache was not found. Fail loudly: a verdict
        # computed from zeros is worse than no verdict.
        if float(item["faces"].abs().sum()) == 0.0:
            raise RuntimeError(
                f"loaded an all-zero clip for {video_id} from {cache_dir}; "
                "the preprocessing cache is missing or unreadable"
            )
        return collate([item]), meta

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        b = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with self._model_lock:
            return self.model(b)

    def predict(
        self,
        video_path: str | Path,
        explain: bool = True,
        cache_dir: str | Path | None = None,
    ) -> Result:
        """Full pipeline on one video file -> ``Result`` (contract 5.5)."""
        t0 = time.time()
        video_path = Path(video_path)
        video_id = video_path.stem
        timings: dict[str, float] = {}

        tmp: tempfile.TemporaryDirectory | None = None
        if cache_dir is None:
            tmp = tempfile.TemporaryDirectory(prefix="ddetect_")
            cdir = Path(tmp.name) / "custom" / video_id
        else:
            cdir = Path(cache_dir)

        try:
            # ---- 1. preprocess (the SAME function training uses) ------
            t = time.time()
            meta = preprocess_video(
                video_path, cdir, self.pre_cfg, video_id=video_id, dataset="custom"
            )
            timings["preprocess_ms"] = (time.time() - t) * 1000

            # ---- 2. inference ----------------------------------------
            t = time.time()
            batch, _ = self._build_batch(cdir, video_id)
            out = self._forward(batch)
            timings["inference_ms"] = (time.time() - t) * 1000

            logit = float(out["logit"][0])
            raw = 1.0 / (1.0 + np.exp(-logit))

            # ---- 3. calibrate + abstain + OOD ------------------------
            cal_p = float(self.cal.probability(np.array([logit]))[0])
            abstain = bool(self.cal.abstains(np.array([cal_p]))[0])

            e_score = float(energy_score(np.array([logit]))[0])
            ood_score = e_score
            if self.ood is not None and "emb" in out:
                m = float(self.ood.score(out["emb"][0:1].float().cpu().numpy())[0])
                # Report the Mahalanobis distance when available: it detects
                # feature-space novelty a confident logit cannot express.
                ood_score = m
            ood_flag = bool(
                np.isfinite(self.cal.ood_threshold) and e_score > self.cal.ood_threshold
            )

            if abstain or ood_flag:
                verdict = Verdict.INCONCLUSIVE
            elif cal_p >= self.cal.threshold:
                verdict = Verdict.MANIPULATED
            else:
                verdict = Verdict.AUTHENTIC

            # ---- 4. assemble the evidence ----------------------------
            res = Result(
                video_id=video_id,
                model_version=self.model_version,
                config_hash=self.config_hash,
                run_dir=self.run_dir,
                verdict=verdict,
                calibrated_prob=cal_p,
                conformal_lo=self.cal.conformal_lo,
                conformal_hi=self.cal.conformal_hi,
                raw_score=raw,
                threshold=self.cal.threshold,
                ood_score=ood_score,
                ood_flag=ood_flag,
                has_audio=bool(meta.has_audio),
                n_faces_found=int(meta.n_faces_found),
                reliability=dict(meta.reliability),
                warnings=list(meta.warnings),
                calibration_degenerate=bool(getattr(self.cal, "degenerate", False)),
            )

            if "frame_logits" in out:
                fl = torch.sigmoid(out["frame_logits"][0]).float().cpu().numpy()
                res.frame_scores = [round(float(x), 5) for x in fl]
                res.frame_timestamps = meta.frame_timestamps[: len(fl)]
            if "sync_seq" in out:
                res.sync_curve = [
                    round(float(x), 5) for x in out["sync_seq"][0].float().cpu().numpy()
                ]
                emb = out.get("emb")
                if "sync_seq" in out and emb is not None:
                    # emb[1] of the sync stream's feature vector is `conf`;
                    # exposed only when the sync stream actually ran.
                    pass
            if "gate_weights" in out:
                from ddetect.models.reliability import STREAMS

                res.gate_weights = {
                    s: round(float(out["gate_weights"][0][i]), 4) for i, s in enumerate(STREAMS)
                }

            sl = cast("dict[str, torch.Tensor]", out.get("stream_logits") or {})
            for name in ("visual", "audio", "sync"):
                if name in sl:
                    available = name == "visual" or bool(meta.has_audio)
                    rel = res.gate_weights.get(name, 1.0)
                    note = ""
                    if name in ("audio", "sync") and not meta.has_audio:
                        note = "no audio track; stream masked off"
                    elif name == "sync" and not meta.mouth_available:
                        note = "mouth region not reliably visible"
                    res.streams.append(
                        StreamScore(
                            name=name,
                            score=round(float(torch.sigmoid(sl[name][0])), 5)
                            if available
                            else None,
                            available=available,
                            reliability=float(rel),
                            note=note,
                        )
                    )

            if getattr(self.cal, "degenerate", False):
                res.warnings.append(
                    "This model's probability calibration was fitted on too little "
                    "validation data, so the percentage shown is not trustworthy. "
                    "Treat the verdict as indicative only."
                )
            if meta.n_faces_found == 0:
                res.warnings.append(
                    "No face was detected, so the visual stream scored centre crops. "
                    "Treat this result as unreliable."
                )

            # ---- 5. explainability -----------------------------------
            if explain:
                t = time.time()
                try:
                    self._attach_explanations(res, batch, cdir)
                except Exception as e:
                    log.warning("explainability failed: %s: %s", type(e).__name__, e)
                timings["explain_ms"] = (time.time() - t) * 1000

            timings["total_ms"] = (time.time() - t0) * 1000
            res.latency_ms = {k: round(v, 1) for k, v in timings.items()}
            return res
        finally:
            if tmp is not None:
                tmp.cleanup()

    # ------------------------------------------------------------------
    def _attach_explanations(self, res: Result, batch: dict[str, Any], cdir: Path) -> None:
        """Grad-CAM++ overlays and the audio spectrum."""
        import cv2

        from ddetect.explain import (
            GradCAMPlusPlus,
            find_cam_layer,
            overlay_cam,
            pick_top_frames,
            png_b64,
            spectrum_figure,
        )

        top = pick_top_frames(res.frame_scores, k=3)
        layer = find_cam_layer(self.model)
        with self._model_lock:
            if layer is not None and top:
                cam = GradCAMPlusPlus(self.model, layer)
                try:
                    with torch.enable_grad():
                        b = {
                            k: (v.to(self.device) if torch.is_tensor(v) else v)
                            for k, v in batch.items()
                        }
                        # Explain ONLY the selected frames. Running the backbone
                        # over all T frames to produce 3 heat maps made Grad-CAM
                        # ~73% of total request latency (measured: 18.6s of 25.4s)
                        # and pushed the full path past the 20s budget. Slicing to
                        # the top-k is numerically identical for those frames --
                        # the visual stream scores frames independently -- and
                        # roughly T/k times cheaper.
                        idx = torch.tensor(top, device=b["faces"].device)
                        b["faces"] = b["faces"][:, idx].contiguous().requires_grad_(True)

                        out = self.model(b)
                        # Per-frame score, not the video logit: a gradient through
                        # the temporal pooling would attribute to the attention
                        # weights rather than to the image evidence.
                        # frame_logits is now exactly the top-k frames, because
                        # the batch was sliced above.
                        target = out["frame_logits"][0] if "frame_logits" in out else out["logit"]
                        maps = cam(target)
                    if maps is not None:
                        faces = sorted((cdir / "faces").glob("*.jpg"))
                        for j, fi in enumerate(top):
                            if fi >= len(faces) or j >= len(maps):
                                continue
                            img = cv2.cvtColor(cv2.imread(str(faces[fi])), cv2.COLOR_BGR2RGB)
                            key = f"frame_{fi}"
                            res.gradcam_keys.append(key)
                            # Stored inline as base64; the API moves these to object
                            # storage and replaces the payload with keys.
                            res._gradcam_png[key] = png_b64(overlay_cam(img, maps[j]))
                finally:
                    cam.remove()
                    self.model.zero_grad(set_to_none=True)

        # Grad-CAM overlays and the spectrum are attached as private
        # attributes and moved to object storage by the API worker, so the
        # Result stays small enough to cache. Declared on the dataclass rather
        # than set blindly, so a typo is caught.
        wav_p = cdir / "audio.wav"
        if wav_p.exists():
            try:
                import soundfile as sf

                y, sr = sf.read(str(wav_p), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                png = spectrum_figure(y[: sr * 5], sr)
                if png:
                    res.spectrum_key = "spectrum"
                    res._spectrum_png = png
            except Exception:
                pass

    # ------------------------------------------------------------------
    def explain_payload(self, res: Result) -> dict[str, Any]:
        """The payload A7 is restricted to.

        Contains ONLY numbers already in the Result. Nothing is recomputed or
        added here, so the grounding eval can replay it faithfully.
        """
        peak = int(np.argmax(res.frame_scores)) if res.frame_scores else None
        return {
            "verdict": res.verdict.value,
            "calibrated_probability": round(res.calibrated_prob, 4),
            "conformal_band": [res.conformal_lo, res.conformal_hi],
            "decision_threshold": res.threshold,
            "abstained": res.verdict is Verdict.INCONCLUSIVE,
            "ood_flag": res.ood_flag,
            "has_audio": res.has_audio,
            "n_faces_found": res.n_faces_found,
            "n_frames_scored": len(res.frame_scores),
            "frame_score_min": round(min(res.frame_scores), 4) if res.frame_scores else None,
            "frame_score_max": round(max(res.frame_scores), 4) if res.frame_scores else None,
            "peak_frame_index": peak,
            "peak_frame_time_s": (
                res.frame_timestamps[peak]
                if peak is not None and peak < len(res.frame_timestamps)
                else None
            ),
            "sync_curve_mean": round(float(np.mean(res.sync_curve)), 4) if res.sync_curve else None,
            "sync_offset_ms": res.sync_offset_ms,
            "streams": [
                {
                    "name": s.name,
                    "score": s.score,
                    "available": s.available,
                    "reliability": round(s.reliability, 3),
                    "note": s.note,
                }
                for s in res.streams
            ],
            "gate_weights": res.gate_weights,
            "reliability": {k: res.reliability.get(k) for k in RELIABILITY_FIELDS},
            "warnings": res.warnings,
            "model_version": res.model_version,
        }


# --------------------------------------------------------------------------
def load_detector(run_dir: str | Path | None = None, **kw: Any) -> Detector:
    """Convenience loader honouring ``$DDETECT_RUN_DIR``."""
    import os

    rd = run_dir or os.environ.get("DDETECT_RUN_DIR")
    if not rd:
        raise ValueError("no run_dir given and $DDETECT_RUN_DIR is unset")
    return Detector.from_run(rd, **kw)
