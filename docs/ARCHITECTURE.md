# Architecture

## The shape of it

```
video ─▶ preprocess cache ─▶ batch dict ─┬─ visual stream ─┐
                                         ├─ audio stream  ─┼─▶ fusion ─▶ logit
                                         └─ sync stream   ─┘
                                                              │
                         calibration ─▶ abstention ─▶ OOD flag ┤
                                                              ▼
            training ─▶ runs/ ─▶ preds CSV ─▶ metrics ─▶ tables ─▶ paper
                              └▶ Detector ─▶ API ─▶ React dashboard
```

## Why the boundaries are where they are

**The cache is the only thing on everyone's critical path.** Once
`processed/{dataset}/{video_id}/` exists, the model work, the metrics work and
the serving work proceed independently.

**Evaluation reads only a CSV.** `ddetect/metrics.py` never imports a model.
That is what let the entire metrics, significance-testing and table-generation
stack be written and unit-tested against hand-made prediction files before any
model trained.

**One `preprocess_video` for training and serving.** The demo cannot drift
from the paper, and `tests/test_detector_parity.py` asserts the two paths
produce identical logits. A bug here is invisible in every log — an early
version of `Detector` resolved its cache root one level too high, found no
frames, and scored an all-zero clip, returning the same confident verdict for
every video. There is now an assertion that refuses a blank clip outright.

**Agents write only to `proposals/`.** The agentic layer can be developed in
parallel without any ability to break the pipeline or contaminate a result.

## The streams

**Visual** — a spatial backbone (Xception / EfficientNet) plus three optional
parts: a frequency branch (fixed SRM high-pass filters + DCT band energies), a
blending-boundary head supervised by SBI masks, and a temporal head over frame
embeddings. Baseline A is this same class with all three switched off, so the
baseline and the proposal cannot diverge as two separate codebases.

**Audio** — two tiers. A log-mel residual CNN that always works and trains on
a laptop, and a frozen WavLM encoder with an AASIST-style graph-attention
backend. Both are reported; the ablation is the point.

**Sync** — mouth crops and the spectrogram are encoded per 1-second window, and
the audio is rolled against the video across ±15 frame offsets. A genuine
recording has a sharp minimum at a consistent offset; a clip whose face and
voice came from different tools has a flat or erratic curve. The *shape*, not
the absolute distance, is the evidence — which is why `conf` (median − min) is
the headline feature. The encoder is trained contrastively on **real video
only**, so it cannot have learned any generator's fingerprint.

**Fusion** — a co-attention transformer over the three stream tokens with a
modality mask, plus a reliability gate that weights each stream by estimated
trustworthiness (audio SNR, face confidence, mouth visibility, blur). Modality
dropout during training is what lets one checkpoint serve both silent and
audio-bearing datasets.

## Implementation notes worth knowing

- **(2+1)D convolutions in the sync stream.** `Conv3d` has no MPS kernel, so a
  3-D stack silently falls back to CPU mid-batch on the Mac. The factorisation
  (Tran et al., CVPR 2018) is MPS-native, uses fewer parameters, and
  outperformed full 3-D convolution in the paper that introduced it.
- **`.contiguous()` before `torch.roll`.** On MPS, rolling a non-contiguous
  tensor trips an `MPSNDArray` assertion that aborts the process with no Python
  traceback.
- **Clip-level degradation.** Compression, blur and downscale are sampled once
  per clip and replayed across its frames. Per-frame sampling would make
  inter-frame quality variance a fingerprint of our own pipeline that no real
  video has.
- **Energy OOD uses the symmetric logit completion.** For a single-logit binary
  head, the obvious `(0, z)` completion reports a confidently-*authentic* video
  as maximally novel. `(+z/2, −z/2)` peaks at ambivalence, which is the
  intended semantics.
- **The sync offset search uses a non-cyclic shift.** `torch.roll` wraps, so
  offsets `k` and `k - W` are the same alignment and tie exactly; 1e-7 of
  batching noise then flips the argmin and moves the offset feature a whole
  step. Comparing only the overlapping region fixes it, at the cost of fewer
  windows contributing at large offsets.
- **Grad-CAM runs on the top-k frames only.** Explaining 3 frames by running
  the backbone over all 16 made it 73% of request latency (18.6s of 25.4s).
  Slicing first is numerically identical for those frames -- the visual stream
  scores frames independently -- and took the full path to 6.8s.
- **The conformal band comes from prediction sets.** Deriving it from the
  threshold collapses it to zero width whenever the nonconformity quantile
  exceeds 0.5 — silently disabling abstention on exactly the models that most
  need it.
