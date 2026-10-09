# AVFORGE — notes for future sessions

Cross-dataset audio-visual deepfake **detector**. The project's claim is an
honest measurement of how badly detectors generalise to unseen forgery methods,
and an attempt to reduce that gap.

Feature tags in code comments (F3, F12, ...) and contract numbers (5.1-5.5 in
`ddetect/contracts.py`) are internal labels for parts of the system.

## Before changing anything

Read `ddetect/contracts.py`. Five interfaces are frozen there and enforced by
`tests/test_contracts.py`. They are what keep the data, model, metrics and
serving parts independent; changing one is a design decision, not a refactor.

## Things that will bite you

- **The cross-dataset gap IS the result.** If a change makes Baseline B's drop
  *smaller*, the first hypothesis is leakage, not success. Run
  `pytest tests/test_leakage.py` before believing any improvement.
- **The threshold must come from the source validation split.** There is no
  code path in `ddetect/metrics.py` that allows otherwise, deliberately. Do not
  add one.
- **Training and serving share `preprocess_video()`.** Do not write a second
  loading path for the API; `tests/test_detector_parity.py` will catch it, but
  the point is that the demo and the paper must describe the same model.
- **Agents cannot read target test splits.** `agents/core/firewall.py`.
  Target tests run once per frozen config via `make eval-final`.
- **Silent failures are the norm in this codebase's bug history.** Every bug
  found so far produced plausible-looking output rather than an exception: a
  blank clip scored confidently, augmentation that did nothing, abstention that
  never fired. Prefer an assertion over a graceful degradation when the
  degraded result would still look like a number.

## Environment

- macOS/MPS is for authoring and smoke tests only. CUDA is authoritative for
  every reported number (`device_report()['authoritative']`).
- `Conv3d` and `AdaptiveAvgPool3d` have no MPS kernels; the sync stream uses a
  (2+1)D factorisation instead. `torch.roll` needs contiguous tensors on MPS.
- The numeric stack is pinned to the numpy 1.x generation because mediapipe
  requires `numpy<2` and OpenCV 5 dropped `CascadeClassifier`.

## Commands

```bash
make smoke     # whole chain on synthetic fixtures, no datasets needed
make test      # full suite (~5 min)
make lint typecheck scope
make serve RUN=runs/<exp>/seed0
python -m experiments.aggregate    # results tables + IEEE LaTeX
```

## Agents

Seven research agents in `agents/` plus A7 in `api/explain_agent.py`. Each does
real deterministic analysis and treats the LLM as an optional narrative layer,
so they all work with no API key. `make agents` runs the eval suites (A1
citations and A7 grounding block CI); `make agent-audit` runs them over the
current state.

The firewall (`agents/core/firewall.py`) is the constraint that matters: only
A5 and A6 may read test results, and only because neither has a training tool.

## Serving

`make serve` needs only the core dependencies: in-memory jobs, local blobs,
in-process worker. Every optional backend (Postgres, Redis, S3, Prometheus,
MLflow, OTel) degrades to that default when its DSN is unset, and `/readyz`
reports which is live. Keep it that way — the single-command demo is what the
tests exercise.

`make export` writes TorchScript/ONNX/INT8 builds of the visual stream, each
parity-checked against eager. **An export that drifts past tolerance is
deleted, not warned about**: a traced graph that drops a branch still runs and
still returns a number.

`docs/DEPLOYMENT.md` covers the containerised stack and what to watch.

`ddetect/report.py` builds suspicious segments and the HTML report from the
`Result.to_dict()` shape only, so a report cannot show a number the detector
did not produce. Its `REPORT_DISCLAIMER` must stay identical to
`api.schemas.DISCLAIMER` (asserted in `tests/test_report.py`).

**One `Detector` is shared by all API worker threads.** Any use of its model
must hold `Detector._model_lock`: Grad-CAM hooks on the shared model otherwise
capture another thread's forward pass and attach the wrong clip's heat map
(`test_concurrent_predictions_match_sequential_ones`).

## Current state

All 32 planned features are built: research pipeline, agent layer, and the
production-ops tier (storage backends, out-of-process worker, auth, rate
limiting, metrics, drift monitoring, model export, tracking, containers, e2e).

**No real dataset numbers exist yet.** The FF++ / Celeb-DF / FakeAVCeleb access
forms are the blocker; DFDC (Kaggle, instant, has audio) is the unblocked path.
Everything currently in `runs/` and `results/` describes a model trained for
two epochs on four synthetic clips and means nothing.

See `docs/EXPERIMENT_LOG.md` for what has been run and what was learned.
