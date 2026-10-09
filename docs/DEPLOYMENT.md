# Deployment

Three configurations, in increasing order of what they need from you.

## 1. Single command (the default)

```bash
make serve RUN=runs/<exp>/seed0
```

In-memory jobs, local blob storage, an in-process thread pool. No Redis, no
Postgres, no S3. This is not a stub — it is a working configuration, and it is
what the test suite exercises.

The in-memory default is deliberate. This service stores verdicts about whether
a real person's video is fake, and uploads of their face. Persisting that by
default would be the wrong default; the operator opts in by configuring a
database, and the retention policy applies either way.

## 2. Containerised

```bash
docker compose -f docker/compose.yml up
docker compose -f docker/compose.yml --profile observability up   # + Prometheus/Grafana
```

API, out-of-process worker, Postgres, Redis and MinIO. Scale the CPU-bound part
with `--scale worker=3` — more worker *processes* beats more threads per
process, because inference is CPU-bound and oversubscribing makes every request
slower rather than throughput higher.

## 3. Behind a CDN

`docker/Dockerfile.web` builds the frontend as a standalone nginx image for
when the UI and API are scaled independently. The API image also embeds the
built assets, so this is only needed if you are actually splitting them.

---

## Configuration

Selection is by DSN presence, so a deployment that forgets to configure
something degrades rather than failing to boot — and `/readyz` reports which
backend is actually live.

| Variable | Effect when unset |
|---|---|
| `DDETECT_RUN_DIR` | **Required.** No detector loads; uploads are refused with 503 |
| `DATABASE_URL` | In-memory jobs |
| `REDIS_URL` | In-process thread pool instead of the arq worker |
| `S3_BUCKET` / `S3_ENDPOINT` | Local disk blobs |
| `API_KEYS` | **Open access.** Set this in production |
| `RATE_LIMIT_ANALYSES` | `10/minute` per client |
| `MAX_UPLOAD_MB`, `MAX_DURATION_S` | 100 MB, 60 s |
| `RETENTION_HOURS` | 24 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Tracing off |
| `DDETECT_TRACKER` | Training untracked (the run directory stays authoritative) |
| `ANTHROPIC_API_KEY` | A7 uses its deterministic template |

---

## Model export

```bash
make export RUN=runs/<exp>/seed0
```

Produces TorchScript, ONNX and INT8 builds of the **visual stream**, each
parity-checked against the eager model. An export that drifts past tolerance is
**deleted, not shipped with a warning** — a traced graph that drops a branch
still runs and still returns a number, it just stops being the model the paper
describes.

Only the visual stream is exported, on purpose: it is the expensive part, it
has a fixed tensor signature, and the audio and sync streams involve dynamic
window counts and a frozen SyncNet that traces badly. The fusion head is three
linear layers — eager costs nothing.

INT8 is reported, not gated. Quantisation genuinely changes the arithmetic, so
the measured drift is surfaced and the deployment decides whether it is
acceptable near its threshold.

---

## What to watch

`/metrics` (Prometheus) and `/v1/drift`. The generic HTTP counters are there,
but these are the ones that say something about *this* service:

- **`avforge_verdicts_total` by verdict.** A sudden swing toward
  `likely_manipulated` is either an attack or a broken checkpoint.
- **`avforge_ood_flags_total`.** Rising means the model is being used outside
  the domain it was calibrated on.
- **`avforge_abstentions_total`.** Rising is the system working as designed;
  *falling to zero* means the band has stopped firing, which is a bug.
- **Score PSI** (`/v1/drift`). Above 0.25 the calibration and the abstention
  band were fitted on a distribution that no longer matches the traffic.
- **`no_face_rate`.** Above ~25% means most verdicts are unreliable regardless
  of what the model says.

The drift baseline is built from the run's own `preds_val.csv`, so it is the
same distribution the threshold and conformal band were fitted on.

---

## Security

See [THREAT_MODEL.md](THREAT_MODEL.md). The containers run as a non-root user
with all capabilities dropped, `no-new-privileges`, and memory and pid limits,
because the process parses hostile video with ffmpeg and OpenCV.

Set `API_KEYS` before exposing this to anything. Rate limiting is on by default
but it is a fairness mechanism, not an authentication one.
