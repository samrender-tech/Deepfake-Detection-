# Experiment log

Append-only. One entry per run that produced a number anyone looked at,
including the ones that failed — a log that records only successes is how a
project ends up repeating a dead end three times.

Entries are written by `ddetect.train` automatically into
`runs/<exp>/seed<N>/`, and summarised here by hand (or by A3) with the
**hypothesis** stated before the result.

---

## Compute budget

| | GPU-hours |
|---|---|
| Estimated for the full grid (`make matrix`) | ~198 |
| Budget ceiling | 350 |
| Used so far | 0 |

Sources, in order of preference: Kaggle (30 free GPU-h/week × 3 accounts),
Colab Pro, then spot A100 for final seeds.

---

## Template

```
### <date> · <exp>/seed<N> · <who>
**Hypothesis.** What this run is supposed to show, written BEFORE it ran.
**Config.** <config hash> · <git sha>
**Result.** AUC / gap / the number that matters.
**Verdict.** Confirmed / refuted / inconclusive — and what changes next.
```

---

## Stage 0 — foundations

### 2026-10-06 · scaffolding · all
**What.** Repository, frozen contracts, preprocessing cache, all four model
streams, training engine, metrics stack, calibration + abstention + OOD,
serving layer, agent guardrails. 180+ tests.

**Findings worth recording** (each was a silent failure, not a crash):

1. **`Detector` scored an all-zero clip.** The cache root was resolved one
   directory too high, so no frames were found and every video received an
   identical confident verdict. Caught by noticing identical visual scores to
   five decimals across three different videos. There is now an assertion that
   refuses a blank clip outright, plus `tests/test_detector_parity.py`.

2. **Degradation augmentation was a no-op.** albumentations 2.0 renamed
   `quality_lower/upper` → `quality_range` and accepted the old names with a
   *warning*. Objective 4 would have been evaluated against augmentation that
   never ran. There is now `assert_albumentations_api()` and a test that the
   degraded image actually differs from the clean one.

3. **Degradation was applied twice** — once consistently per clip, then again
   per frame — which re-randomised it. Inter-frame quality variance would have
   become a fingerprint of our own pipeline that no real video has.

4. **The conformal band collapsed to zero width** whenever the nonconformity
   quantile exceeded 0.5, silently disabling abstention on exactly the models
   that most need it. Fixed by deriving the band from prediction sets.

5. **The energy OOD score was asymmetric**, reporting confidently-*authentic*
   videos as maximally novel (AUROC 0.51). The symmetric logit completion
   gives 0.90.

6. **`supcon` returned NaN** from `-inf × 0` on the masked diagonal.

7. **Grad-CAM was 73% of request latency** (18.6s of 25.4s) because it ran the
   backbone over all 16 frames to explain 3. Slicing to the top-k took it to
   2.8s, and the full request path from 25.4s to **6.8s** — inside the 20s
   budget with room to spare. Measured by `tests/bench_latency.py` on CPU.

8. **`.gitignore` excluded the entire data package.** A bare `data/` pattern
   matches *any* directory of that name, so `ddetect/data/` (preprocessing,
   manifests, dataset, SBI, the leakage guard) and `configs/data/` were both
   silently untracked — 20 files. A clean clone would have failed to import
   with no obvious cause. Patterns are now anchored with a leading slash.

9. **The sync stream's offset search used a cyclic shift.** `torch.roll` wraps,
   so with W windows an offset of `k` and one of `k - W` produce an identical
   alignment and a bit-identical distance. Those exact ties were then broken by
   ~1e-7 of batching noise: `argmin` flipped, `best_offset` moved a whole step
   (0.33 normalised), and the final logit moved **0.15 between batch sizes 1
   and 2** — the same clip scoring differently depending on what else shared
   its batch. Wrapping is also physically meaningless: audio from the end of a
   clip is not a candidate match for lips at the start. Replaced with a
   non-cyclic shift that compares only the overlapping region, capped so every
   offset is a distinct alignment. Caught by the parity test failing
   intermittently (~50%) in the full suite while passing in isolation; the
   invariant is now asserted for every model preset by
   `test_output_does_not_depend_on_batch_composition`.

**Verdict.** The pipeline runs end to end on synthetic fixtures. No real
dataset numbers exist yet — the dataset access forms are the blocker.

**Pattern worth noting.** All of them were silent: each produced plausible
output rather than an exception. None would have been caught by "does it
run?". The ones that were caught at all were caught by *disbelieving a
suspiciously clean result* — three videos scoring identically to five
decimals, a novelty detector at exactly chance, an abstention rate of 0.1%,
a parity test that passed alone and failed in company. That is the habit to
keep when the real datasets arrive.

---

### 2026-10-06 · agent layer and ablation ingredients · all
**What.** The seven research agents (A1-A6, A8) plus the shared infrastructure
they needed — typed tool registry, per-agent budgets with hard stops, a
content-addressed response cache — and three eval suites, two of which block
CI. Also the ablation ingredients: DANN, GroupDRO and IRM; the figure
generator; the offline degradation script; and the DeeperForensics and LAV-DF
parsers.

**Findings.**

10. **`paper/claims.csv` had a commented-out header.** `csv.DictReader` treats
    the first non-comment line as the header, so it silently consumed the first
    data row — six claims parsed where seven existed. Fixed by giving the file
    a real header row.

11. **The BibTeX parser required the closing brace on its own line.** A
    single-line entry parsed as nothing, and "0 references" is
    indistinguishable from an empty bibliography — exactly the silent miss A1
    exists to prevent. Replaced the regex with brace matching.

12. **The logged training loss excluded the DG terms.** `CompositeLoss` sets
    `parts["total"]` before the domain-generalisation terms are added, so the
    number in `metrics.jsonl` understated what was actually optimised. Any
    comparison of ERM against DANN would have been reading the wrong column.

13. **A closure in A1's arXiv parser captured the loop variable**, so every
    entry would have read the last one's body.

**Note on the DG objectives.** DomainBed (Gulrajani & Lopez-Paz, 2021) found
that under a fair protocol almost none of these reliably beat plain ERM. They
are therefore wired as ablations against an honest baseline rather than adopted
— a null result here is worth reporting. SWAD is the method that survived that
scrutiny, which is why it is the default.

**Verdict.** Still no real dataset numbers. A1 immediately flagged two
references in our own bibliography with no resolving identifier, and A3
reports the grid needs 198 GPU-hours against a 40-hour agent allocation.

---

### 2026-10-06 · production-ops tier · all
**What.** Pluggable storage (Postgres/S3 with memory+local fallbacks), the
out-of-process arq worker, API-key auth and rate limiting, Prometheus metrics,
structlog, OpenTelemetry, PSI drift monitoring, TorchScript/ONNX/INT8 export
with parity gates, MLflow/W&B tracking, worker and web Dockerfiles, the full
compose stack, a frontend history view and a Playwright end-to-end suite.

**Design decisions worth recording.**

* **Every optional dependency degrades to a working default.** A missing
  Redis, Postgres, S3, Prometheus, MLflow or OTel endpoint falls back rather
  than failing to boot, and `/readyz` reports which backend is actually live.
  The single-command demo has to keep working.
* **The run directory stays authoritative, not the tracking server.** If
  MLflow were the source of truth, reproducing a result would depend on a
  server being up, and the claim "a clean clone reproduces a headline table"
  would stop being true.
* **An export that fails parity is deleted, not warned about.** A traced graph
  that drops a branch still runs and still returns a number.
* **Only the visual stream is exported.** It is the expensive part with a
  fixed signature; the audio and sync streams have dynamic window counts and
  trace badly, and the fusion head is three linear layers.

**Findings.**

14. **`onnxruntime` dragged in NumPy 2.x** and broke the pinned stack
    (mediapipe requires `numpy<2`). Pinned `onnxruntime<1.20` / `onnx<1.17`,
    which are the last versions on the NumPy 1 side of the boundary.

15. **INT8 quantisation failed with `NoQEngine`** on Apple Silicon. PyTorch
    reports the missing engine at prepack time rather than at configuration
    time, which is a confusing place to discover it. Now selects `qnnpack`
    explicitly and reports a skip rather than crashing when no engine exists.

16. **Prometheus labels were about to be keyed on the raw request path**,
    which would have produced one label set per job id — unbounded cardinality
    that eventually fells the scraper. Labels use the route template.

**Verdict.** The service is deployable. It still has nothing worth deploying:
no real dataset numbers exist, and the export, drift baseline and dashboards
are all currently describing a model trained for two epochs on four synthetic
clips.

---

### 2026-10-06 · dataset tooling and preflight · all
**What.** The download helpers F1 promised but that did not exist, plus a
preflight check.

* `scripts/download_datasets.py` — automated for DFDC (Kaggle, resumable), and
  for the five form-gated sets it prints the exact steps, the expected layout
  and the follow-up commands. It does not try to automate around a licence
  agreement.
* `scripts/preflight.py` — checks the things that waste the most time when
  wrong: disk, device, dependency ABI, dataset presence, cache coverage, tree
  cleanliness, and the leakage gate.

**Finding.**

17. **torchaudio was ABI-incompatible with the installed torch** (2.11 against
    2.2.2) — collateral from pinning numpy back after onnxruntime pulled
    NumPy 2. It failed at *import* with a dlopen symbol error rather than at
    resolve time, and the test suite never caught it because nothing imports
    torchaudio directly. Preflight found it on its first run. The trio is now
    pinned together in pyproject, and a test asserts their versions match.

**Verdict.** Everything that can be built has been built. What remains needs
access approvals and GPU hours:

* FF++, Celeb-DF v2 and FakeAVCeleb are behind forms that take days.
* DFDC is instant and is the only instantly-available set with audio.
* The grid is ~198 GPU-hours and 0 have been run.
* **Disk is a real constraint**: 33 GB free on this machine against ~26 GB for
  FF++ + Celeb-DF caches alone, and DFDC needs 25 GB before preprocessing.

---

## Stage 1 — pending

Next: submit the FF++, Celeb-DF v2 and FakeAVCeleb access forms; download the
DFDC Kaggle sample; run `make manifest` with `--audit` to produce the audio
evidence table that settles the no-audio finding empirically.
