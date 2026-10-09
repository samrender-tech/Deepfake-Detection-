# Model card — AVFORGE deepfake detector

Last updated: 2026-10-06 · Status: **research prototype, not for deployment**

## What it does

Takes a short video, samples frames, locates and crops the largest consistent
face track, and estimates the probability that the face or voice was
manipulated. Where audio exists, it additionally scores the speech and the
agreement between lip motion and speech.

Outputs **three** states, never two:

| Outcome | Meaning |
|---|---|
| Likely authentic | Calibrated probability below the decision threshold, outside the abstention band |
| Likely manipulated | Above the threshold, outside the abstention band |
| **Inconclusive** | Inside the conformal abstention band, or flagged as unlike the training distribution |

"Inconclusive" is a first-class answer, not a failure. At cross-dataset
accuracy it is frequently the only honest one.

## Intended use

Research into detector generalisation. Coursework and teaching. A starting
point for further work on audio-visual forgery detection.

## Out-of-scope use — do not do these

This model must **not** be used to make or support decisions about a person,
including:

- legal or forensic proceedings, or evidence of any kind
- journalism or fact-checking, without independent corroboration
- employment, admissions, immigration, insurance or credit decisions
- content moderation that results in automated enforcement against an account
- any use where a false "manipulated" would defame someone, or a false
  "authentic" would launder a real deepfake

A false positive accuses a real person of fabricating a video. A false
negative certifies a forgery. Both are serious; the model is not accurate
enough on unseen manipulation methods for either risk to be acceptable in a
consequential decision.

## Performance

Numbers are produced by `experiments/aggregate.py` and are **not** transcribed
by hand. Each row records the number of seeds; headline rows are mean ± std
over three.

| Experiment | Train | Test | AUC | Status |
|---|---|---|---|---|
| Baseline A | FF++ c23 | FF++ c23 | _pending_ | target ≥0.95 (reproduce literature) |
| **Baseline B** | FF++ c23 | Celeb-DF, DFDC | _pending_ | **the headline: the honest drop** |
| V-full | FF++ c23 | Celeb-DF, DFDC | _pending_ | + SBI, degradation, frequency, temporal |
| AV-proposed | DFDC | FakeAVCeleb | _pending_ | audio-visual |
| Full system | FF++ + DFDC | all four | _pending_ | one checkpoint |

**The cross-dataset gap is the result.** A small gap is either a genuine
generalisation win or a leak — `pytest tests/test_leakage.py` must pass before
any such number is believed.

### Breakdowns reported

Per forgery method · per compression level (CRF 23/32/40) · per face-size
quartile · with and without audio · calibration (ECE, reliability diagram) ·
risk–coverage at 70/80/90% coverage · OOD detection AUROC.

## Known failure modes

- **Unseen manipulation methods.** The central limitation; the OOD flag exists
  to surface it rather than hide it.
- **Small or heavily compressed faces.** Faces below ~48px are dropped; faces
  below ~80px are measurably worse.
- **Silent video.** The audio and sync streams are masked off; accuracy falls
  back to the visual stream alone, which is the harder setting.
- **Multiple speakers.** The face tracker follows one subject; a manipulated
  second face may be missed entirely.
- **Non-frontal faces, occlusion, heavy motion blur.** Reduced face-detection
  confidence, reflected in the reliability gate.
- **Adversarial input.** No robustness to deliberate attack is claimed. The
  evaluation includes an FGSM/PGD grid purely to document the vulnerability.
- **Cross-domain miscalibration.** The calibrator is fitted on source-domain
  validation data; its reliability degrades on a shifted test set, and the
  measured degradation is reported rather than hidden.

## Demographic performance

FakeAVCeleb carries race and gender directory metadata, so per-group
performance is reported where sample sizes permit. FF++, Celeb-DF and DFDC do
not carry reliable demographic labels, so **per-group performance on those sets
is unknown**. We do not infer protected attributes to fill that gap. Readers
should assume performance varies across groups by an unmeasured amount.

## Training data

FaceForensics++ · Celeb-DF v2 · DFDC · FakeAVCeleb. See
[DATASET_CARD.md](DATASET_CARD.md). Self-Blended Images additionally generates
training fakes from real frames by image-space blending — no generative model
is involved (see Scope below).

## Scope: a detector, not a generator

No synthetic media is created by this project. The two places that manipulate
pixels or audio — Self-Blended Images, and the red-team agent — operate on
*existing real media only*: self-blending a frame with a warped copy of itself,
re-encoding, and splicing real audio onto a different real video. No
generative model, face-swap tool or voice cloner is trained, run or shipped.
`scripts/scope_audit.py` enforces this in CI.

## Privacy

Faces are processed, never enrolled. No biometric templates or identities are
persisted. Uploads are deleted as soon as they have been analysed; derived
artefacts expire on a retention timer (default 24h) and
`DELETE /v1/analyses/{id}` removes them immediately. Verdicts are logged with
their model version so a disputed result can be reconstructed and contested.

## The explanation text

The plain-language summary shown with each verdict is produced either by a
language model under strict grounding constraints, or by a deterministic
template. Which one is shown is labelled in the UI. The model may reference
only numbers present in the detector's own output; any invented quantity,
certainty language, or statement about who is in the video or why causes the
output to be discarded in favour of the template. See
[AGENT_POLICY.md](AGENT_POLICY.md).

## Monitoring in production

The service reports whether it is still being used on the kind of input it was
calibrated for. `/v1/drift` gives the Population Stability Index of the live
score distribution against the validation baseline, the out-of-distribution
flag rate, and the fraction of uploads with no detectable face.

A rising OOD rate or a PSI above 0.25 means the calibration and the abstention
band were fitted on a distribution that no longer matches the traffic — the
probabilities should not be trusted until that is understood. A falling
abstention rate is also worth investigating: the band firing less often is more
likely a bug than an improvement.

## Contact

Raise an issue in the repository. For a disputed verdict, include the job id;
the audit log can reconstruct what was served.
