# Dataset card

None of these datasets is redistributed in this repository. Each requires its
own access request and carries its own licence; all are released for research
use and all are cited in the paper.

## Summary

| Dataset | Role | Audio | Access | Scale |
|---|---|---|---|---|
| FaceForensics++ | primary training set | **none** | request form | ~1,000 source videos × 4 methods × 3 compressions |
| Celeb-DF v2 | cross-dataset test | **none / not aligned** | request form | ~5,600 fakes, higher quality |
| DFDC | AV training set | **yes** | instant (Kaggle) | large, varied, includes audio fakes |
| FakeAVCeleb | cross-dataset AV test | **yes** | request form | 4 real/fake video×audio combinations |
| DeeperForensics-1.0 | robustness test | none | request form | perturbation suite |
| LAV-DF | backup AV test | yes | request form | localised AV forgeries |

**Submit the FF++, Celeb-DF and FakeAVCeleb forms on day one** — approval takes
several days. DFDC is instant and carries audio, so Stage 1 is never blocked.

```bash
make datasets          # what each one is, and how to get it
make get-dfdc          # the one that is automated (Kaggle)
make preflight         # is this machine ready: disk, GPU, deps, data
```

`scripts/download_datasets.py <name>` prints the exact steps, the expected
directory layout and the commands to run afterwards for the form-gated sets.
It does not try to automate around a licence agreement.

## The audio finding

FF++ and Celeb-DF v2 do not carry usable audio. This is verified empirically,
not assumed: `python -m ddetect.data.build_manifest --dataset ffpp --audit`
runs `ffprobe` over every video and prints the count.

The consequence is structural. The audio-visual objective cannot be evaluated
on an FF++ → Celeb-DF pairing at all, so it moves to DFDC → FakeAVCeleb, and
the project runs two protocols sharing one model. See the README.

## Splits

Identity-disjoint by construction, and verified by `tests/test_leakage.py`:

- **FF++** uses the official `train/val/test.json` identity splits. A forgery
  `000_003` involves *both* identities, so its grouping key is the sorted pair
  — using only the target would let 000's face appear on both sides.
- **Celeb-DF** honours `List_of_testing_videos.txt` exactly.
- **DFDC** splits by part folder; a fake inherits its `original` video's
  identity so the two cannot be separated.
- **FakeAVCeleb** is test-only by default, so nothing from it can enter a
  training loader by accident.

Datasets without an official split use a deterministic hash of the identity,
never of the filename.

## Integrity checks run before any training

- identity overlap across splits (fails the build)
- perceptual-hash near-duplicates within a dataset across splits (fails)
- near-duplicates *across* datasets (warns — this compromises a
  train-on-one/test-on-other claim for those videos specifically)
- label/method consistency
- missing or unreadable files

## Known biases and caveats

- **Source skew.** FF++ and Celeb-DF are both built from public video of
  public figures, which is not representative of ordinary uploads.
- **Compression.** FF++ ships raw/c23/c40; most research reports c23, which is
  cleaner than typical social-media video. The degradation augmentation and
  the CRF breakdown exist because of this.
- **Demographics.** Only FakeAVCeleb carries group metadata. Performance
  across groups on the other three is unmeasured, and we do not infer
  protected attributes to fill the gap.
- **Generator coverage.** Each dataset covers a handful of forgery tools from
  the year it was built. Methods released since are represented by none of
  them — which is the entire premise of the project.
- **Consent.** These datasets contain real people's faces, collected under the
  terms each dataset documents. They are used for research only; no identity
  is enrolled, and no biometric template is stored.
