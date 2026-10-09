# Threat model

The service accepts arbitrary video from the internet and hands it to ffmpeg,
OpenCV and PyTorch. Every upload is treated as hostile.

## Assets

| Asset | Why it matters |
|---|---|
| Uploaded video | Contains a real person's face. Must not leak or persist beyond its retention window. |
| Verdicts and explanations | Statements about whether a person's video is fake. Must be auditable and must not overclaim. |
| Model checkpoints | Reproducibility of published numbers. |
| The host | Must not become a pivot into the network. |

## Attacks and mitigations

### 1. Malicious file disguised as video
**Attack.** Upload a PE/ELF binary, a shell script or a polyglot named `.mp4`.
**Mitigation.** Container magic bytes are sniffed; the extension is never
trusted. `tests/test_security.py` plants a renamed executable and asserts
rejection.

### 2. Resource exhaustion
**Attack.** A declared-small / actually-huge upload; a multi-hour clip; a
decompression bomb; many concurrent requests.
**Mitigation.** The size cap is enforced **per chunk while streaming to disk**,
not from a client-supplied `Content-Length`. Duration, resolution and codec are
checked after probing. The worker pool is deliberately small so requests queue
rather than thrash. Container limits: 4 GB memory, 256 pids.

### 3. SSRF through ffmpeg
**Attack.** A crafted container — an HLS playlist, a `concat` demuxer script —
that makes ffmpeg issue outbound HTTP requests from inside the network.
**Mitigation.** Every ffmpeg invocation passes `-protocol_whitelist file` and
`-nostdin`, with no shell. A test asserts the flag is present on the code path
that actually runs, not just in a helper.

### 4. Path traversal
**Attack.** `../../etc/passwd` as a storage key or artifact name.
**Mitigation.** The client's filename never reaches the filesystem; a random
32-hex key does. Artifact names are regex-constrained and the resolved path is
checked to be inside the artifact root. Parameterised tests cover ten hostile
keys.

### 5. Data exfiltration / retention
**Attack.** Uploads accumulate; someone later obtains the store.
**Mitigation.** The upload is deleted as soon as it has been analysed. Derived
artefacts expire on a retention timer, and `DELETE /v1/analyses/{id}` removes
them immediately — tested, not promised. No biometric templates or identities
are persisted.

### 6. Prompt injection into the explainer
**Attack.** A video whose filename or metadata carries text intended to steer
A7's output.
**Mitigation.** A7 never sees the filename, the raw file, or any user-supplied
string. Its input is a fixed-schema payload of numbers computed by the
detector. Its output is then checked programmatically against that payload;
anything not grounded in it is discarded.

### 7. Misuse of the verdict
**Attack.** Someone uses a "likely manipulated" output to accuse a real person.
**Mitigation.** Partly social, partly technical: three-state output, a
mandatory disclaimer in every response, prominent limitations in the UI, an
explicit out-of-scope list in the model card, and an OOD flag that fires when
the input is unlike anything the model was trained on. The API has no response
shape that renders a bare binary.

### 8. Supply chain
**Mitigation.** `pip-audit` and an SBOM in CI, a committed lockfile, and
`scripts/scope_audit.py` failing the build if a generative-media dependency
appears.

## Out of scope

- Adversarial robustness of the detector itself. Not claimed; the FGSM/PGD grid
  documents the vulnerability rather than defending against it.
- Authenticated multi-tenancy. The default deployment is a single-tenant demo
  with optional API keys.
- Protecting against a hostile operator with host access.

## Residual risk

The largest residual risk is not technical: it is that a user reads a
probabilistic estimate as a determination. Every design decision in the output
path — three states, calibration, abstention, the OOD flag, the disclaimer —
exists to reduce it, and none of them eliminates it.
