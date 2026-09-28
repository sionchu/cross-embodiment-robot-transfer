# Project handoff

## Objective
Complete the G0/E0 foundation before custom transfer experiments.

## Scope
Repository hygiene, pinned official GR00T N1.7/DROID interfaces and provider-agnostic
GPU execution. E0 is an inference sanity gate and needs 16 GB+ VRAM; E5 fine-tuning later
needs 40 GB+ VRAM. AGENTS.md remains canonical. The existing Colab notebook/CLI are
optional adapters. No simulator, training or E1 work.

## Acceptance criteria
G0: clean committed tree, no credentials/large payloads, ignore rules and source
licenses documented. E0: real Linux x86_64 NVIDIA CUDA evidence with at least 16 GB-class
VRAM, gated access, both official sample inference runs and exact code/model/environment provenance.

## Completed
- Inspected bootstrap HEAD ff6ec2a and the complete AGENTS.md; no user changes.
- Verified official upstream HEAD and model SHAs, dependency pins, CLI and sample
  metadata; recorded differences from the plan's older minimum-version notes.
- Added the scoped workflow and failure-aware result capture.
- E0 compact evidence now includes sanitized, allow-listed failure classification;
  unknown exceptions use UNKNOWN_FAILURE without raw messages or tracebacks.
- Colab HF_TOKEN preflight explicitly fails for missing or not-shared secrets;
  isolated HF_HOME and child-process-only token passing are preserved.
- Attempted the real Colab runtime gate on 2026-09-28 through the signed-in Aside
  Google session. The notebook requested A100 high-memory, but Colab explicitly
  reported the selected GPU type unavailable and allocated T4 instead.
- Stopped before bundle clone, GR00T installation, gated-model access or inference;
  no inference metrics were produced. Sanitized allocation evidence is committed under
  `artifacts/environment/2026-09-28-colab-a100-unavailable.json`.
- Follow-up Google One support confirmed on 2026-09-28 that the bundled Google AI Pro
  Colab CCU benefit is currently reported for Canada, Mexico and the United States, not
  South Korea; support stated that separate provisioning/escalation or a no-extra-cost
  South Korea path is not available. Sanitized regional-policy evidence is committed under
  `artifacts/environment/2026-09-28-colab-region-ineligible.json`.
- Retried E0 on a real Colab Tesla T4. Bundle and project-HEAD checks passed, and the GPU
  preflight reached the pinned dependency install. `uv sync --locked` then failed because
  the aarch64 torchcodec local wheel was still an unresolved Git LFS pointer. The workflow
  now pulls that wheel LFS object explicitly before sync; upstream lock/project files remain
  unmodified.

## Current checkpoint
G0 hygiene implementation and local checks complete.
E0 now reaches a clean real-T4 runtime with bundle/HEAD checks, 16 GB-class GPU preflight,
the pinned GR00T checkout, scoped LFS resolution, and `uv sync --locked` all passing.
Verified runtime stack: Tesla T4, torch 2.9.0+cu128, torchcodec 0.8.0, FFmpeg 6.1.1.
The first remaining blocker is authentication: Colab secret `HF_TOKEN` is missing or not
shared with the notebook, so Cosmos gated-access validation and both official inference runs
have not started yet.

## Decisions and reasons
- Use official standalone inference unmodified; pin code, lock and model snapshots.
- Use bundled DROID data, not a regenerated sample from a moving source dataset.
- Run offline after isolated downloads so the nested backbone cannot drift.
- Keep raw runtime logs ignored; review compact evidence before a results commit.
- Use the repository path recorded in AGENTS.md. The request's extra separator
  between 01 and _Robotics did not identify an existing directory.

## Verification evidence
- Real Colab T4 allocation reached the pinned install path; `nvidia-smi` reported Tesla T4
  with 15360 MiB VRAM. Bundle and project HEAD checks passed before installation.
- `python -m unittest discover -s tests -v`: 8 tests passed (the original five
  plus sanitized known/unknown failure evidence and mocked runtime coverage).
  Sensitive messages are excluded; successful metric evidence has no failure code.
  These are wrapper tests, not model inference.
- Python AST/compile checks passed for both modules and four Python Colab cells;
  Git Bash `bash -n` passed for the installation cell. Source JSON parsed.
- `git diff --check` and `git diff --cached --check`: passed; all four
  hardening changes were reviewed.
- `git check-ignore`: excluded representative secrets, weights, data, media,
  raw evidence and bundle paths. Added text files were all under 100 KB and no
  common HF/GitHub token or private-key patterns matched (limited pattern check).
- Official source links and immutable revisions are in references/.
- Colab runtime evidence: the fresh notebook visibly targeted `A100 high-memory`;
  Colab then reported the selected GPU type unavailable and connected a T4. The
  resource panel also showed no active subscription/compute units. Execution was
  stopped and the T4 runtime was terminated. Raw screenshot evidence remains outside
  Git; its SHA-256 is recorded in the sanitized JSON artifact.

## Not executed
HF/Cosmos gated-access validation, both GR00T checkpoint inference runs, and all E1+ work.
The Linux/T4 dependency installation now completes successfully. No public push, release,
or reference-repository modification.

## Blockers
E0 compute/install is no longer blocked. A measured free-tier T4 reaches a complete pinned
GR00T environment. The active blocker is Hugging Face authentication: the browser is logged
out, no saved Hugging Face credential was found in Aside Vault, and Colab has no shared
`HF_TOKEN` secret. The connected Hugging Face app is authenticated as `JE92`, but that OAuth
credential cannot be exported into Colab through the available connector tools. Cosmos gated
access has therefore not yet been tested.
For E5 fine-tuning, 40 GB+ VRAM will still be required. Google AI Pro's bundled Colab CCU
benefit is not available to this South Korea account; support said it cannot be separately
provisioned or escalated. Do not use VPN/region spoofing or false account-country workarounds.

## Modified files
.gitignore, .gitattributes, README.md, references/source_manifest.json,
references/gr00t_contract.md, scripts/colab/README.md, scripts/colab/e0.py,
tests/test_e0.py, HANDOFF.md,
artifacts/environment/2026-09-28-colab-a100-unavailable.json, and
artifacts/environment/2026-09-28-colab-region-ineligible.json.

## Next concrete action
Resume at Hugging Face authentication, not environment setup. Log in to the `JE92` Hugging
Face account, review/accept the standard NVIDIA gate for `nvidia/Cosmos-Reason2-2B`, create a
minimal read token, save it as Colab secret `HF_TOKEN`, and grant this notebook access. Then
rerun only cell 5 / the E0 runner in the already-prepared notebook if the T4 runtime is still
alive; otherwise rerun the notebook from the top. Do not start E1 until both official model
inference runs finish and `e0.json` says PASS. Reserve 40 GB+ GPU access for E5 fine-tuning.
