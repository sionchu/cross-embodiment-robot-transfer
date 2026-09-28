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
E0 runtime was attempted on a real T4 and remains incomplete. After correcting the earlier
A100-only gate, the T4 reached dependency installation successfully. The current retry fix
resolves the pinned aarch64 torchcodec LFS wheel before `uv sync --locked`. HF gated-access
check and model inference have not been reached yet.

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
A complete Linux dependency installation, HF/Cosmos gated-access validation, both GR00T
checkpoint inference runs, and all E1+ work. The first T4 install attempt stopped during
`uv sync --locked` at the unresolved LFS wheel issue described above. No public push,
release, or reference-repository modification.

## Blockers
E0 is no longer blocked on A100/40 GB compute. It can run on an authorized Linux x86_64
NVIDIA CUDA GPU with 16 GB+ VRAM, including the measured free-tier T4. The first T4 install
failure had a concrete local cause (unresolved LFS wheel) and is fixed in the workflow. The
next expected blocker is Hugging Face/Cosmos gated access because that has not yet been tested.
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
Rerun E0 on the Colab free-tier T4 (or any other authorized Linux NVIDIA GPU with 16 GB+
VRAM). The official Colab CLI is installed in WSL2 and can be used for terminal lifecycle
and `colab usage` evidence after one-time OAuth authorization. Do not start E1. Once
`nvidia-smi` and the runner confirm the 16 GB inference gate, continue to installation,
gated-access validation and both official checkpoint inference runs. Reserve 40 GB+ GPU
access for E5 fine-tuning.
