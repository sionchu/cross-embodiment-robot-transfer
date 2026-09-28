# Project handoff

## Objective
Complete the G0/E0 foundation before custom transfer experiments.

## Scope
Repository hygiene, pinned official GR00T N1.7/DROID interfaces and Colab A100
sanity execution. AGENTS.md remains canonical. No simulator, training or E1 work.

## Acceptance criteria
G0: clean committed tree, no credentials/large payloads, ignore rules and source
licenses documented. E0: real A100 evidence, gated access, both official sample
inference runs and exact code/model/environment provenance.

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
  no inference metrics were produced. Sanitized failure evidence is committed under
  `artifacts/environment/2026-09-28-colab-a100-unavailable.json`.

## Current checkpoint
G0 hygiene implementation and local checks complete.
E0 runtime was attempted but remains incomplete. The first allocation surfaced T4 instead
of the requested A100, but follow-up account inspection identified the upstream blocker:
the active Google AI Pro 5TB subscription is not being provisioned into Colab. Colab
shows 0 compute units even though the plan is officially eligible for 200 CCU. No GR00T
install, HF gated-access check or inference was started.

## Decisions and reasons
- Use official standalone inference unmodified; pin code, lock and model snapshots.
- Use bundled DROID data, not a regenerated sample from a moving source dataset.
- Run offline after isolated downloads so the nested backbone cannot drift.
- Keep raw runtime logs ignored; review compact evidence before a results commit.
- Use the repository path recorded in AGENTS.md. The request's extra separator
  between 01 and _Robotics did not identify an existing directory.

## Verification evidence
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
Linux dependency installation, HF/Cosmos gated-access validation, both GR00T
checkpoint inference runs, and all E1+ work. No public push, release, or
reference-repository modification.

## Blockers
The current blocker is Colab entitlement provisioning, not a decision to buy more compute.
The same Google account has a verified active Google AI Pro (5 TB) subscription billed
directly through Google Play, but Colab shows 0 compute units and no included-benefit
claim/activate control. Official Google documentation says eligible Google AI Pro members
receive 200 CCU. Do not purchase Pay As You Go or Colab Pro as a workaround. Once the
included entitlement appears, request A100 again and keep the E0 A100 gate unchanged.
HF secret/gated access has not yet been tested because execution stopped earlier.

## Modified files
.gitignore, .gitattributes, README.md, references/source_manifest.json,
references/gr00t_contract.md, scripts/colab/README.md, scripts/colab/e0.py,
tests/test_e0.py, HANDOFF.md,
artifacts/environment/2026-09-28-colab-a100-unavailable.json, and
artifacts/environment/2026-09-28-google-ai-pro-colab-entitlement-missing.json.

## Next concrete action
Resolve the missing Google AI Pro → Colab 200 CCU entitlement on the existing account;
do not purchase additional Colab compute. Then rerun the same pinned E0 workflow and
request A100. Do not start E1. Once A100 is confirmed by `nvidia-smi`, continue to
installation, gated-access validation and both official checkpoint inference runs.
