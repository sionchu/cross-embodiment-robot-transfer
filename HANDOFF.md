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

## Current checkpoint
G0 hygiene implementation and local checks complete; preserve a clean tree after commit.
E0 runtime remains incomplete/pending: actual A100 inference has not been run,
and no Colab A100 inference evidence exists yet.

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

## Not executed
Linux dependency install, gated access with owner credentials, A100 GPU inference,
and all E1+ work. No public push, release, or reference-repository modification.

## Blockers
No connected Colab A100 runtime or owner-approved HF secret is available in this
session. The supplied Colab workflow is ready for the owner to run; runtime
compatibility and metrics are unverified until its evidence is returned.

## Modified files
.gitignore, .gitattributes, README.md, references/source_manifest.json,
references/gr00t_contract.md, scripts/colab/README.md, scripts/colab/e0.py,
tests/test_e0.py, HANDOFF.md.

## Next concrete action
Run the Colab workflow on A100 and review both checkpoint logs and result JSON.
If environment installation fails, preserve the log and diagnose the actual error
before adjusting a pin. E0 remains the earliest incomplete experiment.
