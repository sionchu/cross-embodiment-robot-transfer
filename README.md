# Cross-embodiment robot transfer

An evidence-first study of GR00T adaptation to a simulated UR10e insertion task.
The canonical scope and gate plan are in [AGENTS.md](AGENTS.md).

Current priority: deterministic UR10e insertion -> 10 successful expert demonstrations
-> one GR00T LeRobot v2 dataset -> local ACT GPU training -> checkpoint-driven
closed-loop held-out evaluation. GR00T N1.7 remains the foundation-model integration
target. The official UR10e environment and deterministic expert smoke now run locally.
Seeds 0, 0 and 1 succeeded; repeated seed results matched exactly. A deliberate
15 mm miss was rejected as a physical collision. See [measured G1 evidence](artifacts/simulation/g1_smoke.json)
and [local execution/action contract](scripts/local/README.md). Ten successful expert
demonstrations now form one canonical dataset: 487 synchronized RGB/state/action
frames. Schema, corruption checks, the pinned official GR00T episode loader and
its action normalization round-trip passed; see [G2 evidence](artifacts/datasets/validation.json).
Maintained LeRobot ACT has now trained from scratch on the RTX 4070 Ti for 1,000
steps and reloaded its real checkpoint successfully. See [G3 training evidence](artifacts/training/act-10-001/summary.json)
and [per-step loss](artifacts/training/act-10-001/loss.csv). Learned-policy closed-loop
evaluation has not yet run; the training result does not establish insertion success.

E0 is an independent GR00T runtime gate, not a prerequisite for local target learning.
The existing project record reports a T4 pinned installation and gated access followed
by `OFFICIAL_INFERENCE_FAILED`; no E0 PASS artifact is committed. Colab entitlement
and allocation work is deferred by owner decision. The recorded South Korea limitation
concerns bundled Google AI Pro compute benefits, not a claim that all Colab service is
unavailable. Official GR00T inference requires 16 GB-class VRAM and fine-tuning 40 GB+;
the local RTX 4070 Ti 12 GB is assigned to Isaac Sim and ACT. GR00T training and
closed-loop GR00T results remain unexecuted.

- [Colab / cloud GPU E0 workflow](scripts/colab/README.md)
- [Pinned sources and environment](references/source_manifest.json)
- [Verified upstream contract](references/gr00t_contract.md)
- [Execution checkpoint](HANDOFF.md)

Local checks after the [isolated ML setup](scripts/local/README.md) and dataset generation:

```text
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Weights, datasets, caches and raw run folders stay outside Git. Review compact
JSON evidence before copying it into `artifacts/environment/` for a later evidence
commit. Public push and release remain gated by AGENTS.md.
