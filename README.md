# Cross-embodiment robot transfer

An evidence-first study of GR00T adaptation to a simulated UR10e insertion task.
The canonical scope and gate plan are in [AGENTS.md](AGENTS.md).

Current priority: deterministic UR10e insertion -> 10 successful expert demonstrations
-> one GR00T LeRobot v2 dataset -> local ACT GPU training -> checkpoint-driven
closed-loop held-out evaluation. GR00T N1.7 remains the foundation-model integration
target. The official UR10e environment and deterministic expert smoke now run locally.
Seeds 0, 0 and 1 succeeded; repeated seed results matched exactly. A deliberate
15 mm miss was rejected as a physical collision. See [measured G1 evidence](artifacts/simulation/g1_smoke.json)
and [local execution/action contract](scripts/local/README.md). Demonstration dataset
export, ACT training and learned-policy evaluation have not yet run.

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

Local checks (Python 3.12 with NumPy and SciPy; E0-only tests use the standard library):

```text
python -m unittest discover -s tests -v
```

Weights, datasets, caches and raw run folders stay outside Git. Review compact
JSON evidence before copying it into `artifacts/environment/` for a later evidence
commit. Public push and release remain gated by AGENTS.md.
