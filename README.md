# Cross-embodiment robot transfer

An evidence-first study of GR00T adaptation to a simulated UR10e insertion task.
The canonical scope and gate plan are in [AGENTS.md](AGENTS.md).

Current slice: repository hygiene and the official N1.7 DROID sanity workflow.
A real Colab E0 allocation attempt on 2026-09-28 requested A100 high-memory, but
Colab allocated T4 instead, so the run stopped at `GPU_NOT_A100` before install or
inference. Both checkpoint results remain **unverified**. No custom fine-tuning or
simulator implementation may start before E0 passes.

- [Colab A100 E0 workflow](scripts/colab/README.md)
- [Pinned sources and environment](references/source_manifest.json)
- [Verified upstream contract](references/gr00t_contract.md)
- [Execution checkpoint](HANDOFF.md)

Local checks (Python 3.12, standard library only):

```text
python -m unittest discover -s tests -v
```

Weights, datasets, caches and raw run folders stay outside Git. Review compact
JSON evidence before copying it into `artifacts/environment/` for a later evidence
commit. Public push and release remain gated by AGENTS.md.
