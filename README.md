# Cross-embodiment robot transfer

An evidence-first study of GR00T adaptation to a simulated UR10e insertion task.
The canonical scope and gate plan are in [AGENTS.md](AGENTS.md).

Current slice: repository hygiene and the official N1.7 DROID sanity workflow.
Official GR00T N1.7 guidance requires 16 GB+ VRAM for inference and 40 GB+ for later
fine-tuning. A free-tier Tesla T4 completed the pinned environment and gated-access
checks and reached the official inference stage, but E0 has **not passed**: the latest
run ended with `OFFICIAL_INFERENCE_FAILED`. The next run preserves a sanitized per-model
log so the failure can be diagnosed without exposing credentials. No custom fine-tuning
or simulator implementation may start before E0 passes.

- [Colab / cloud GPU E0 workflow](scripts/colab/README.md)
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
