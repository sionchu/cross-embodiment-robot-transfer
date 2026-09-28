# Cross-embodiment robot transfer

An evidence-first study of GR00T adaptation to a simulated UR10e insertion task.
The canonical scope and gate plan are in [AGENTS.md](AGENTS.md).

Current slice: repository hygiene and the official N1.7 DROID sanity workflow.
A 2026-09-28 Colab attempt allocated T4 after A100 was unavailable. That is no longer
a blocker: official GR00T N1.7 guidance requires only 16 GB+ VRAM for inference, while
40 GB+ is required later for fine-tuning. The project will therefore rerun E0 on the
measured T4-class runtime and reserve 40 GB+ compute for E5. Both checkpoint results
remain **unverified**. No custom fine-tuning or simulator implementation may start
before E0 passes.

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
