"""Diagnostic driver (outside repo): FIRe source-collapse localization.

mode fire0    : identical FIRe code path, 0 source optimization steps (code-path control)
mode fire1000 : original FIRe recipe (1000 source steps), unchanged
Both: probe input-sensitivity (std of predictions over 256 fixed target frames)
right after source phase and during target training. act.py is not modified.
"""
import sys, json, argparse, hashlib
from pathlib import Path
ROOT = Path(r'D:\Workspace\01_Robotics\cross-embodiment-robot-transfer')
sys.path.insert(0, str(ROOT / 'scripts/local'))
import numpy as np, torch
import act

PROBE_STEPS = {1, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000}
LOG = None
TARGET = {}


def probe(model, data, tag, step):
    idx = np.linspace(0, len(data) - 1, 256).astype(int)
    was_training = model.training
    model.eval(); outs = []
    with torch.no_grad():
        for k in range(0, len(idx), 64):
            ids = idx[k:k + 64]
            batch = {'observation.state': data.states[ids].cuda(),
                     act.VIDEO: torch.stack([data.images[i] for i in ids]).cuda()}
            outs.append(model.predict_action_chunk(batch)[:, 0].float().cpu())
    if was_training: model.train()
    pred = torch.cat(outs); gt = data.actions[idx]
    rec = {'tag': tag, 'step': step,
           'pred_std_norm': pred.std(0).tolist(), 'gt_std_norm': gt.std(0).tolist(),
           'l1_norm': (pred - gt).abs().mean(0).tolist(), 'pred_mean_norm': pred.mean(0).tolist()}
    with LOG.open('a') as f: f.write(json.dumps(rec) + '\n')
    print('PROBE', tag, step, 'std', [round(x, 4) for x in rec['pred_std_norm']],
          'l1', round(float(np.mean(rec['l1_norm'])), 4), flush=True)


_orig_optimize = act.optimize

def optimize(model, loader, steps, path, phase='target'):
    """Copy of act.optimize loop plus read-only probes (no RNG use in eval forward)."""
    import csv, time
    data = TARGET.get('data')
    optimizer = torch.optim.AdamW(model.get_optim_params(), lr=model.config.optimizer_lr,
                                  weight_decay=model.config.optimizer_weight_decay)
    start = time.perf_counter(); losses = []; iterator = iter(loader)
    if phase == 'target' and data is not None: probe(model, data, phase, 0)
    with path.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['step', 'loss', 'l1_loss', 'kld_loss', 'grad_norm']); w.writeheader()
        for step in range(1, steps + 1):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            batch = {k: v.cuda() for k, v in batch.items()}
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, metrics = model(batch)
            if not torch.isfinite(loss): raise RuntimeError('NONFINITE_LOSS')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            v = float(loss.detach()); losses.append(v)
            w.writerow({'step': step, 'loss': v, **metrics, 'grad_norm': float(norm)})
            if data is not None and step in PROBE_STEPS: probe(model, data, phase, step)
            if step == 1 or step % 500 == 0: print(json.dumps({'phase': phase, 'step': step, 'loss': v}), flush=True)
    torch.cuda.synchronize()
    return losses, time.perf_counter() - start, batch

act.optimize = optimize
_orig_source = act.source_pretrain


def source_pretrain(model, target_data, args):
    TARGET['data'] = target_data
    probe(model, target_data, 'before_source', 0)
    if args.source_steps == 0:
        head = {k: v.detach().clone() for k, v in model.model.action_head.state_dict().items()}
        cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state_all()
        model.model.action_head.load_state_dict(head, strict=True)
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
        report = {'mechanism': 'DIAGNOSTIC fire0: source path with 0 optimization steps'}
    else:
        report = _orig_source(model, target_data, args)
    probe(model, target_data, 'after_source_head_restored', 0)
    return report

act.source_pretrain = source_pretrain

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['fire0', 'fire1000'], required=True)
    p.add_argument('--name', required=True); a = p.parse_args()
    args = argparse.Namespace(dataset=ROOT / 'datasets/ur10e_50', output=ROOT / 'artifacts/training' / a.name,
        dataset_report=ROOT / 'artifacts/datasets/ur10e_50.json', chunk_size=1, sampling='precision-balanced',
        lr=1e-4, backbone_lr=1e-5, backbone_weights='imagenet', episodes=None,
        source_prior=(ROOT / 'artifacts/datasets/fire_motion_prior/manifest.json').resolve(),
        source_steps=0 if a.mode == 'fire0' else 1000, steps=5000, batch_size=8, seed=20260929)
    LOG = Path(r'D:\Workspace\_claude_scratch') / f'{a.name}_probe.jsonl'
    if LOG.exists(): LOG.unlink()
    act.train(args)
