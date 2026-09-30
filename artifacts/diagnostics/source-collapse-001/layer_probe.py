"""Which ACT layer becomes input-invariant during FIRe source pretraining? (outside repo)"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(r'D:\Workspace\01_Robotics\cross-embodiment-robot-transfer')
sys.path.insert(0, str(ROOT / 'scripts/local'))
import numpy as np, torch, random
import act
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature

seed = 20260929
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
config = ACTConfig(device='cuda', chunk_size=1, n_action_steps=1,
    input_features={act.VIDEO: PolicyFeature(FeatureType.VISUAL, (3, 240, 320)),
                    'observation.state': PolicyFeature(FeatureType.STATE, (6,))},
    output_features={'action': PolicyFeature(FeatureType.ACTION, (6,))},
    pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1', optimizer_lr=1e-4, optimizer_lr_backbone=1e-5)
model = act.LeRobotACT(config).cuda()
target = act.CanonicalDataset(ROOT / 'datasets/ur10e_50', 1)
source = act.SourceMotionDataset((ROOT / 'artifacts/datasets/fire_motion_prior/manifest.json').resolve(), target.statistics, 1)
m = model.model
names = {'backbone': m.backbone, 'state_proj': m.encoder_robot_state_input_proj,
         'encoder': m.encoder, 'decoder': m.decoder}
out = {}

def probe(tag, data):
    idx = np.linspace(0, len(data) - 1, 128).astype(int)
    acts = {k: [] for k in names}; hooks = []
    for k, mod in names.items():
        def h(mod, i, o, k=k):
            o = o['feature_map'] if isinstance(o, dict) else o
            acts[k].append(o.detach().float().flatten(1).cpu() if o.dim() != 3 or k == 'backbone' else o.detach().float().transpose(0, 1).flatten(1).cpu())
        hooks.append(mod.register_forward_hook(h))
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), 32):
            ids = idx[s:s + 32]
            model.predict_action_chunk({'observation.state': data.states[ids].cuda(),
                                        act.VIDEO: torch.stack([data.images[i] for i in ids]).cuda()})
    for h in hooks: h.remove()
    res = {}
    for k, v in acts.items():
        x = torch.cat(v)
        res[k] = {'across_input_std_mean': float(x.std(0).mean()), 'abs_mean': float(x.abs().mean())}
    out[tag] = res
    print(tag, json.dumps({k: round(v['across_input_std_mean'], 6) for k, v in res.items()}), flush=True)

probe('init_target', target); probe('init_source', source)
loader = torch.utils.data.DataLoader(source, batch_size=8, shuffle=True, num_workers=0,
                                     generator=torch.Generator().manual_seed(seed + 1))
opt = torch.optim.AdamW(model.get_optim_params(), lr=1e-4, weight_decay=config.optimizer_weight_decay)
it = iter(loader)
for step in range(1, 1001):
    try: b = next(it)
    except StopIteration: it = iter(loader); b = next(it)
    b = {k: v.cuda() for k, v in b.items()}
    model.train(); opt.zero_grad(set_to_none=True)
    loss, met = model(b); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
    if step in (10, 50, 200, 1000):
        print('step', step, 'loss', round(float(loss), 4), met, flush=True)
        probe(f'src{step}_target', target); probe(f'src{step}_source', source)
Path(r'D:\Workspace\_claude_scratch\layer_probe.json').write_text(json.dumps(out, indent=1))
