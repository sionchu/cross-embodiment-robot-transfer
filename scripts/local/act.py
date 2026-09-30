"""Maintained LeRobot ACT on the single canonical GR00T v2 dataset."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time

import av
import numpy as np
import pandas as pd
from PIL import Image
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy as LeRobotACT

from dataset import ROOT, VIDEO, jsonl, validate, write_json

IMAGE_SIZE=(320,240)
IMAGE_MEAN=torch.tensor([.485,.456,.406])[:,None,None]
IMAGE_STD=torch.tensor([.229,.224,.225])[:,None,None]


def image_tensor(rgb, expected_shape=(480,640,3)):
    rgb=np.asarray(rgb)
    if rgb.shape!=expected_shape or rgb.dtype!=np.uint8:
        raise ValueError('INVALID_RGB')
    resized=np.array(Image.fromarray(rgb).resize(IMAGE_SIZE,Image.Resampling.BILINEAR))
    return (torch.from_numpy(resized).permute(2,0,1).float()/255-IMAGE_MEAN)/IMAGE_STD


class CanonicalDataset(torch.utils.data.Dataset):
    """Runtime adapter only: no second dataset or files are produced."""
    def __init__(self, root, horizon):
        self.horizon=horizon
        self.statistics=json.loads((root/'meta/stats.json').read_text())
        self.states=[]; self.actions=[]; self.images=[]; self.ends=[]
        for episode in jsonl(root/'meta/episodes.jsonl'):
            i=episode['episode_index']
            frame=pd.read_parquet(root/f'data/chunk-000/episode_{i:06d}.parquet')
            self.states.extend(np.stack(frame['observation.state']))
            self.actions.extend(np.stack(frame.action))
            with av.open(str(root/f'videos/chunk-000/{VIDEO}/episode_{i:06d}.mp4')) as container:
                decoded=[image_tensor(f.to_ndarray(format='rgb24')) for f in container.decode(video=0)]
            if len(decoded)!=len(frame):
                raise ValueError('DECODE_LENGTH_MISMATCH')
            self.images.extend(decoded)
            self.ends.extend([len(self.states)]*len(frame))
        self.states=self.normalize(torch.tensor(np.asarray(self.states)),'observation.state')
        self.actions=self.normalize(torch.tensor(np.asarray(self.actions)),'action')

    def normalize(self, values, key):
        s=self.statistics[key]
        return (values-torch.tensor(s['mean']))/torch.tensor(s['std']).clamp_min(1e-6)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        ids=torch.arange(index,index+self.horizon)
        pad=ids>=self.ends[index]
        ids=ids.clamp(max=self.ends[index]-1)
        return {'observation.state':self.states[index],VIDEO:self.images[index],
                'action':self.actions[ids],'action_is_pad':pad}


class SourceMotionDataset(CanonicalDataset):
    """Actual FIRe RGB plus derived pose transitions; source-only feature objective."""
    def __init__(self, manifest, target_statistics, horizon):
        from fire import CACHE, DATASET, REVISION
        from insertion import CONFIG
        self.manifest=json.loads(manifest.read_text(encoding='utf-8'))
        if (self.manifest['dataset']!=DATASET or self.manifest['revision']!=REVISION
                or self.manifest['target_task']!=CONFIG):
            raise ValueError('SOURCE_PRIOR_PROVENANCE_MISMATCH')
        self.horizon=horizon
        self.statistics={'observation.state':target_statistics['observation.state'],
                         'action':self.manifest['action_statistics']}
        self.states=[];self.actions=[];self.images=[];self.ends=[]
        for episode in self.manifest['episodes']:
            path=ROOT/episode['trajectory'];video=CACHE/episode['source_video']
            if (hashlib.sha256(path.read_bytes()).hexdigest()!=episode['sha256']
                    or hashlib.sha256(video.read_bytes()).hexdigest()!=episode['source_video_sha256']):
                raise ValueError('SOURCE_PRIOR_HASH_MISMATCH')
            frame=pd.read_csv(path)
            if len(frame)!=episode['frames'] or not frame[self.manifest['action_columns']].iloc[-1].isna().all():
                raise ValueError('SOURCE_PRIOR_TERMINAL_MISMATCH')
            states=frame[self.manifest['state_columns']].to_numpy(dtype=np.float32)[:-1]
            actions=frame[self.manifest['action_columns']].to_numpy(dtype=np.float32)[:-1]
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError('NONFINITE_SOURCE_PRIOR')
            with av.open(str(video)) as container:
                images=[image_tensor(f.to_ndarray(format='rgb24'),(256,256,3)) for f in container.decode(video=0)]
            if len(images)!=len(frame): raise ValueError('SOURCE_RGB_POSE_LENGTH_MISMATCH')
            self.states.extend(states);self.actions.extend(actions);self.images.extend(images[:-1])
            self.ends.extend([len(self.states)]*len(states))
        self.states=self.normalize(torch.tensor(np.asarray(self.states)),'observation.state')
        self.actions=self.normalize(torch.tensor(np.asarray(self.actions)),'action')


def optimize(model, loader, steps, path, phase='target'):
    optimizer=torch.optim.AdamW(model.get_optim_params(),lr=model.config.optimizer_lr,
                                weight_decay=model.config.optimizer_weight_decay)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();losses=[]
    iterator=iter(loader)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','loss','l1_loss','kld_loss','grad_norm'])
        writer.writeheader()
        for step in range(1,steps+1):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader);batch=next(iterator)
            batch={k:v.cuda() for k,v in batch.items()}
            model.train();optimizer.zero_grad(set_to_none=True)
            loss,metrics=model(batch)
            if not torch.isfinite(loss): raise RuntimeError('NONFINITE_LOSS')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            value=float(loss.detach());losses.append(value)
            writer.writerow({'step':step,'loss':value,**metrics,'grad_norm':float(norm)})
            if step==1 or step%50==0:
                stream.flush();print(json.dumps({'phase':phase,'step':step,'loss':value,'elapsed_s':time.perf_counter()-start}),flush=True)
    torch.cuda.synchronize()
    return losses,time.perf_counter()-start,batch


def source_pretrain(model, target_data, args):
    """Transfer learned features, not the source-normalized output head.

    Target z commands have near-zero variance. Reusing a source-normalized action
    head would change its physical meaning. Restore the exact original target head
    and target RNG before the identical target objective/optimizer/sampling begins.
    """
    data=SourceMotionDataset(args.source_prior,target_data.statistics,model.config.chunk_size)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    head={k:v.detach().clone() for k,v in model.model.action_head.state_dict().items()}
    def feature_hash():
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith('model.action_head.'): continue
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=feature_hash()
    backbone_only=getattr(args,'source_transfer','features')=='backbone'
    # Backbone-only: measured full-transfer runs left the ACT transformer input-invariant
    # on target frames; keep only source-trained ResNet weights, restore all else exactly.
    kept={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('model.backbone.')}
    loader=torch.utils.data.DataLoader(data,batch_size=args.batch_size,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(args.seed+1))
    loss,elapsed,_=optimize(model,loader,args.source_steps,args.output/'source_loss.csv','source_prior')
    after=feature_hash()
    if before==after: raise RuntimeError('SOURCE_FEATURES_DID_NOT_CHANGE')
    if backbone_only:
        missing,unexpected=model.load_state_dict(kept,strict=False)
        if unexpected or any(not k.startswith('model.backbone.') for k in missing):
            raise RuntimeError('BACKBONE_ONLY_RESTORE_MISMATCH')
        current=model.state_dict()
        if any(not torch.equal(current[k],v) for k,v in kept.items()): raise RuntimeError('BACKBONE_ONLY_RESTORE_MISMATCH')
        after=feature_hash()
        if before==after: raise RuntimeError('SOURCE_BACKBONE_DID_NOT_CHANGE')
    model.model.action_head.load_state_dict(head,strict=True)
    for k,v in model.model.action_head.state_dict().items():
        if not torch.equal(v,head[k]): raise RuntimeError('TARGET_HEAD_RESET_MISMATCH')
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    if not torch.equal(torch.get_rng_state(),cpu_rng) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng,strict=True)):
        raise RuntimeError('TARGET_RNG_RESTORE_MISMATCH')
    report={'manifest':args.source_prior.relative_to(ROOT).as_posix(),
        'manifest_sha256':hashlib.sha256(args.source_prior.read_bytes()).hexdigest(),
        'mechanism':('source pose-motion ACT pretraining; transfer ResNet backbone only; all other weights restored to exact target initialization'
                     if backbone_only else 'source pose-motion ACT feature pretraining; reset source action head to exact original target head'),
        'transfer_scope':'backbone' if backbone_only else 'features',
        'source_episodes':[e['episode'] for e in data.manifest['episodes']],
        'source_transitions':len(data),'steps':args.source_steps,'batch_size':args.batch_size,
        'elapsed_s':elapsed,'first_loss':loss[0],'last_loss':loss[-1],
        'source_action_statistics':data.statistics['action'],'state_normalization':'unchanged target training statistics',
        'image':'actual FIRe wrist RGB256x256 -> bilinear320x240 -> CHW/ImageNet normalization; no crop',
        'features_before_sha256':before,'features_after_sha256':after,
        'target_action_head_restored_exactly':True,'target_rng_restored':True,
        'target_optimizer':'fresh AdamW; source optimizer discarded',
        'source_commands_used':False,'source_physical_unit_calibration':False}
    write_json(args.output/'source_pretraining.json',report)
    return report


class ACTPolicy:
    """Observation {state: (6,), front: uint8 HWC} -> physical (configured horizon,6) deltas."""
    def __init__(self, checkpoint):
        self.model=LeRobotACT.from_pretrained(checkpoint,local_files_only=True,strict=True)
        self.device=next(self.model.parameters()).device
        self.metadata=json.loads((Path(checkpoint)/'normalization.json').read_text())
        self.stats=self.metadata['statistics']

    def reset(self):
        self.model.reset()

    @torch.inference_mode()
    def predict(self, observation):
        state=np.asarray(observation['state'],dtype=np.float32)
        if state.shape!=(6,) or not np.isfinite(state).all():
            raise ValueError('INVALID_STATE')
        s=self.stats['observation.state']; a=self.stats['action']
        state=(torch.from_numpy(state)-torch.tensor(s['mean']))/torch.tensor(s['std']).clamp_min(1e-6)
        batch={'observation.state':state[None].to(self.device),
               VIDEO:image_tensor(observation['front'])[None].to(self.device)}
        normalized=self.model.predict_action_chunk(batch)[0].cpu()
        chunk=(normalized*torch.tensor(a['std']).clamp_min(1e-6)+torch.tensor(a['mean'])).numpy()
        if chunk.shape!=(self.model.config.chunk_size,6) or not np.isfinite(chunk).all():
            raise ValueError('INVALID_ACTION_CHUNK')
        return chunk


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA_REQUIRED')
    args.output.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'checkpoints'/args.output.name
    if checkpoint.exists():
        raise FileExistsError('CHECKPOINT_ALREADY_EXISTS')
    pins=json.loads((ROOT/'references/source_manifest.json').read_text())
    installed=json.loads(importlib.metadata.distribution('lerobot').read_text('direct_url.json'))
    if installed['vcs_info']['commit_id']!=pins['local']['lerobot']['commit']:
        raise ValueError('LEROBOT_PIN_MISMATCH')
    evidence=validate(args.dataset)
    recorded=json.loads(args.dataset_report.read_text())
    if evidence['dataset_sha256']!=recorded['dataset_sha256'] or recorded['official_loader']!='PASS':
        raise ValueError('DATASET_EVIDENCE_MISMATCH')
    # Seed list must already be versioned; learned evaluation cannot define it later.
    frozen=subprocess.check_output(['git','show','HEAD:configs/seeds.json'],cwd=ROOT)
    if json.loads(frozen)!=json.loads((ROOT/'configs/seeds.json').read_text()):
        raise ValueError('EVALUATION_SEEDS_NOT_COMMITTED')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    config=ACTConfig(device='cuda',chunk_size=args.chunk_size,n_action_steps=1,
        input_features={VIDEO:PolicyFeature(FeatureType.VISUAL,(3,240,320)),
                        'observation.state':PolicyFeature(FeatureType.STATE,(6,))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(6,))},
        pretrained_backbone_weights=None if args.backbone_weights=='none' else 'ResNet18_Weights.IMAGENET1K_V1',optimizer_lr=args.lr,optimizer_lr_backbone=args.backbone_lr)
    model=LeRobotACT(config).cuda()
    data=CanonicalDataset(args.dataset,config.chunk_size)
    selected=list(range(len(evidence['train_seeds']))) if args.episodes is None else args.episodes
    if not selected or len(set(selected))!=len(selected) or any(i<0 or i>=len(evidence['train_seeds']) for i in selected):
        raise ValueError('INVALID_TRAINING_EPISODES')
    lengths=[e['length'] for e in jsonl(args.dataset/'meta/episodes.jsonl')]
    starts=np.r_[0,np.cumsum(lengths)]
    indices=[j for i in selected for j in range(int(starts[i]),int(starts[i+1]))]
    training_data=torch.utils.data.Subset(data,indices)
    sampler=None
    sampling={'mode':args.sampling}
    if args.sampling!='uniform':
        action_stats=data.statistics['action']
        physical=data.actions[indices]*torch.tensor(action_stats['std']).clamp_min(1e-6)+torch.tensor(action_stats['mean'])
        critical=torch.linalg.vector_norm(physical[:,3:],dim=1)>.003
        count=int(critical.sum())
        if not 0<count<len(indices): raise ValueError('BOTH_ORIENTATION_SAMPLING_GROUPS_REQUIRED')
        if args.sampling=='orientation-balanced':
            weights=torch.where(critical,.5/count,.5/(len(indices)-count)).double()
        else:
            lateral=(torch.linalg.vector_norm(physical[:,:2],dim=1)>.0003)&~critical
            groups=[critical,lateral,~(critical|lateral)]
            names=['rotation','lateral_alignment','settled']
            if args.sampling=='precision-balanced':
                task=json.loads((args.dataset/'meta/provenance.json').read_text())['generator']['config']
                bound=task['translation_limit_m']
                fine=lateral&(physical[:,:2].abs().amax(dim=1)<bound-1e-8)
                groups=[critical,lateral&~fine,fine,~(critical|lateral)]
                names=['rotation','saturated_lateral','unsaturated_lateral','settled']
                sampling['unsaturated_action_upper_bound_m']=bound-1e-8
            counts=[int(mask.sum()) for mask in groups]
            if min(counts)<1: raise ValueError('ALL_PHASE_SAMPLING_GROUPS_REQUIRED')
            weights=torch.zeros(len(indices),dtype=torch.double)
            for mask,n in zip(groups,counts): weights[mask]=1/(len(groups)*n)
            sampling.update({'phase_order':names,
                'phase_counts':counts,'phase_probability_mass':[float(weights[m].sum()) for m in groups],
                'lateral_action_threshold_m':.0003})
        sampler=torch.utils.data.WeightedRandomSampler(weights,len(indices),replacement=True,
                    generator=torch.Generator().manual_seed(args.seed))
        sampling.update({'rotation_threshold_rad':.003,'critical_frames':count,'total_frames':len(indices),
            'critical_probability_mass':float(weights[critical].sum()/weights.sum()),
            'replacement':True,'sampler_seed':args.seed,'target_data_unchanged':True})
    loader=torch.utils.data.DataLoader(training_data,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,num_workers=0,
                                       generator=torch.Generator().manual_seed(args.seed))
    tracked=next(p for p in model.parameters() if p.requires_grad)
    before=tracked.detach().clone()
    context={'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_lock_sha256':hashlib.sha256((ROOT/'scripts/local/requirements.lock.txt').read_bytes()).hexdigest(),
        'gr00t_commit':pins['upstream']['commit'],'lerobot_commit':pins['local']['lerobot']['commit'],
        'dataset_id':args.dataset.name,'dataset_sha256':evidence['dataset_sha256'],
        'dataset_validation_report':args.dataset_report.resolve().relative_to(ROOT).as_posix(),
        'train_seeds':[evidence['train_seeds'][i] for i in selected],
        'training_episode_indices':selected,'training_frames':len(indices),'sampling':sampling,
        'normalization_source':'full canonical training dataset; unchanged for subset sanity',
        'held_out_seeds':evidence['held_out_seeds'],
        'training_seed':args.seed,'steps':args.steps,'batch_size':args.batch_size,'initialization':'from scratch' if args.backbone_weights=='none' else 'ImageNet ResNet18; fresh ACT transformer/action head',
        'backbone_initialization':args.backbone_weights,
        'device':torch.cuda.get_device_name(0),'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'packages':{k:importlib.metadata.version(k) for k in ('torch','torchvision','lerobot','av','numpy')},
        'checkpoint':checkpoint.relative_to(ROOT).as_posix(),'resume':False,
        'track':'source-informed' if args.source_prior else 'target-only',
        'source_prior':None if not args.source_prior else {'manifest':args.source_prior.relative_to(ROOT).as_posix(),'steps':args.source_steps,'target_steps_separate':args.steps,
            'transfer_scope':getattr(args,'source_transfer','features')}}
    write_json(args.output/'run.json',context)
    config.save_pretrained(args.output)
    source_report=source_pretrain(model,data,args) if args.source_prior else None
    losses,elapsed,batch=optimize(model,loader,args.steps,args.output/'loss.csv')
    changed=float((tracked.detach()-before).abs().max())
    if changed<=0: raise RuntimeError('WEIGHTS_DID_NOT_CHANGE')
    model.save_pretrained(checkpoint)
    write_json(checkpoint/'normalization.json',{'statistics':data.statistics,'image_size_wh':list(IMAGE_SIZE),
        'image_mean':[.485,.456,.406],'image_std':[.229,.224,.225],'dataset_sha256':evidence['dataset_sha256']})
    probe={k:v[:1] for k,v in batch.items() if k not in ('action','action_is_pad')}
    expected=model.predict_action_chunk(probe).cpu()
    reloaded=LeRobotACT.from_pretrained(checkpoint,local_files_only=True,strict=True)
    actual=reloaded.predict_action_chunk(probe).cpu()
    if not torch.allclose(expected,actual,atol=1e-6,rtol=0) or not torch.isfinite(actual).all():
        raise RuntimeError('CHECKPOINT_RELOAD_MISMATCH')
    summary={**context,'source_pretraining':source_report,'status':'PASS','elapsed_s':elapsed,'first_loss':losses[0],
        'last_loss':losses[-1],'first_50_mean_loss':float(np.mean(losses[:50])),
        'last_50_mean_loss':float(np.mean(losses[-50:])),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'tracked_parameter_max_change':changed,
        'checkpoint_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir() if p.is_file()},
        'reload':'PASS','reload_chunk_shape':list(actual.shape),'evaluation':'NOT_RUN'}
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/ur10e_10')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-report',type=Path,default=ROOT/'artifacts/datasets/validation.json')
    parser.add_argument('--chunk-size',type=int,default=16)
    parser.add_argument('--sampling',choices=['uniform','orientation-balanced','phase-balanced','precision-balanced'],default='uniform')
    parser.add_argument('--lr',type=float,default=1e-4,help='Main ACT optimizer rate; historical default preserved.')
    parser.add_argument('--backbone-lr',type=float,default=1e-4,help='Explicit controlled backbone optimizer rate; historical default preserved.')
    parser.add_argument('--backbone-weights',choices=['none','imagenet'],default='none')
    parser.add_argument('--episodes',type=int,nargs='+',help='Explicit canonical episode subset for overfit sanity; no data copy.')
    parser.add_argument('--source-prior',type=Path,help='Pinned FIRe motion-prior manifest for source-informed feature initialization.')
    parser.add_argument('--source-steps',type=int,default=1000)
    parser.add_argument('--source-transfer',choices=['features','backbone'],default='features',
                        help='Weights kept after source pretraining; features is the historical recipe.')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20260929)
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.chunk_size)<1 or min(args.backbone_lr,args.lr)<=0: parser.error('steps, batch-size and chunk-size must be positive')
    if args.source_prior:
        args.source_prior=args.source_prior.resolve()
        if args.source_steps<1: parser.error('source-steps must be positive')
    train(args)
