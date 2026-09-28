"""Audit real ACT tensors, action distributions and recorded closed-loop failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIMS=['dx','dy','dz','dRx','dRy','dRz']

def distribution(x):
    return {k:v.tolist() for k,v in dict(min=x.min(0),max=x.max(0),mean=x.mean(0),
        std=x.std(0),p01=np.quantile(x,.01,axis=0),p50=np.quantile(x,.5,axis=0),
        p99=np.quantile(x,.99,axis=0)).items()}

def audit(args):
    import av
    import pandas as pd
    import torch
    from act import ACTPolicy, CanonicalDataset, image_tensor
    from insertion import expert,CONFIG
    from dataset import VIDEO
    torch.set_num_threads(4)
    if (args.output/'audit.json').exists(): raise FileExistsError('AUDIT_ALREADY_EXISTS')
    args.output.mkdir(parents=True,exist_ok=True)
    data=CanonicalDataset(ROOT/'datasets/ur10e_10',16)
    policy=ACTPolicy(args.checkpoint)
    predictions=[]; truth=[]; states=[]; expert_labels=[]; tensor_evidence=None
    rows=[]; sample_ids=[]; offset=0
    for ep in (range(10) if args.episodes is None else args.episodes):
        folder=ROOT/'datasets/ur10e_10'
        lengths=[json.loads(line)['length'] for line in (folder/'meta/episodes.jsonl').read_text().splitlines()]
        offset=sum(lengths[:ep])
        df=pd.read_parquet(folder/f'data/chunk-000/episode_{ep:06d}.parquet')
        with av.open(str(folder/f'videos/chunk-000/{VIDEO}/episode_{ep:06d}.mp4')) as c:
            images=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
        for i,(s,a,rgb) in enumerate(zip(df['observation.state'],df.action,images,strict=True)):
            if tensor_evidence is None:
                captured={}
                def hook(module,inputs):
                    captured.update({k:v.detach().cpu().clone() for k,v in inputs[0].items() if isinstance(v,torch.Tensor)})
                handle=policy.model.model.register_forward_pre_hook(hook)
            predicted=policy.predict({'state':s,'front':rgb})[0]
            if tensor_evidence is None:
                handle.remove()
                np.testing.assert_array_equal(captured['observation.state'][0],data[offset+i]['observation.state'])
                np.testing.assert_array_equal(captured[VIDEO][0],data[offset+i][VIDEO])
                tensor_evidence={k:{'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),
                    'max':float(v.max()),'mean':float(v.mean())} for k,v in captured.items()}
                tensor_evidence['train_vs_adapter_input']='BIT_EXACT_ON_DECODED_DEMO'
                tensor_evidence['state_values']=captured['observation.state'][0].tolist()
            sample_ids.append(offset+i);predictions.append(predicted);truth.append(a);states.append(s);expert_labels.append(expert(s)[0])
            rows.append({'episode':ep,'frame':i,'time_s':i/15,
                **{f'gt_{k}':float(v) for k,v in zip(DIMS,a)},
                **{f'pred_{k}':float(v) for k,v in zip(DIMS,predicted)}})
        offset+=len(df)
    pred=np.asarray(predictions);gt=np.asarray(truth);state=np.asarray(states)
    bounds=np.array([CONFIG['translation_limit_m']]*3+[CONFIG['rotation_limit_rad']]*3)
    stats=policy.stats['action'];denom=np.maximum(stats['std'],1e-6)
    roundtrip=(data.actions[sample_ids].numpy()*denom+stats['mean'])
    np.testing.assert_allclose(roundtrip,gt,atol=1e-8,rtol=0)
    np.testing.assert_allclose(expert_labels,gt,atol=1e-8,rtol=0)
    with (args.output/'open_loop.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    evidence={'checkpoint':args.checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256':hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dimensions':DIMS,'units':['m']*3+['rad']*3,'tensor_audit':tensor_evidence,
        'normalization_roundtrip':'PASS','dataset_vs_recomputed_expert':'PASS',
        'temporal':{'observation_history':1,'chunk_length':policy.model.config.chunk_size,'executed_per_query':1,'control_hz':15,'interpolation':False},
        'open_loop_mae':np.abs(pred-gt).mean(0).tolist(),
        'open_loop_rmse':np.sqrt(((pred-gt)**2).mean(0)).tolist(),
        'dataset_actions':distribution(gt),'expert_at_dataset_states':distribution(np.asarray(expert_labels)),
        'predicted_at_dataset_states':distribution(pred),
        'sign_agreement_nontrivial':[float(np.mean(np.sign(pred[:,i][abs(gt[:,i])>1e-4])==np.sign(gt[:,i][abs(gt[:,i])>1e-4]))) for i in range(6)],
        'action_std_effective':denom.tolist(),'action_clipping_fraction_above_1e8':np.mean(abs(pred)>bounds+1e-8,axis=0).tolist()}
    if args.baseline:
        failures=[];raw_all=[];applied_all=[];ref_all=[];time_rows=[]
        for line in (ROOT/'artifacts/eval/act-10-002/episodes.jsonl').read_text().splitlines():
            row=json.loads(line);trace=json.loads((ROOT/row['raw_artifact_dir']/f"{row['seed']}_trace.json").read_text())
            ss=np.asarray([t['state'] for t in trace]+[row['final_state']])
            aa=np.asarray([t['raw_chunk'][0] for t in trace]);ee=np.asarray([t['action'] for t in trace]);rr=np.asarray([expert(s)[0] for s in ss[:-1]])
            raw_all.extend(aa);applied_all.extend(ee);ref_all.extend(rr)
            lateral=np.linalg.norm(ss[:,:2],axis=1);angle=np.linalg.norm(ss[:,3:],axis=1)
            track=(ss[1:,:3]-ss[:-1,:3])-ee[:,:3]
            for i,t in enumerate(trace):
                time_rows.append({'seed':row['seed'],'step':i,'time_s':i/15,'lateral_m':float(lateral[i]),
                    'orientation_rad':float(angle[i]),'distance_to_goal_m':float(np.linalg.norm(ss[i,:3]-[0,0,.03])),
                    **{f'state_{k}':float(v) for k,v in zip(DIMS,ss[i])},
                    **{f'raw_{k}':float(v) for k,v in zip(DIMS,aa[i])},
                    **{f'executed_{k}':float(v) for k,v in zip(DIMS,ee[i])}})
            failures.append({'seed':row['seed'],'outcome':row['outcome'],'steps':row['steps'],
                'min_lateral_m':float(lateral.min()),'final_lateral_m':float(lateral[-1]),
                'min_orientation_rad':float(angle.min()),'final_orientation_rad':float(angle[-1]),
                'max_depth_m':float(ss[:,2].max()),'max_contact_n':row['max_contact_n'],
                'ever_geometrically_aligned':bool(np.any((lateral<CONFIG['lateral_tolerance_m'])&(angle<CONFIG['angular_tolerance_rad']))),
                'translation_tracking_mae_m':np.abs(track).mean(0).tolist(),
                'raw_action_std':aa.std(0).tolist(),'expert_disagreement_mae':np.abs(ee-rr).mean(0).tolist(),
                'finite_actions':bool(np.isfinite(aa).all()),
                'state_outside_train_range_fraction':np.mean((ss<state.min(0)-1e-7)|(ss>state.max(0)+1e-7),axis=0).tolist(),
                'trace_sha256':row['trace_sha256']})
        evidence['baseline_failures']=failures
        evidence['closed_loop_predicted']=distribution(np.asarray(raw_all))
        evidence['closed_loop_executed']=distribution(np.asarray(applied_all))
        evidence['expert_at_policy_states']=distribution(np.asarray(ref_all))
        with (args.output/'closed_loop.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(time_rows[0]));w.writeheader();w.writerows(time_rows)
    (args.output/'audit.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:evidence[k] for k in ['checkpoint','tensor_audit','open_loop_mae','sign_agreement_nontrivial']},indent=2))

def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (args.output/'open_loop.csv').open() as f: rows=[r for r in csv.DictReader(f) if r['episode']=='0']
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    for k,ax in zip(DIMS,axes.flat):
        ax.plot([float(r['time_s']) for r in rows],[float(r['gt_'+k]) for r in rows],label='Dataset expert')
        ax.plot([float(r['time_s']) for r in rows],[float(r['pred_'+k]) for r in rows],label='ACT first action')
        limit=max(.003 if k in DIMS[:3] else .025, max(abs(float(r[key])) for r in rows for key in ['gt_'+k,'pred_'+k]))
        ax.set(title=k+' ('+('m' if k in DIMS[:3] else 'rad')+')',xlabel='s',ylim=(-1.15*limit,1.15*limit))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.legend(fontsize=8)
    fig.savefig(args.output/'open_loop.png',dpi=140);plt.close(fig)
    p=args.output/'closed_loop.csv'
    if p.exists():
        with p.open() as f: rows=list(csv.DictReader(f))
        seeds=sorted({int(r['seed']) for r in rows})
        seeds=([1000,1001,1019] if {1000,1001,1019}<=set(seeds) else seeds[:3])
        fig,axes=plt.subplots(3,len(seeds),figsize=(4*len(seeds),8),layout='constrained',squeeze=False)
        for col,seed in enumerate(seeds):
            rr=[r for r in rows if int(r['seed'])==seed];x=[float(r['time_s']) for r in rr]
            for ax,key,label in zip(axes[:,col],['lateral_m','orientation_rad','state_dz'],['Lateral (m)','Angle (rad)','Depth (m)']):
                ax.plot(x,[float(r[key]) for r in rr]);ax.set(title=f'Seed {seed}: {label}',xlabel='s')
        fig.savefig(args.output/'closed_loop.png',dpi=140);plt.close(fig)

def features(args):
    import torch
    import av
    import pandas as pd
    from act import LeRobotACT, ACTConfig, image_tensor
    from dataset import VIDEO
    torch.set_num_threads(2)
    cfg=ACTConfig.from_pretrained(args.checkpoint);cfg.device='cpu'
    model=LeRobotACT.from_pretrained(args.checkpoint,config=cfg,strict=True,local_files_only=True)
    stats=json.loads((args.checkpoint/'normalization.json').read_text())['statistics']
    df=pd.read_parquet(ROOT/'datasets/ur10e_10/data/chunk-000/episode_000000.parquet')
    with av.open(str(ROOT/f'datasets/ur10e_10/videos/chunk-000/{VIDEO}/episode_000000.mp4')) as c:
        images=[f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    observed={}
    for name in ['encoder_robot_state_input_proj','encoder_img_feat_input_proj']:
        def hook(module,inputs,out,key=name):
            observed[key]={'rms':float(out.square().mean().sqrt()),'max_abs':float(out.abs().max())}
        getattr(model.model,name).register_forward_hook(hook)
    attention=model.model.encoder.layers[0].self_attn
    def full_weights(module,inputs,kwargs):
        return inputs,{**kwargs,'average_attn_weights':False}
    attention.register_forward_pre_hook(full_weights,with_kwargs=True)
    def attention_probe(module,inputs,out):
        weights=out[1]
        observed['first_attention']={'mean_entropy':float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()),
            'fraction_queries_max_probability_above_099':float((weights.max(-1).values>.99).float().mean())}
    attention.register_forward_hook(attention_probe)
    records=[]
    with torch.inference_mode():
        for image_index,state_index in [(0,0),(1,1),(10,10),(30,30),(1,0),(1,30)]:
            q=stats['observation.state'];state=torch.tensor(np.array(df['observation.state'].iloc[state_index]))
            state=(state-torch.tensor(q['mean']))/torch.tensor(q['std']).clamp_min(1e-6)
            chunk=model.predict_action_chunk({'observation.state':state[None],VIDEO:image_tensor(images[image_index])[None]})
            records.append({'image_frame':image_index,'state_frame':state_index,'features':dict(observed),
                'first_action_normalized':chunk[0,0].tolist()})
    report={'checkpoint':args.checkpoint.relative_to(ROOT).as_posix(),'device':'CPU diagnostic only',
        'checkpoint_sha256':hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
    (args.output/'feature_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

def sensitivity(args):
    import torch
    from act import LeRobotACT, ACTConfig, CanonicalDataset
    from insertion import expert
    cfg=ACTConfig.from_pretrained(args.checkpoint);cfg.device='cpu'
    torch.set_num_threads(4)
    model=LeRobotACT.from_pretrained(args.checkpoint,config=cfg,strict=True,local_files_only=True).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    data=CanonicalDataset(ROOT/'datasets/ur10e_10',1)
    ss=np.maximum(data.statistics['observation.state']['std'],1e-6)
    aa=np.maximum(data.statistics['action']['std'],1e-6)
    rows=[]
    for i in [0,1,10,30]:
        state=data.states[i].detach().clone().requires_grad_(True)
        def forward(value):
            return model.model({'observation.state':value[None],'observation.images':[data.images[i][None]]})[0][0,0]
        jac=torch.autograd.functional.jacobian(forward,state).detach().numpy()*aa[:,None]/ss[None,:]
        physical=data.states[i].numpy()*ss+data.statistics['observation.state']['mean']
        numerical=[]
        for k in range(6):
            delta=np.eye(6)[k]*1e-5
            numerical.append((expert(physical+delta)[0]-expert(physical-delta)[0])/2e-5)
        rows.append({'frame':i,'physical_state':physical.tolist(),'physical_action_jacobian':jac.tolist(),
                     'expert_local_jacobian':np.stack(numerical,axis=1).tolist()})
    report={'checkpoint':args.checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256':hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows':'action dimension','columns':'state dimension',
        'method':'autograd in evaluation mode, fixed real training image; physical units; no model update','records':rows}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'state_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rotation_diagonals':[np.diag(r['physical_action_jacobian'])[3:].tolist() for r in rows]}))


def rollouts(args):
    from scipy.spatial.transform import Rotation
    from insertion import CONFIG, expert
    summary=json.loads((args.rollout/'summary.json').read_text())
    if summary['execution_status']!='PASS': raise ValueError('ROLLOUT_EXECUTION_INCOMPLETE')
    rows=[]; episodes=[]
    for line in (args.rollout/'episodes.jsonl').read_text().splitlines():
        episode=json.loads(line); path=ROOT/episode['raw_artifact_dir']/f"{episode['seed']}_trace.json"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=episode['trace_sha256']: raise ValueError('TRACE_HASH_MISMATCH')
        trace=json.loads(path.read_text())
        if any('next_state' not in x for x in trace): raise ValueError('POST_STATE_NOT_RECORDED')
        states=np.asarray([x['state'] for x in trace]+[trace[-1]['next_state']])
        raw=np.asarray([x['raw_chunk'][0] for x in trace]); applied=np.asarray([x['action'] for x in trace])
        expected=np.asarray([expert(x)[0] for x in states[:-1]])
        translation_error=np.diff(states[:,:3],axis=0)-applied[:,:3]
        rotations=Rotation.from_rotvec(states[:,3:])
        actual_delta=rotations[1:]*rotations[:-1].inv()
        rotation_error=(Rotation.from_rotvec(applied[:,3:]).inv()*actual_delta).magnitude()
        lateral=np.linalg.norm(states[:,:2],axis=1); angle=np.linalg.norm(states[:,3:],axis=1)
        aligned=(lateral<=CONFIG['lateral_tolerance_m'])&(angle<=CONFIG['angular_tolerance_rad'])
        disagreement=np.linalg.norm(applied[:,3:]-expected[:,3:],axis=1)
        divergent=np.flatnonzero(disagreement>.003)
        episodes.append({'seed':episode['seed'],'outcome':episode['outcome'],'trace_sha256':episode['trace_sha256'],
            'min_lateral_m':float(lateral.min()),'min_orientation_rad':float(angle.min()),
            'ever_aligned':bool(aligned.any()),'first_rotation_disagreement_above_003_step':int(divergent[0]) if len(divergent) else None,
            'translation_tracking_mae_m':np.abs(translation_error).mean(0).tolist(),
            'rotation_tracking_mean_error_rad':float(rotation_error.mean()),'max_depth_m':float(states[:,2].max()),
            'max_contact_n':episode['max_contact_n'],'final_state':episode['final_state']})
        for i,t in enumerate(trace):
            rows.append({'seed':episode['seed'],'step':i,'time_s':t['timestamp'],
                'lateral_m':float(lateral[i]),'orientation_rad':float(angle[i]),
                'distance_to_goal_m':float(np.linalg.norm(states[i,:3]-[0,0,.03])),
                'rotation_tracking_error_rad':float(rotation_error[i]),'max_contact_n':t['max_contact_n'],
                'terminal':t['terminal'] or '', 'geometrically_aligned':bool(aligned[i]),
                **{f'state_{k}':float(v) for k,v in zip(DIMS,states[i])},
                **{f'raw_{k}':float(v) for k,v in zip(DIMS,raw[i])},
                **{f'executed_{k}':float(v) for k,v in zip(DIMS,applied[i])}})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'closed_loop.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={'evaluation':args.rollout.relative_to(ROOT).as_posix(),'checkpoint_sha256':summary['checkpoint_files_sha256'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes':episodes}
    (args.output/'rollout_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


def vae_probe(args):
    import torch
    from act import ACTConfig,LeRobotACT,CanonicalDataset
    from dataset import VIDEO
    cp=args.checkpoint
    cfg=ACTConfig.from_pretrained(cp);cfg.device='cpu'
    torch.set_num_threads(4);torch.manual_seed(0)
    m=LeRobotACT.from_pretrained(cp,config=cfg,strict=True,local_files_only=True).eval()
    data=CanonicalDataset(ROOT/'datasets/ur10e_10',1)
    ids=[0]+[i for i in range(1,len(data)) if data.ends[i]!=data.ends[i-1]]
    b={k:torch.stack([data[i][k] for i in ids]) for k in data[0]};b['observation.images']=[b[VIDEO]]
    with torch.inference_mode():
     zero=m.model(b)[0]
     # Enable only CVAE branch, keeping all attention/dropout children in eval mode.
     m.model.training=True
     seen={}
     def save_mu(module,inputs,out):seen['mu']=out[:,:cfg.latent_dim]
     h1=m.model.vae_encoder_latent_output_proj.register_forward_hook(save_mu)
     def use_mu(module,inputs):return (seen['mu'],)
     h2=m.model.encoder_latent_input_proj.register_forward_pre_hook(use_mu)
     posterior,(mu,logvar)=m.model(b)
     h2.remove();h1.remove()
     samples=torch.stack([m.model(b)[0] for _ in range(16)])
     kld=(-.5*(1+logvar-mu.square()-logvar.exp())).sum(-1)
     gt=b['action'];scale=torch.tensor(np.maximum(data.statistics['action']['std'],1e-6))
     r={'checkpoint':cp.relative_to(ROOT).as_posix(),'checkpoint_sha256':hashlib.sha256((cp/'model.safetensors').read_bytes()).hexdigest(),
     'diagnostic_only':'training labels used solely to inspect posterior/inference gap; never used in evaluation',
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_frames':ids,
     'zero_latent_mae_physical':((zero-gt).abs()*scale).mean((0,1)).tolist(),
     'posterior_mean_mae_physical':((posterior-gt).abs()*scale).mean((0,1)).tolist(),
     'sampled_posterior_mae_physical':((samples-gt).abs()*scale).mean((0,1,2)).tolist(),
     'posterior_mu_max_abs':float(mu.abs().max()),'mean_kld':float(kld.mean()),
     'zero_vs_posterior_max_action_difference_physical':((zero-posterior).abs()*scale).amax((0,1)).tolist()}
    (args.output/'vae_probe.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2))


def recovery(args):
    paths=['artifacts/training/act-overfit-001','artifacts/training/act-overfit-002']
    training=[json.loads((ROOT/p/'summary.json').read_text()) for p in paths]
    configs=[json.loads((ROOT/p/'config.json').read_text()) for p in paths]
    difference={k:[configs[0][k],configs[1][k]] for k in configs[0] if configs[0][k]!=configs[1][k]}
    if set(difference)!={'pretrained_backbone_weights'}: raise ValueError('NOT_SINGLE_FACTOR')
    for key in ['dataset_sha256','training_episode_indices','train_seeds','steps','batch_size','training_seed','normalization_source']:
        if training[0][key]!=training[1][key]: raise ValueError('TRAINING_PROTOCOL_MISMATCH')
    evaluations=['artifacts/eval/act-overfit-train-002','artifacts/eval/act-overfit-train-003']
    results=[json.loads((ROOT/p/'summary.json').read_text()) for p in evaluations]
    rows=[json.loads((ROOT/p/'episodes.jsonl').read_text()) for p in evaluations]
    for key in ['environment_config_hash','dataset_sha256','seeds','action_execution']:
        if results[0][key]!=results[1][key]: raise ValueError('CONTROL_PROTOCOL_MISMATCH')
    np.testing.assert_array_equal(rows[0]['initial_state'],rows[1]['initial_state'])
    if any(x['execution_status']!='PASS' for x in results): raise ValueError('INCOMPLETE_ROLLOUT')
    features=[json.loads((ROOT/f'artifacts/diagnostics/act-overfit-{i:03d}/feature_probe.json').read_text()) for i in [1,2]]
    report={'experiment':'one-episode controlled ACT sanity','changed_config':difference,
        'same_data_seed_steps_normalization_control_and_detector':True,'training_condition_seed':10,
        'outcomes':[r['outcome'] for r in rows],'held_out_claim':False,
        'image_projection_rms':[f['records'][0]['features']['encoder_img_feat_input_proj']['rms'] for f in features],
        'attention_saturated_query_fraction':[f['records'][0]['features']['first_attention']['fraction_queries_max_probability_above_099'] for f in features],
        'checkpoint_sha256':[t['checkpoint_files_sha256']['model.safetensors'] for t in training],
        'training_evidence':paths,'rollout_evidence':evaluations,
        'interpretation':'Restoring maintained ACT visual initialization resolves failed one-episode learning without new data or task relaxation. Ten-demo held-out recovery remains a separate experiment.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'recovery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/act-10-001')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--episodes',type=int,nargs='+')
    p.add_argument('--plot',action='store_true')
    p.add_argument('--features',action='store_true')
    p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--vae-probe',action='store_true')
    p.add_argument('--rollout',type=Path)
    p.add_argument('--recovery',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve()
    if a.rollout: a.rollout=a.rollout.resolve()
    vae_probe(a) if a.vae_probe else rollouts(a) if a.rollout else sensitivity(a) if a.sensitivity else plot(a) if a.plot else features(a) if a.features else recovery(a) if a.recovery else audit(a)
