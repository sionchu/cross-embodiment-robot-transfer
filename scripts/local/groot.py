"""Real N1.7 policy path; contract validation does not pretend to run a model."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from dataset import ROOT, TASK, write_json


def load_contract(upstream):
    pins=json.loads((ROOT/'references/source_manifest.json').read_text())
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()!=pins['upstream']['commit']:
        raise ValueError('UPSTREAM_PIN_MISMATCH')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=upstream,text=True).strip():
        raise ValueError('UPSTREAM_DIRTY')
    sys.path.insert(0,str(upstream.resolve()))
    spec=importlib.util.spec_from_file_location('target_groot_config',ROOT/'configs/groot_ur10e.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.MODALITIES,pins


def observation_to_groot(observation,modalities):
    from gr00t.data.utils import parse_observation_gr00t
    state=np.asarray(observation['state'],dtype=np.float32)
    rgb=np.asarray(observation['front'])
    if state.shape!=(6,) or not np.isfinite(state).all(): raise ValueError('INVALID_STATE')
    if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8: raise ValueError('INVALID_RGB')
    # Official helper adds batch dimensions; retain one time step for image/state.
    return parse_observation_gr00t({'video.front':rgb[None],'state.eef_relative_pose':state[None],
        'annotation.human.task_description':observation.get('task',TASK)},modalities)


def decode_action(action):
    # Official Gr00tPolicy.get_action has already decoded physical units.
    chunk=np.asarray(action['eef_delta'])
    if chunk.shape!=(1,16,6) or not np.isfinite(chunk).all(): raise ValueError('INVALID_GR00T_ACTION')
    return chunk[0].astype(np.float32)


def check_checkpoint(checkpoint,modalities):
    from gr00t.data.utils import to_json_serializable
    config=json.loads((checkpoint/'config.json').read_text())
    if config.get('model_type')!='Gr00tN1d7' or config.get('model_name')!='nvidia/Cosmos-Reason2-2B':
        raise ValueError('NOT_N17_CHECKPOINT')
    processor=checkpoint if (checkpoint/'processor_config.json').exists() else checkpoint/'processor'
    saved=json.loads((processor/'processor_config.json').read_text())['processor_kwargs']
    target=saved['modality_configs'].get('new_embodiment')
    if target!=to_json_serializable(modalities): raise ValueError('TARGET_MODALITY_MISMATCH')
    statistics=json.loads((processor/'statistics.json').read_text())['new_embodiment']
    for modality,key in [('state','eef_relative_pose'),('action','eef_delta')]:
        for stat in ('min','max','mean','std'):
            value=np.asarray(statistics[modality][key][stat])
            if value.shape!=(6,) or not np.isfinite(value).all(): raise ValueError('CHECKPOINT_STATS_MISMATCH')
    ids=json.loads((processor/'embodiment_id.json').read_text())
    if 'new_embodiment' not in ids: raise ValueError('TARGET_EMBODIMENT_ID_MISSING')
    if config.get('action_horizon',0)<16 or saved.get('max_action_horizon',0)<16:
        raise ValueError('CHECKPOINT_HORIZON_TOO_SHORT')
    weights=list(checkpoint.glob('*.safetensors'))
    if not weights or any(p.stat().st_size==0 for p in weights): raise ValueError('CHECKPOINT_WEIGHTS_MISSING')
    index=checkpoint/'model.safetensors.index.json'
    if index.exists():
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            if not (checkpoint/name).is_file(): raise ValueError('CHECKPOINT_SHARD_MISSING')
    return {'status':'PASS','model_type':config['model_type'],'weight_files':len(weights),
            'config_sha256':hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()}


class GR00TPolicy:
    """Same observation -> physical action_chunk interface as local ACTPolicy."""
    def __init__(self,checkpoint,upstream,e0_evidence):
        self.modalities,pins=load_contract(upstream)
        # A base/DROID checkpoint cannot represent a trained NEW_EMBODIMENT policy.
        check_checkpoint(Path(checkpoint),self.modalities)
        e0=json.loads(Path(e0_evidence).read_text())
        if e0.get('status')!='PASS' or e0.get('upstream_commit')!=pins['upstream']['commit']:
            raise RuntimeError('E0_NOT_PASSED_FOR_PINNED_UPSTREAM')
        if sys.platform!='linux' or not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<pins['environment']['inference_min_vram_bytes']:
            raise RuntimeError('MODEL_COMPUTE_NOT_AVAILABLE')
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag
        self.model=Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT,str(checkpoint),device='cuda:0',strict=True)

    def reset(self):
        self.model.reset()

    def predict(self,observation):
        action,_=self.model.get_action(observation_to_groot(observation,self.modalities))
        return decode_action(action)


def preflight(dataset,upstream,checkpoint=None):
    modalities,pins=load_contract(upstream)
    evidence=json.loads((ROOT/'artifacts/datasets/validation.json').read_text())
    for relative,digest in evidence['files_sha256'].items():
        if hashlib.sha256((dataset/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('DATASET_HASH_MISMATCH')
    handle=None
    if sys.platform=='win32':
        handle=os.add_dll_directory(str(ROOT/'.external/ffmpeg-7.1.1/ffmpeg-7.1.1-full_build-shared/bin'))
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    loader=LeRobotEpisodeLoader(dataset,modalities); episode=loader[0]
    observation={'state':episode['state.eef_relative_pose'].iloc[0],'front':episode['video.front'].iloc[0]}
    batched=observation_to_groot(observation,modalities)
    assert batched['video']['front'].shape==(1,1,480,640,3)
    assert batched['state']['eef_relative_pose'].shape==(1,1,6)
    assert batched['language']['annotation.human.task_description']==[[TASK]]
    values=np.stack(episode['action.eef_delta'].iloc[:16]).astype(np.float32)
    processor=StateActionProcessor({'new_embodiment':modalities},{'new_embodiment':loader.get_dataset_statistics()},
                                   use_relative_action=True,clip_outliers=False)
    restored=processor.unapply_action(processor.apply_action({'eef_delta':values},'new_embodiment'),'new_embodiment')
    np.testing.assert_allclose(decode_action({'eef_delta':restored['eef_delta'][None]}),values,atol=1e-7,rtol=0)
    if handle: handle.close()
    memory=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    available=sys.platform=='linux' and memory>=pins['environment']['inference_min_vram_bytes']
    return {'integration_status':'PASS','model_execution':'NOT_RUN',
        'runtime_status':'AVAILABLE_UNTESTED' if available else 'MODEL_COMPUTE_NOT_AVAILABLE',
        'checkpoint_validation':check_checkpoint(checkpoint,modalities) if checkpoint else {'status':'NOT_RUN','reason':'TARGET_CHECKPOINT_NOT_AVAILABLE'},
        'embodiment':'NEW_EMBODIMENT','observation_shapes':{'front':[1,1,480,640,3],'state':[1,1,6]},
        'official_observation_batching':'PASS','official_action_roundtrip':'PASS','decoded_chunk_shape':[16,6],
        'gr00t_commit':pins['upstream']['commit'],'base_model_reference':pins['models'][0],
        'dataset_id':dataset.name,'dataset_sha256':json.loads((ROOT/'artifacts/datasets/validation.json').read_text())['dataset_sha256'],
        'platform':sys.platform,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_total_bytes':memory,'minimum_inference_bytes':pins['environment']['inference_min_vram_bytes'],
        'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/ur10e_10')
    parser.add_argument('--upstream',type=Path,default=ROOT/'.external/Isaac-GR00T')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report=preflight(args.dataset,args.upstream,args.checkpoint)
    args.report.parent.mkdir(parents=True,exist_ok=True); write_json(args.report,report)
    print(json.dumps(report,indent=2))
