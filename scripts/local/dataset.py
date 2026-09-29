"""One canonical GR00T LeRobot v2 dataset: export, validate, official-loader check."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
TASK='Insert the fixed square peg into the receptacle.'
VIDEO='observation.images.front'
MODALITY={
    'state':{'eef_relative_pose':{'start':0,'end':6}},
    'action':{'eef_delta':{'start':0,'end':6}},
    'video':{'front':{'original_key':VIDEO}},
    'annotation':{'human.task_description':{}},
}


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    values=np.asarray(values,dtype=np.float32)
    return {key:value.tolist() for key,value in {
        'mean':values.mean(0),'std':values.std(0),'min':values.min(0),'max':values.max(0),
        'q01':np.quantile(values,.01,axis=0),'q99':np.quantile(values,.99,axis=0)}.items()}


def video_metadata(path):
    command=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_streams',
             '-show_entries','frame=best_effort_timestamp_time:stream=width,height,avg_frame_rate,codec_name',
             '-of','json',str(path)]
    return json.loads(subprocess.check_output(command,text=True))


def check_rows(df, episode, offset, bounds):
    n=episode['length']
    required={'observation.state','action','timestamp','frame_index','episode_index','index',
              'task_index','annotation.human.task_description','next.done','next.reward'}
    if not required <= set(df.columns):
        raise ValueError('MISSING_COLUMNS')
    if len(df)!=n or n<2:
        raise ValueError('EPISODE_LENGTH_MISMATCH')
    state=np.stack(df['observation.state'])
    action=np.stack(df['action'])
    if state.shape!=(n,6) or action.shape!=(n,6):
        raise ValueError('DIMENSION_MISMATCH')
    if state.dtype!=np.float32 or action.dtype!=np.float32:
        raise ValueError('NUMERIC_DTYPE_MISMATCH')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('NONFINITE_DATA')
    if np.any(np.abs(action)>np.asarray(bounds)+1e-8):
        raise ValueError('ACTION_RANGE_VIOLATION')
    if not np.allclose(df.timestamp,np.arange(n)/15,atol=1e-6,rtol=0):
        raise ValueError('INVALID_TIMESTAMPS')
    for column,expected in [('frame_index',np.arange(n)),('index',np.arange(offset,offset+n)),
                            ('episode_index',np.full(n,episode['episode_index']))]:
        if not np.array_equal(df[column],expected):
            raise ValueError('EPISODE_DISCONTINUITY')
    if not (df['task_index'].eq(0).all() and df['annotation.human.task_description'].eq(0).all()):
        raise ValueError('INVALID_TASK')
    if not np.array_equal(df['next.done'],np.arange(n)==n-1):
        raise ValueError('INVALID_TERMINAL_FLAGS')
    if not np.array_equal(df['next.reward'],(np.arange(n)==n-1).astype(float)):
        raise ValueError('INVALID_REWARD')
    return state,action


def export(raw, destination):
    raw_dirs=[folder.resolve() for folder in ([raw] if isinstance(raw,Path) else raw)]
    runs=[json.loads((folder/'result.json').read_text()) for folder in raw_dirs]
    frozen=json.loads(subprocess.check_output(['git','show','HEAD:configs/seeds.json'],cwd=ROOT))
    if frozen!=json.loads((ROOT/'configs/seeds.json').read_text()):
        raise ValueError('SEEDS_NOT_COMMITTED')
    entries=[(folder,i,e) for folder,run in zip(raw_dirs,runs,strict=True) for i,e in enumerate(run['episodes'])]
    episodes=[e for _,_,e in entries]
    count=len(episodes)
    pool=frozen.get('train_pool',frozen['train'])
    if count not in frozen.get('target_data_sizes',[10]) or [e['seed'] for e in episodes]!=pool[:count]:
        raise ValueError('ORDERED_NESTED_TRAIN_EPISODES_REQUIRED')
    for run in runs:
        if run['status']!='PASS': raise ValueError('UNSUCCESSFUL_GENERATION_RUN')
        for key in ('config','asset','script_sha256'):
            if run[key]!=runs[0][key]: raise ValueError('GENERATOR_PROTOCOL_MISMATCH')
    result=runs[0] if len(runs)==1 else {
        'status':'PASS','config':runs[0]['config'],'asset':runs[0]['asset'],
        'script_sha256':runs[0]['script_sha256'],'episodes':episodes,
        'source_runs':[{'path':folder.relative_to(ROOT).as_posix(),
            'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
            **{key:run[key] for key in ('project_commit','environment','isaac_sim_version','gpu')}}
            for folder,run in zip(raw_dirs,runs,strict=True)]}
    splits={'train':pool[:count],'development':frozen['development'],'held_out':frozen['held_out'],'rule':frozen['rule']}
    destination.mkdir(parents=True,exist_ok=False)
    meta=destination/'meta'; meta.mkdir()
    data=destination/'data/chunk-000'; data.mkdir(parents=True)
    videos=destination/f'videos/chunk-000/{VIDEO}'; videos.mkdir(parents=True)
    episode_meta=[]; episode_stats=[]; all_states=[]; all_actions=[]; offset=0
    for i,(source,source_index,episode) in enumerate(entries):
        trace=json.loads((source/f'{source_index:02d}_trace.json').read_text())
        n=len(trace)
        if episode['outcome']!='SUCCESS' or episode['recorded_frames']!=n or n!=episode['steps']:
            raise ValueError('UNSUCCESSFUL_OR_UNRECORDED_EPISODE')
        if episode['unique_rgb_frames']<2:
            raise ValueError('FROZEN_RGB')
        if not np.allclose([r['simulation_timestamp'] for r in trace],np.arange(n)/15,atol=1e-6,rtol=0):
            raise ValueError('PHYSICS_VIDEO_CADENCE_MISMATCH')
        state=np.asarray([r['state'] for r in trace],dtype=np.float32)
        action=np.asarray([r['action'] for r in trace],dtype=np.float32)
        df=pd.DataFrame({'observation.state':list(state),'action':list(action),
                         'timestamp':np.arange(n,dtype=np.float32)/15,
                         'frame_index':np.arange(n,dtype=np.int64),
                         'episode_index':np.full(n,i,dtype=np.int64),
                         'index':np.arange(offset,offset+n,dtype=np.int64),
                         'task_index':np.zeros(n,dtype=np.int64),
                         'annotation.human.task_description':np.zeros(n,dtype=np.int64),
                         'next.done':np.arange(n)==n-1,
                         'next.reward':(np.arange(n)==n-1).astype(np.float32)})
        df.to_parquet(data/f'episode_{i:06d}.parquet',index=False)
        shutil.copyfile(source/f'{source_index:02d}_front.mp4',videos/f'episode_{i:06d}.mp4')
        episode_meta.append({'episode_index':i,'tasks':[TASK],'length':n,'seed':episode['seed'],
                             'success':True,'expert':'measured_jacobian_scripted',
                             'final_state':episode['final_state'],'max_contact_n':episode['max_contact_n']})
        episode_stats.append({'episode_index':i,'stats':{'observation.state':stats(state),'action':stats(action)}})
        all_states.extend(state); all_actions.extend(action); offset+=n
    scalar=lambda dtype:{'dtype':dtype,'shape':[1],'names':None}
    features={'observation.state':{'dtype':'float32','shape':[6],'names':['x','y','z','rx','ry','rz']},
              'action':{'dtype':'float32','shape':[6],'names':['dx','dy','dz','dRx','dRy','dRz']},
              VIDEO:{'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],
                     'info':{'video.fps':15,'video.codec':'h264','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}}
    for key in ['frame_index','episode_index','index','task_index','annotation.human.task_description']:
        features[key]=scalar('int64')
    features.update({'timestamp':scalar('float32'),'next.reward':scalar('float32'),'next.done':scalar('bool')})
    info={'codebase_version':'v2.1','robot_type':'ur10e_fixed_peg','fps':15,'total_episodes':count,
          'total_frames':offset,'total_tasks':1,'total_videos':count,'total_chunks':1,'chunks_size':1000,
          'splits':{'train':f'0:{count}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
    write_json(meta/'info.json',info); write_json(meta/'modality.json',MODALITY)
    write_json(meta/'stats.json',{'observation.state':stats(all_states),'action':stats(all_actions)})
    for name,rows in [('episodes.jsonl',episode_meta),('episodes_stats.jsonl',episode_stats),('tasks.jsonl',[{'task_index':0,'task':TASK}])]:
        (meta/name).write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    write_json(meta/'provenance.json',{'source':'project-generated Isaac Sim expert; no real robot data',
        'generator':result,'splits':splits,'gr00t_commit':json.loads((ROOT/'references/source_manifest.json').read_text())['upstream']['commit'],
        'state':'hole-frame peg-tip position in metres; Log(R_hole^-1 R_wrist) in radians',
        'action':'hole-frame delta translation in metres and left-composed rotation vector in radians',
        'rotation_composition':'R_next = R_hole Exp(dR) R_hole^-1 R_current',
        'timing':'pre-action observation and RGB; one command per 4 physics steps at 60 Hz; video 15 Hz'})


def validate(root, upstream=None, ffmpeg_bin=None):
    info=json.loads((root/'meta/info.json').read_text())
    modality=json.loads((root/'meta/modality.json').read_text())
    provenance=json.loads((root/'meta/provenance.json').read_text())
    episodes=jsonl(root/'meta/episodes.jsonl')
    if modality!=MODALITY:
        raise ValueError('MODALITY_MISMATCH')
    frozen=json.loads(subprocess.check_output(['git','show','HEAD:configs/seeds.json'],cwd=ROOT))
    count=len(episodes)
    if info['codebase_version']!='v2.1' or info['fps']!=15 or count not in frozen.get('target_data_sizes',[10]) or info['total_episodes']!=count:
        raise ValueError('INVALID_DATASET_METADATA')
    for key in ['observation.state','action']:
        if info['features'][key]['shape']!=[6] or info['features'][key]['dtype']!='float32':
            raise ValueError('FEATURE_DIMENSION_MISMATCH')
    if info['features'][VIDEO]['shape']!=[480,640,3]:
        raise ValueError('VIDEO_SHAPE_MISMATCH')
    if jsonl(root/'meta/tasks.jsonl')!=[{'task_index':0,'task':TASK}]:
        raise ValueError('INVALID_TASK_METADATA')
    splits=provenance['splits']
    if set(splits['train']) & (set(splits['held_out'])|set(splits['development'])):
        raise ValueError('TRAIN_EVAL_OVERLAP')
    if [e['seed'] for e in episodes]!=splits['train'] or len(set(splits['train']))!=count or splits['train']!=frozen.get('train_pool',frozen['train'])[:count]:
        raise ValueError('SEED_MISMATCH')
    if any(splits[k]!=frozen[k] for k in ('development','held_out')):
        raise ValueError('EVAL_PROTOCOL_MISMATCH')
    c=provenance['generator']['config']; bounds=[c['translation_limit_m']]*3+[c['rotation_limit_rad']]*3
    total=0; states=[]; actions=[]; video_frames=[]; writers=[]
    for i,e in enumerate(episodes):
        if e['episode_index']!=i or not e['success'] or e['tasks']!=[TASK]:
            raise ValueError('INVALID_EPISODE_METADATA')
        df=pd.read_parquet(root/info['data_path'].format(episode_chunk=0,episode_index=i))
        state,action=check_rows(df,e,total,bounds)
        metadata=pq.read_metadata(root/info['data_path'].format(episode_chunk=0,episode_index=i))
        writer={'created_by':metadata.created_by,'pandas_version':json.loads(metadata.metadata.get(b'pandas',b'{}')).get('pandas_version')}
        if writer not in writers: writers.append(writer)
        path=root/info['video_path'].format(episode_chunk=0,episode_index=i,video_key=VIDEO)
        video=video_metadata(path); stream=video['streams'][0]
        times=np.array([float(f['best_effort_timestamp_time']) for f in video['frames']])
        if len(times)!=len(df) or (stream['width'],stream['height'])!=(640,480):
            raise ValueError('VIDEO_STATE_LENGTH_MISMATCH')
        if stream['avg_frame_rate']!='15/1' or not np.allclose(times,df.timestamp,atol=1e-6,rtol=0):
            raise ValueError('VIDEO_STATE_TIMESTAMP_MISMATCH')
        states.extend(state); actions.extend(action); total+=len(df); video_frames.append(len(times))
    if total!=info['total_frames']:
        raise ValueError('TOTAL_FRAME_MISMATCH')
    expected_stats={'observation.state':stats(states),'action':stats(actions)}
    saved=json.loads((root/'meta/stats.json').read_text())
    for key in expected_stats:
        for stat,value in expected_stats[key].items():
            if not np.allclose(value,saved[key][stat],atol=1e-8):
                raise ValueError('STATS_MISMATCH')
    report={'status':'PASS','episodes':len(episodes),'frames':total,'video_frames':video_frames,
            'train_seeds':splits['train'],'held_out_seeds':splits['held_out'],'official_loader':'NOT_RUN','parquet_writers':writers,
            'episode_results':episodes,
            'generator_source_sha256':provenance['generator']['script_sha256'],
            'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'modality_config_sha256':hashlib.sha256((ROOT/'configs/groot_ur10e.py').read_bytes()).hexdigest(),
            'project_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    if upstream:
        dll_handle=None
        if sys.platform=='win32':
            directory=ffmpeg_bin or ROOT/'.external/ffmpeg-7.1.1/ffmpeg-7.1.1-full_build-shared/bin'
            dll_handle=os.add_dll_directory(str(directory.resolve()))
        pin=provenance['gr00t_commit']
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()!=pin:
            raise ValueError('UPSTREAM_PIN_MISMATCH')
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=upstream,text=True).strip():
            raise ValueError('UPSTREAM_DIRTY')
        sys.path.insert(0,str(upstream.resolve()))
        spec=importlib.util.spec_from_file_location('ur10e_config',ROOT/'configs/groot_ur10e.py')
        config=importlib.util.module_from_spec(spec); spec.loader.exec_module(config)
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.state_action.state_action_processor import StateActionProcessor
        loader=LeRobotEpisodeLoader(root,config.MODALITIES)
        statistics=loader.get_dataset_statistics()
        processor=StateActionProcessor({'new_embodiment':config.MODALITIES},{'new_embodiment':statistics},
                                        clip_outliers=False,use_relative_action=True)
        for i in range(len(loader)):
            episode=loader[i]
            if len(episode)!=episodes[i]['length'] or episode['video.front'].iloc[0].shape!=(480,640,3):
                raise ValueError('OFFICIAL_LOADER_MISMATCH')
            values=np.stack(episode['action.eef_delta'])
            normalized=processor.apply_action({'eef_delta':values},'new_embodiment')
            restored=processor.unapply_action(normalized,'new_embodiment')['eef_delta']
            if not np.allclose(restored,values,atol=1e-7,rtol=0):
                raise ValueError('OFFICIAL_ACTION_ROUNDTRIP_FAILED')
        report['official_loader']='PASS'
        report['official_action_roundtrip']='PASS'
        report['gr00t_commit']=pin
        report['reader_environment']={key:importlib.metadata.version(key) for key in ('torch','torchcodec','numpy','pandas','pyarrow')}
        report['ffmpeg_reader']='7.1.1 shared' if sys.platform=='win32' else 'system'
        if dll_handle:
            dll_handle.close()
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
    report['files_sha256']=files
    report['dataset_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,nargs='+',help='Export this recorded run first; destination must be new.')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--ffmpeg-bin',type=Path,help='Windows shared FFmpeg DLL directory; defaults to the pinned local setup.')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    report={'status':'FAILED'}
    try:
        if args.raw:
            export(args.raw,args.dataset)
        report=validate(args.dataset,args.upstream,args.ffmpeg_bin)
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.report,report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('files_sha256','episode_results')},indent=2))
