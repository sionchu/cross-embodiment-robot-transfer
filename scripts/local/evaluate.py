"""Actual checkpoint/expert closed-loop evaluation in Isaac Sim."""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from insertion import CONFIG, ROOT, InsertionEnv, expert


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PolicyProcess:
    def __init__(self,checkpoint,log):
        environment=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME'):
            environment.pop(key,None)
        environment['PATH']=';'.join(p for p in environment.get('PATH','').split(';') if 'isaacsim' not in p.lower())
        self.process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'scripts/local/policy_worker.py'),
            '--checkpoint',str(checkpoint.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True,encoding='utf-8',bufsize=1,env=environment,cwd=ROOT)
        self.responses=queue.Queue()
        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read,daemon=True).start()
        if self.receive(120)!={'ready':True}: raise RuntimeError('POLICY_NOT_READY')

    def receive(self,timeout=30):
        try: line=self.responses.get(timeout=timeout)
        except queue.Empty: raise RuntimeError('POLICY_RESPONSE_TIMEOUT') from None
        if line is None: raise RuntimeError('POLICY_PROCESS_EXITED')
        result=json.loads(line)
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    def request(self,value):
        self.process.stdin.write(json.dumps(value)+'\n'); self.process.stdin.flush()
        return self.receive()

    def close(self):
        self.process.stdin.close()
        try: self.process.wait(timeout=10)
        except subprocess.TimeoutExpired: self.process.terminate(); self.process.wait(timeout=10)


def aggregate(rows):
    n=len(rows); successes=sum(r['outcome']=='SUCCESS' for r in rows); p=successes/n
    z=1.959963984540054; d=1+z*z/n
    center=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return {'episodes':n,'successes':successes,'success_rate':p,
        'success_wilson_95':[max(0,center-half),min(1,center+half)],
        'outcomes':dict(Counter(r['outcome'] for r in rows)),
        'collision_rate':sum(r['outcome']=='COLLISION' for r in rows)/n,
        'timeout_rate':sum(r['outcome']=='TIMEOUT' for r in rows)/n,
        **{'mean_'+k:float(np.mean([r[k] for r in rows])) for k in
           ('lateral_error_m','depth_error_m','orientation_error_rad','duration_s','path_length_m','inference_ms','clipped_steps')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=['expert','act'],required=True)
    parser.add_argument('--split',choices=['development','held_out'],required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/act-10-001')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-offset',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--seed-count',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    seeds=json.loads((ROOT/'configs/seeds.json').read_text())[args.split]
    frozen=json.loads(subprocess.check_output(['git','show','HEAD:configs/seeds.json'],cwd=ROOT))
    if seeds!=frozen[args.split]: raise ValueError('SEEDS_NOT_COMMITTED')
    args.output.mkdir(parents=True,exist_ok=False)
    # Measured Kit render initialization stall after 16 stage resets. Bound each
    # process to 10 episodes; both policies retain the same full fixed seed list.
    if args.seed_count is None and len(seeds)>10:
        combined=[]; parts=[]
        for offset in range(0,len(seeds),10):
            child=args.output/f'{args.output.name}-part-{offset//10:02d}'
            command=[sys.executable,str(Path(__file__).resolve()),'--policy',args.policy,
                '--split',args.split,'--checkpoint',str(args.checkpoint),'--output',str(child),
                '--seed-offset',str(offset),'--seed-count','10']
            completed=subprocess.run(command)
            summary_path=child/'summary.json'
            if not summary_path.exists(): raise RuntimeError('EVALUATION_CHILD_NO_SUMMARY')
            part=json.loads(summary_path.read_text())
            if completed.returncode!=0 or part['execution_status']!='PASS':
                raise RuntimeError('EVALUATION_CHILD_FAILED')
            parts.append(part)
            combined.extend(json.loads(line) for line in (child/'episodes.jsonl').read_text().splitlines())
        if [r['seed'] for r in combined]!=seeds: raise RuntimeError('INCOMPLETE_BATCHED_EVALUATION')
        for key in ('environment_config_hash','checkpoint_files_sha256','source_hashes','dataset_sha256'):
            if any(p[key]!=parts[0][key] for p in parts): raise RuntimeError('CHILD_PROTOCOL_MISMATCH')
        result={**parts[0],**aggregate(combined),'seeds':seeds,'completed_episodes':len(combined),
                'process_isolation':'10 episodes per fresh simulator process','batches':len(parts)}
        (args.output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined),encoding='utf-8')
        write_json(args.output/'summary.json',result)
        return
    if args.seed_count is not None:
        if args.seed_offset<0 or not 1<=args.seed_count<=10: raise ValueError('INVALID_SEED_SLICE')
        seeds=seeds[args.seed_offset:args.seed_offset+args.seed_count]
        if not seeds: raise ValueError('EMPTY_SEED_SLICE')
    raw=ROOT/'artifacts/runtime'/('eval-'+args.output.name); raw.mkdir(parents=True,exist_ok=False)
    config_hash=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
    pins=json.loads((ROOT/'references/source_manifest.json').read_text())
    data=json.loads((ROOT/'artifacts/datasets/validation.json').read_text())
    context={'policy':args.policy,'split':args.split,'seeds':seeds,
        'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hashes':{p:sha(ROOT/'scripts/local'/p) for p in ('evaluate.py','policy_worker.py','act.py','insertion.py')},
        'environment_config_hash':config_hash,'environment_config':CONFIG,
        'gr00t_commit':pins['upstream']['commit'],'lerobot_commit':pins['local']['lerobot']['commit'],
        'dataset_id':'ur10e_10','dataset_sha256':data['dataset_sha256'],
        'model_checkpoint_id':None,'checkpoint_files_sha256':None,
        'action_execution':'first action of fresh 16-step ACT chunk each control step; per-axis clipping recorded',
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        'execution_status':'FAILED'}
    worker=None; app=None; rows=[]; log=(raw/'policy.log').open('w',encoding='utf-8')
    write_json(args.output/'summary.json',context)
    try:
        if args.policy=='act':
            training=json.loads((ROOT/'artifacts/training'/args.checkpoint.name/'summary.json').read_text())
            if training['status']!='PASS' or training['dataset_sha256']!=data['dataset_sha256']:
                raise ValueError('TRAINING_EVIDENCE_MISMATCH')
            for name,digest in training['checkpoint_files_sha256'].items():
                if sha(args.checkpoint/name)!=digest: raise ValueError('CHECKPOINT_HASH_MISMATCH')
            context['model_checkpoint_id']=args.checkpoint.relative_to(ROOT).as_posix()
            context['checkpoint_files_sha256']=training['checkpoint_files_sha256']
            worker=PolicyProcess(args.checkpoint,log)
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'width':640,'height':480})
        from PIL import Image
        env=InsertionEnv(app); context['asset']=env.asset
        bounds=np.array([CONFIG['translation_limit_m']]*3+[CONFIG['rotation_limit_rad']]*3)
        dt=CONFIG['physics_dt']*CONFIG['control_substeps']
        with (args.output/'episodes.jsonl').open('w',encoding='utf-8') as results:
            for index,seed in enumerate(seeds):
                context['stage']=f'reset seed {seed}'
                context['completed_episodes']=len(rows)
                write_json(args.output/'summary.json',context)
                state=env.reset(seed); initial=state.copy(); trace=[]; frames=[]; positions=[state[:3].copy()]
                outcome=None; clipped=0; latency=[]
                if worker: worker.request({'op':'reset'})
                while outcome is None:
                    rgb=env.image()
                    if not np.allclose(env.state(),state,atol=1e-9,rtol=0): raise RuntimeError('CAPTURE_STATE_MISMATCH')
                    if env.steps==0: Image.fromarray(rgb).save(raw/f'{seed}_start.png')
                    if index==0: frames.append(rgb)
                    if worker:
                        start=time.perf_counter()
                        response=worker.request({'op':'predict','state':state.tolist(),'front':base64.b64encode(rgb.tobytes()).decode('ascii')})
                        chunk=np.asarray(response['chunk'],dtype=float)
                        if chunk.shape!=(16,6) or not np.isfinite(chunk).all(): raise ValueError('INVALID_ACTION_CHUNK')
                        latency.append(response['inference_ms']); wall_ms=(time.perf_counter()-start)*1000
                    else:
                        start=time.perf_counter(); action,_=expert(state); chunk=action[None]
                        wall_ms=(time.perf_counter()-start)*1000; latency.append(wall_ms)
                    action=np.clip(chunk[0],-bounds,bounds)
                    clipped+=int(np.any(action!=chunk[0]))
                    trace.append({'state':state.tolist(),'raw_chunk':chunk.tolist(),'action':action.tolist(),
                                  'timestamp':env.steps*dt,'request_ms':wall_ms,'inference_ms':latency[-1]})
                    state,outcome=env.step(action); positions.append(state[:3].copy())
                Image.fromarray(env.image()).save(raw/f'{seed}_end.png')
                write_json(raw/f'{seed}_trace.json',trace)
                if frames:
                    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x480',
                        '-r','15','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(raw/f'{seed}.mp4')],
                        input=b''.join(f.tobytes() for f in frames),check=True)
                row={'seed':seed,'outcome':outcome,'steps':env.steps,'initial_state':initial.tolist(),
                    'final_state':state.tolist(),'max_contact_n':env.max_contact,'clipped_steps':clipped,
                    'lateral_error_m':float(np.linalg.norm(state[:2])),'depth_error_m':float(abs(.03-state[2])),
                    'orientation_error_rad':float(np.linalg.norm(state[3:])),'duration_s':env.steps*dt,
                    'path_length_m':float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
                    'inference_ms':float(np.mean(latency)),
                    'raw_artifact_dir':raw.relative_to(ROOT).as_posix(),
                    'trace_sha256':sha(raw/f'{seed}_trace.json'),'end_image_sha256':sha(raw/f'{seed}_end.png')}
                rows.append(row); results.write(json.dumps(row,allow_nan=False)+'\n'); results.flush()
                print('EVALUATED',json.dumps(row),flush=True)
        context.update(aggregate(rows)); context['execution_status']='PASS'; context['stage']='complete'
    except Exception as exc:
        context['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        context['completed_episodes']=len(rows)
        write_json(args.output/'summary.json',context)
        if worker: worker.close()
        log.close()
        if app: app.close()


if __name__=='__main__':
    main()
