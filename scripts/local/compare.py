"""Generate the compact table/plot from complete matched held-out evaluations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from evaluate import aggregate, ROOT
from insertion import CONFIG, terminal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-run',default='expert-002')
    parser.add_argument('--act-run',default='act-10-002')
    args=parser.parse_args()
    names=(args.expert_run,args.act_run)
    seeds=json.loads((ROOT/'configs/seeds.json').read_text())['held_out']
    runs=[]; episodes=[]
    for name in names:
        folder=ROOT/'artifacts/eval'/name
        summary=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
        if summary['execution_status']!='PASS' or summary['split']!='held_out' or [r['seed'] for r in rows]!=seeds:
            raise ValueError('INCOMPLETE_EVALUATION')
        for key,value in aggregate(rows).items():
            if summary[key]!=value: raise ValueError('AGGREGATE_MISMATCH')
        bounds=np.array([CONFIG['translation_limit_m']]*3+[CONFIG['rotation_limit_rad']]*3)
        for row in rows:
            if terminal(row['final_state'],row['max_contact_n'],row['steps'])!=row['outcome']:
                raise ValueError('TERMINAL_OUTCOME_MISMATCH')
            trace_file=ROOT/row['raw_artifact_dir']/f"{row['seed']}_trace.json"
            if hashlib.sha256(trace_file.read_bytes()).hexdigest()!=row['trace_sha256']:
                raise ValueError('TRACE_HASH_MISMATCH')
            trace=json.loads(trace_file.read_text())
            if len(trace)!=row['steps']: raise ValueError('TRACE_LENGTH_MISMATCH')
            np.testing.assert_array_equal(trace[0]['state'],row['initial_state'])
            raw=np.asarray([t['raw_chunk'] for t in trace])
            expected_horizon=summary.get('action_chunk_size',16 if summary['policy']=='act' else 1)
            if raw.shape!=(row['steps'],expected_horizon,6) or not np.isfinite(raw).all():
                raise ValueError('INVALID_RECORDED_ACTION_CHUNKS')
            applied=np.asarray([t['action'] for t in trace])
            np.testing.assert_array_equal(applied,np.clip(raw[:,0],-bounds,bounds))
            if int(np.any(applied!=raw[:,0],axis=1).sum())!=row['clipped_steps']:
                raise ValueError('CLIPPING_COUNT_MISMATCH')
            np.testing.assert_allclose([t['timestamp'] for t in trace],np.arange(row['steps'])/15,atol=1e-9,rtol=0)
        runs.append(summary); episodes.append(rows)
    if [r['policy'] for r in runs]!=['expert','act']: raise ValueError('POLICY_LABEL_MISMATCH')
    for key in ('environment_config_hash','dataset_sha256','seeds'):
        if runs[0][key]!=runs[1][key]: raise ValueError('COMPARISON_PROTOCOL_MISMATCH')
    if runs[0]['source_hashes']['insertion.py']!=runs[1]['source_hashes']['insertion.py']:
        raise ValueError('SIMULATOR_SOURCE_MISMATCH')
    for a,b in zip(*episodes,strict=True):
        np.testing.assert_allclose(a['initial_state'],b['initial_state'],atol=1e-9,rtol=0)
    groot=json.loads((ROOT/'artifacts/environment/gr00t_local_contract.json').read_text())
    if groot['model_execution']!='NOT_RUN': raise ValueError('UPDATE_COMPARISON_FOR_EXECUTED_GR00T')
    records=[]
    for label,run in zip(('Expert','ACT'),runs,strict=True):
        records.append({'policy':label,'training_data':'scripted' if label=='Expert' else '10 target demos',
            'actual_training':'N/A' if label=='Expert' else 'yes','closed_loop_evaluated':'yes',
            'episodes':run['episodes'],'successes':run['successes'],'success_rate':run['success_rate'],
            'collision_rate':run['collision_rate'],'timeout_rate':run['timeout_rate']})
    records.append({'policy':'GR00T N1.7','training_data':'target-compatible','actual_training':'no',
        'closed_loop_evaluated':'no','episodes':'N/A','successes':'N/A','success_rate':'N/A','collision_rate':'N/A','timeout_rate':'N/A'})
    with (ROOT/'artifacts/eval/comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    table=['| Policy | Training data | Actual training | Closed-loop evaluated | Success |',
           '|---|---|---|---|---|']
    for row in records:
        success='N/A' if row['success_rate']=='N/A' else f"{row['successes']}/{row['episodes']} ({row['success_rate']:.0%})"
        table.append(f"| {row['policy']} | {row['training_data']} | {row['actual_training']} | {row['closed_loop_evaluated']} | {success} |")
    table.extend(['',f"[Expert episodes](artifacts/eval/{args.expert_run}/episodes.jsonl) · "
        f"[ACT episodes](artifacts/eval/{args.act_run}/episodes.jsonl) · "
        f"[ACT checkpoint evidence](artifacts/training/{Path(runs[1]['model_checkpoint_id']).name}/summary.json)"])
    readme=ROOT/'README.md' ; text=readme.read_text(encoding='utf-8')
    start='<!-- RESULTS:START -->'; end='<!-- RESULTS:END -->'
    if start not in text: raise ValueError('README_RESULT_MARKERS_MISSING')
    before,rest=text.split(start,1); _,after=rest.split(end,1)
    readme.write_text(before+start+'\n'+'\n'.join(table)+'\n'+end+after,encoding='utf-8',newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,3.6),layout='constrained')
    rates=np.array([r['success_rate'] for r in runs])*100
    ci=np.array([r['success_wilson_95'] for r in runs])*100
    ax.bar([0,1],rates,color=['#536b79','#bb7540'],width=.5)
    ax.errorbar([0,1],rates,yerr=np.maximum(np.stack([rates-ci[:,0],ci[:,1]-rates]),0),fmt='none',ecolor='#202830',capsize=5)
    ax.set(xticks=[0,1,2],xticklabels=['Expert','ACT','GR00T N1.7'],xlim=(-.5,2.5),ylim=(-5,110),ylabel='Insertion success (%)',
           title='UR10e simulation: 20 fixed held-out seeds')
    ax.text(2,45,'N/A\nNot executed',ha='center',color='#64717b')
    ax.text(.02,.96,'95% Wilson intervals; 10 target training demos',transform=ax.transAxes,va='top',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    figures=ROOT/'artifacts/figures'; figures.mkdir(exist_ok=True)
    fig.savefig(figures/'held_out_success.png',dpi=160); plt.close(fig)
    # Retain the first learned-policy failure, chosen by seed order rather than appearance.
    failure=next((r for r in episodes[1] if r['outcome']!='SUCCESS'),None)
    if failure:
        source=ROOT/failure['raw_artifact_dir']/f"{failure['seed']}_end.png"
        if hashlib.sha256(source.read_bytes()).hexdigest()!=failure['end_image_sha256']:
            raise ValueError('FAILURE_IMAGE_HASH_MISMATCH')
        shutil.copyfile(source,figures/'act_first_failure.png')
    videos=[]
    for label,rows in zip(('Expert','ACT'),episodes,strict=True):
        for row in rows:
            video=ROOT/row['raw_artifact_dir']/f"{row['seed']}.mp4"
            if not video.exists(): continue
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-count_frames','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(video)],text=True))['streams'][0]
            if int(probe['nb_read_frames'])!=row['steps'] or probe['r_frame_rate']!='15/1':
                raise ValueError('EVALUATION_VIDEO_CADENCE_MISMATCH')
            videos.append({'policy':label,'seed':row['seed'],'outcome':row['outcome'],
                'path':video.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
                'frames':row['steps'],'fps':15,'width':probe['width'],'height':probe['height'],
                'source':'actual Isaac Sim front RGB; pre-action frames; unaltered 15 Hz playback'})
    (ROOT/'artifacts/video_manifest.json').write_text(json.dumps(videos,indent=2)+'\n',encoding='utf-8',newline='\n')
    evidence={'comparison':'PASS','matched_initial_states':True,
        'all_trace_hashes_and_chunk_execution':'PASS','all_geometric_terminal_outcomes':'PASS','seeds':seeds,
        'matplotlib':matplotlib.__version__,'numpy':np.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_hash_encoding':'JSON parsed then sorted compact UTF-8; independent of CRLF/LF',
        'inputs_sha256':{f'artifacts/eval/{name}/{file}':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest() for name,run,rows in zip(names,runs,episodes,strict=True) for file,value in [('summary.json',run),('episodes.jsonl',rows)]},
        'first_failure_seed':None if failure is None else failure['seed']}
    (ROOT/'artifacts/eval/comparison_validation.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(table))


if __name__=='__main__': main()
