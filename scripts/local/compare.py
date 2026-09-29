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


def verified_run(name, require_transitions=False):
    seeds=json.loads((ROOT/'configs/seeds.json').read_text())['held_out']
    folder=ROOT/'artifacts/eval'/name
    summary=json.loads((folder/'summary.json').read_text())
    if summary['environment_config']!=CONFIG: raise ValueError('TASK_VERSION_MISMATCH')
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
        detailed=all('next_state' in t and 'max_contact_n' in t and 'terminal' in t for t in trace)
        if require_transitions and not detailed: raise ValueError('DETAILED_TRANSITIONS_REQUIRED')
        if detailed:
            np.testing.assert_array_equal(trace[-1]['next_state'],row['final_state'])
            if trace[-1]['max_contact_n']!=row['max_contact_n']:
                raise ValueError('CONTACT_SUMMARY_MISMATCH')
            for i,t in enumerate(trace):
                expected_terminal=terminal(t['next_state'],t['max_contact_n'],i+1)
                if t['terminal']!=expected_terminal or (i<len(trace)-1 and expected_terminal is not None):
                    raise ValueError('TRACE_TERMINAL_MISMATCH')
                if i: np.testing.assert_array_equal(trace[i-1]['next_state'],t['state'])
        raw=np.asarray([t['raw_chunk'] for t in trace])
        expected_horizon=summary.get('action_chunk_size',16 if summary['policy']=='act' else 1)
        if raw.shape!=(row['steps'],expected_horizon,6) or not np.isfinite(raw).all():
            raise ValueError('INVALID_RECORDED_ACTION_CHUNKS')
        applied=np.asarray([t['action'] for t in trace])
        np.testing.assert_array_equal(applied,np.clip(raw[:,0],-bounds,bounds))
        if int(np.any(applied!=raw[:,0],axis=1).sum())!=row['clipped_steps']:
            raise ValueError('CLIPPING_COUNT_MISMATCH')
        np.testing.assert_allclose([t['timestamp'] for t in trace],np.arange(row['steps'])/15,atol=1e-9,rtol=0)
    return summary,rows



def data_efficiency(names):
    """Only complete, paired actual runs enter the measured learning curve."""
    records=[];reference=None;reference_config=None;reference_training=None
    controls=('steps','batch_size','training_seed','backbone_initialization')
    for name in names:
        run,rows=verified_run(name,require_transitions=True)
        if run['policy']!='act': raise ValueError('STUDY_REQUIRES_ACT')
        training_folder=ROOT/'artifacts/training'/Path(run['model_checkpoint_id']).name
        training=json.loads((training_folder/'summary.json').read_text())
        config=json.loads((training_folder/'config.json').read_text())
        n=len(training['training_episode_indices']);track=training.get('track','target-only')
        if n not in (10,25,50) or track not in ('target-only','source-informed'):
            raise ValueError('INVALID_STUDY_CONDITION')
        if training['status']!='PASS' or run['checkpoint_files_sha256']!=training['checkpoint_files_sha256']:
            raise ValueError('STUDY_CHECKPOINT_MISMATCH')
        if run['dataset_sha256']!=training['dataset_sha256']: raise ValueError('STUDY_DATASET_MISMATCH')
        if reference is None: reference=rows;reference_config=config;reference_training=training;reference_run=run
        for key in ('environment_config_hash','seeds','action_chunk_size','action_execution'):
            if run[key]!=reference_run[key]: raise ValueError('STUDY_EXECUTION_PROTOCOL_MISMATCH')
        if run['source_hashes']['insertion.py']!=reference_run['source_hashes']['insertion.py']:
            raise ValueError('STUDY_SIMULATOR_SOURCE_MISMATCH')
        if config!=reference_config or any(training[k]!=reference_training[k] for k in controls):
            raise ValueError('STUDY_TRAINING_PROTOCOL_MISMATCH')
        if training['sampling']['mode']!='precision-balanced': raise ValueError('STUDY_SAMPLING_MISMATCH')
        for x,y in zip(reference,rows,strict=True):
            np.testing.assert_allclose(x['initial_state'],y['initial_state'],atol=1e-9,rtol=0)
        prior=training.get('source_pretraining')
        if track=='source-informed':
            if (not prior or prior['steps']!=1000 or prior['source_transitions']!=381
                    or not prior['target_action_head_restored_exactly'] or not prior['target_rng_restored']
                    or prior['features_before_sha256']==prior['features_after_sha256']):
                raise ValueError('SOURCE_INFLUENCE_NOT_VERIFIED')
        elif prior: raise ValueError('TARGET_ONLY_CONTAINS_SOURCE_TRAINING')
        records.append({'target_demos':n,'track':track,'evaluation_run':name,
            'successes':run['successes'],'episodes':run['episodes'],'success_rate':run['success_rate'],
            'wilson_low':run['success_wilson_95'][0],'wilson_high':run['success_wilson_95'][1],
            'mean_lateral_error_m':run['mean_lateral_error_m'],'mean_orientation_error_rad':run['mean_orientation_error_rad'],
            'target_steps':training['steps'],'source_steps':0 if not prior else prior['steps'],
            'source_episodes':0 if not prior else len(prior['source_episodes']),
            'dataset_sha256':run['dataset_sha256'],'checkpoint_sha256':run['checkpoint_files_sha256']['model.safetensors'],
            'normalization_sha256':run['checkpoint_files_sha256']['normalization.json']})
    pairs={(r['target_demos'],r['track']):r for r in records}
    if len(pairs)!=len(records): raise ValueError('DUPLICATE_STUDY_CONDITION')
    sizes=sorted({r['target_demos'] for r in records})
    for n in sizes:
        if any((n,t) not in pairs for t in ('target-only','source-informed')): raise ValueError('INCOMPLETE_STUDY_PAIR')
        a,b=[pairs[n,t] for t in ('target-only','source-informed')]
        if any(a[k]!=b[k] for k in ('dataset_sha256','normalization_sha256')): raise ValueError('PAIRED_DATA_MISMATCH')
    records=[pairs[n,t] for n in sizes for t in ('target-only','source-informed')]
    target=ROOT/'artifacts/eval/data_efficiency'
    with target.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    report={'status':'COMPLETE' if sizes==[10,25,50] else 'PARTIAL','records':records,
        'protocol':'same task/controller/detector; fixed20 seeds; one training seed;5000 target steps; source extra1000 steps reported separately',
        'task_equivalence':'ANALOGOUS_TASK','interpretation':'target-data efficiency, not compute efficiency or a broad benchmark',
        'all_raw_traces_actions_terminal_outcomes_and_initial_states':'VERIFIED',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for track,label,color,marker in [('target-only','Target-only ACT','#536b79','o'),
            ('source-informed','FIRe-informed ACT (+3 source demos)','#368777','s')]:
        rr=[pairs[n,track] for n in sizes];rates=np.array([r['success_rate'] for r in rr])*100
        low=np.array([r['wilson_low'] for r in rr])*100;high=np.array([r['wilson_high'] for r in rr])*100
        ax.errorbar(sizes,rates,yerr=np.maximum(np.stack([rates-low,high-rates]),0),label=label,color=color,marker=marker,capsize=4)
    ax.set(xticks=[10,25,50],xlim=(5,55),ylim=(-5,105),xlabel='Target demonstrations',ylabel='Closed-loop success (%)',
        title='UR10e target-data efficiency / fixed 20 held-out seeds')
    ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False)
    ax.text(.02,.76,'95% Wilson intervals; one training seed',transform=ax.transAxes,va='top',fontsize=8)
    fig.savefig(ROOT/'artifacts/figures/data_efficiency.png',dpi=160);plt.close(fig)
    table=['| Target demos | Target-only ACT | FIRe-informed ACT |','|---|---|---|']
    for n in sizes:
        results=[f"{pairs[n,t]['successes']}/{pairs[n,t]['episodes']} ({pairs[n,t]['success_rate']:.0%})" for t in ('target-only','source-informed')]
        table.append(f'| {n} | '+ ' | '.join(results)+' |')
    missing=sorted({10,25,50}-set(sizes))
    table+=['','Both tracks use 5000 target updates. FIRe-informed adds 3 real source episodes and 1000 source updates.',
        'Remaining target-data sizes: '+', '.join(map(str,missing))+'.' if missing else 'All three paired target-data sizes were executed.',
        '', '[Structured evidence](artifacts/eval/data_efficiency.json) · [Measured CSV](artifacts/eval/data_efficiency.csv)',
        '', '![Measured target-data efficiency](artifacts/figures/data_efficiency.png)']
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');start='<!-- STUDY:START -->';end='<!-- STUDY:END -->'
    if start not in text:
        text=text.replace('## Closed-loop recovery','## Data-efficiency experiment\n\n'+start+'\n'+end+'\n\n## Closed-loop recovery')
    before,rest=text.split(start,1);_,after=rest.split(end,1)
    p.write_text(before+start+'\n'+'\n'.join(table)+'\n'+end+after,encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'conditions':len(records),'sizes':sizes}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-run',default='expert-002')
    parser.add_argument('--act-run',default='act-10-002')
    parser.add_argument('--recovered-run',help='Additional complete ACT run; preserve original baseline in the comparison.')
    parser.add_argument('--study-runs',nargs='+',help='Actual paired ACT evaluation run names at10/25/50 target demos.')
    args=parser.parse_args()
    if args.study_runs:
        data_efficiency(args.study_runs)
        return
    names=[args.expert_run,args.act_run]+([args.recovered_run] if args.recovered_run else [])
    labels=['Expert','ACT baseline']+(['ACT recovered'] if args.recovered_run else [])
    seeds=json.loads((ROOT/'configs/seeds.json').read_text())['held_out']
    runs=[]; episodes=[]
    for name in names:
        summary,rows=verified_run(name);runs.append(summary);episodes.append(rows)
    if [r['policy'] for r in runs]!=['expert']+['act']*(len(runs)-1): raise ValueError('POLICY_LABEL_MISMATCH')
    for run,other_rows in zip(runs[1:],episodes[1:],strict=True):
        for key in ('environment_config_hash','dataset_sha256','seeds'):
            if runs[0][key]!=run[key]: raise ValueError('COMPARISON_PROTOCOL_MISMATCH')
        if runs[0]['source_hashes']['insertion.py']!=run['source_hashes']['insertion.py']:
            raise ValueError('SIMULATOR_SOURCE_MISMATCH')
        for a,b in zip(episodes[0],other_rows,strict=True):
            np.testing.assert_allclose(a['initial_state'],b['initial_state'],atol=1e-9,rtol=0)
    groot=json.loads((ROOT/'artifacts/environment/gr00t_local_contract.json').read_text())
    if groot['model_execution']!='NOT_RUN': raise ValueError('UPDATE_COMPARISON_FOR_EXECUTED_GR00T')
    records=[]
    for label,run in zip(labels,runs,strict=True):
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
    table.extend(['',' · '.join(f'[{label} episodes](artifacts/eval/{name}/episodes.jsonl)' for label,name in zip(labels,names,strict=True))])
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
    positions=np.arange(len(runs))
    ax.bar(positions,rates,color=['#536b79','#bb7540','#368777'][:len(runs)],width=.5)
    ax.errorbar(positions,rates,yerr=np.maximum(np.stack([rates-ci[:,0],ci[:,1]-rates]),0),fmt='none',ecolor='#202830',capsize=5)
    ax.set(xticks=list(positions)+[len(runs)],xticklabels=labels+['GR00T N1.7'],xlim=(-.5,len(runs)+.5),ylim=(-5,110),ylabel='Insertion success (%)',
           title='UR10e simulation: 20 fixed held-out seeds')
    ax.text(len(runs),45,'N/A\nNot executed',ha='center',color='#64717b')
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
    for label,rows in zip(labels,episodes,strict=True):
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
