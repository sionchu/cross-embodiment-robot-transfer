"""Reproducible media preparation from the canonical evidence-linked edit manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'media/edit_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(path)],text=True))['streams'][0]


def prepare_source(manifest):
    folder=ROOT/'media/output/source';folder.mkdir(parents=True,exist_ok=True)
    for name,asset in manifest['assets'].items():
        if asset.get('source_type')!='PUBLIC_REAL_DATA': continue
        source=ROOT/asset['path']
        if sha(source)!=asset['sha256']: raise ValueError('PUBLIC_SOURCE_HASH_MISMATCH')
        start,end=asset['frame_range_half_open'];camera=asset['camera'].removesuffix('_view')
        clip=folder/f'real_franka_e{asset["episode"]}_{camera}_f{start:03d}-{end-1:03d}.mp4'
        if clip.exists() and 'clip_sha256' in asset and sha(clip)!=asset['clip_sha256']:
            raise ValueError('PREPARED_CLIP_CHANGED')
        if not clip.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(source),'-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                '-map_metadata','-1','-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p',str(clip)],check=True)
        info=probe(clip)
        if int(info['nb_read_frames'])!=end-start or info['r_frame_rate']!=f'{asset["fps"]}/1':
            raise ValueError('PREPARED_SOURCE_CADENCE_MISMATCH')
        contact=folder/f'{clip.stem}_contact.png'
        if not contact.exists():
            picks=f"select='eq(n,0)+eq(n,{(end-start)//2})+eq(n,{end-start-1})',tile=3x1"
            subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-vf',picks,'-frames:v','1',str(contact)],check=True)
        asset.update({'clip':clip.relative_to(ROOT).as_posix(),'clip_sha256':sha(clip),'clip_probe':info,
            'truth_status':'CLIP_VERIFIED','modifications':'frame trim and H.264 re-encoding; no speed or geometry changes'})
    manifest['source_preparation']={'script_sha256':sha(Path(__file__)),
        'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'licensing':'references/FIRe-source-credit.md and references/FIRe-Apache-2.0.txt'}
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print('SOURCE_CLIPS_VERIFIED; FINAL_ASSEMBLY_NOT_RUN')


# Rendering uses only recorded frames and structured evidence. The manifest owns
# timing, asset selection, captions and claims; this file owns their presentation.
import csv
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial.transform import Rotation

BG=(16,28,36);WHITE=(239,244,245);MUTED=(170,188,195)
TEAL=(68,180,158);AMBER=(235,169,98);PANEL=(28,43,52)


def read_json(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


@lru_cache(maxsize=80)
def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),size)


def text(canvas,xy,value,size=36,color=WHITE,bold=False,max_width=None):
    value=str(value)
    if max_width:
        while size>16 and font(size,bold).getlength(value)>max_width: size-=1
    ImageDraw.Draw(canvas).text(xy,value,font=font(size,bold),fill=color)


def panel(canvas,box,color=PANEL):
    ImageDraw.Draw(canvas).rounded_rectangle(box,radius=18,fill=color)


def paste_fit(canvas,image,box):
    x,y,w,h=map(int,box);fitted=ImageOps.contain(image,(w,h),Image.Resampling.LANCZOS)
    canvas.paste(fitted,(x+(w-fitted.width)//2,y+(h-fitted.height)//2))


class Video:
    def __init__(self,path):
        self.path=path;self.info=probe(path);self.fps=float(__import__('fractions').Fraction(self.info['r_frame_rate']))
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
        self.frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,int(self.info['height']),int(self.info['width']),3)
        if len(self.frames)!=int(self.info['nb_read_frames']): raise ValueError('DECODED_FRAME_COUNT_MISMATCH')
    def at(self,t):
        return Image.fromarray(self.frames[min(max(int(t*self.fps),0),len(self.frames)-1)])


class Renderer:
    def __init__(self,manifest):
        self.m=manifest;self.out=ROOT/'media/output';self.out.mkdir(exist_ok=True)
        self.study=read_json('artifacts/eval/data_efficiency.json')
        if self.study['status']!='COMPLETE' or len(self.study['records']) not in (6,9): raise ValueError('COMPLETE_STUDY_REQUIRED')
        self.tracks=list(dict.fromkeys(r['track'] for r in self.study['records']))
        self.diagnosis=None
        if any(s['mode']=='diagnosis' for s in manifest['shots']):
            d=manifest['selection']['diagnosis'];self.diagnosis={'figure':Image.open(ROOT/d['figure']).convert('RGB'),'seed':d['seed']}
            for key in ('target_evaluation','backbone_evaluation'):
                row=next(json.loads(l) for l in (ROOT/d[key]/'episodes.jsonl').read_text().splitlines() if json.loads(l)['seed']==d['seed'])
                if sha(ROOT/row['video']['path'])!=row['video']['sha256']: raise ValueError('DIAGNOSIS_CLIP_HASH_MISMATCH')
                self.diagnosis[key]=(Video(ROOT/row['video']['path']),row)
        self.groot=read_json('artifacts/environment/gr00t_50_contract.json')
        if self.groot['integration_status']!='PASS' or self.groot['model_execution']!='NOT_RUN': raise ValueError('REVIEW_GR00T_STATUS_LABELS')
        self.primary=read_json(manifest['selection']['primary_evaluation']+'/summary.json')
        self.training=read_json(manifest['selection']['training_run']+'/summary.json')
        self.target_demos=len(self.training['training_episode_indices'])
        if self.primary['successes']<1 or self.primary['execution_status']!='PASS': raise ValueError('LEARNED_SUCCESS_REQUIRED')
        self.model_hash=manifest['selection']['checkpoint_sha256']
        if sha(ROOT/manifest['selection']['learned_checkpoint']/'model.safetensors')!=self.model_hash: raise ValueError('SELECTED_CHECKPOINT_CHANGED')
        if self.primary['checkpoint_files_sha256']['model.safetensors']!=self.model_hash: raise ValueError('PRIMARY_CHECKPOINT_MISMATCH')
        self.videos={};self.receipts={}
        for name,asset in manifest['assets'].items():
            if 'receipt' in asset:
                receipt=read_json(asset['receipt']);self.receipts[name]=receipt
                if receipt['source_type']=='PROJECT_SIM_LEARNED':
                    if receipt['checkpoint_sha256']['model.safetensors']!=self.model_hash: raise ValueError('CAPTURE_CHECKPOINT_MISMATCH')
                if name=='close' and receipt['outcome']!='SUCCESS': raise ValueError('SUCCESSFUL_CLOSE_CAPTURE_REQUIRED')
                path=ROOT/receipt['video'];digest=receipt['video_sha256']
                asset.update({'path':receipt['video'],'sha256':digest,'truth_status':'CAPTURE_VERIFIED'})
            else:
                if sha(ROOT/asset['path'])!=asset['sha256']: raise ValueError('REAL_SOURCE_CHANGED')
                path=ROOT/asset['clip'];digest=asset['clip_sha256']
            if sha(path)!=digest: raise ValueError('MEDIA_ASSET_HASH_MISMATCH')
            self.videos[name]=Video(path)
        prior=read_json('artifacts/datasets/fire_motion_prior/manifest.json')
        ep=next(e for e in prior['episodes'] if e['episode']==29)
        if sha(ROOT/ep['trajectory'])!=ep['sha256']: raise ValueError('TRAJECTORY_CHANGED')
        self.trajectory=list(csv.DictReader((ROOT/ep['trajectory']).open()))
        self.xyz=np.array([[float(r[k]) for k in ('x','y','z')] for r in self.trajectory])
        self.rot=Rotation.from_rotvec([[float(r[k]) for k in ('rx','ry','rz')] for r in self.trajectory]).as_matrix()
        self.ecosystem=read_json('references/source_manifest.json')['ecosystem_visual_context']
        self.wall_rows=[json.loads(s) for s in (ROOT/manifest['selection']['wall_evaluation']/'episodes.jsonl').read_text().splitlines()]
        self.wall_rows=[r for r in self.wall_rows if r['seed'] in manifest['selection']['wall_seeds']]
        wall_summary=read_json(manifest['selection']['wall_evaluation']+'/summary.json')
        if wall_summary['execution_status']!='PASS' or [r['seed'] for r in self.wall_rows]!=manifest['selection']['wall_seeds']:
            raise ValueError('INCOMPLETE_PROSPECTIVE_EVALUATION_WALL')
        if wall_summary['checkpoint_files_sha256']['model.safetensors']!=self.model_hash: raise ValueError('WALL_CHECKPOINT_MISMATCH')
        primary_rows={r['seed']:r for r in (json.loads(line) for line in
            (ROOT/manifest['selection']['primary_evaluation']/'episodes.jsonl').read_text().splitlines())}
        for row in self.wall_rows:
            np.testing.assert_allclose(row['initial_state'],primary_rows[row['seed']]['initial_state'],atol=1e-9,rtol=0)
        for receipt in self.receipts.values():
            if receipt['source_type']=='PROJECT_SIM_LEARNED':
                repeat=read_json(receipt['evaluation']+'/summary.json')
                row=json.loads((ROOT/receipt['evaluation']/'episodes.jsonl').read_text())
                if repeat['environment_config_hash']!=self.primary['environment_config_hash']: raise ValueError('CAPTURE_TASK_CHANGED')
                np.testing.assert_allclose(row['initial_state'],primary_rows[row['seed']]['initial_state'],atol=1e-9,rtol=0)
        self.wall=[]
        for row in self.wall_rows:
            v=row['video'];path=ROOT/v['path'];trace_path=ROOT/row['raw_artifact_dir']/f"{row['seed']}_trace.json"
            if sha(path)!=v['sha256'] or sha(trace_path)!=row['trace_sha256']: raise ValueError('WALL_ASSET_HASH_MISMATCH')
            end=ROOT/row['raw_artifact_dir']/f"{row['seed']}_end.png"
            if sha(end)!=row['end_image_sha256']: raise ValueError('WALL_END_IMAGE_MISMATCH')
            self.wall.append((Video(path),json.loads(trace_path.read_text()),Image.open(end).convert('RGB')))
        self.curve=Image.open(ROOT/'artifacts/figures/data_efficiency.png').convert('RGB')
        self.overlays=self.out/'overlays';self.overlays.mkdir(exist_ok=True)
        self.make_training_plot()
        self.loss_plot=Image.open(self.overlays/'training_loss.png').convert('RGB')
        for shot in self.m['shots']:
            for reference in shot['evidence_refs']:
                if not (ROOT/reference).is_file(): raise FileNotFoundError(reference)
            shot['evidence_sha256']={reference:sha(ROOT/reference) for reference in shot['evidence_refs']}
            shot['truth_status']='ARTIFACTS_VERIFIED'

    def make_training_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=list(csv.DictReader((ROOT/self.m['selection']['training_run']/'loss.csv').open()))
        loss=np.array([float(r['loss']) for r in rows]);steps=np.array([int(r['step']) for r in rows])
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        ax.plot(steps,loss,color='#b5c9c5',linewidth=.5,alpha=.55,label='Recorded loss')
        ax.plot(steps[49:],np.convolve(loss,np.ones(50)/50,mode='valid'),color='#26806e',linewidth=2,label='50-update mean')
        ax.set(xlabel='Target optimization updates',ylabel='ACT loss',yscale='log',title=f'Actual target-only ACT / {self.target_demos} demonstrations')
        ax.legend();ax.spines[['top','right']].set_visible(False)
        fig.savefig(self.overlays/'training_loss.png',dpi=180);plt.close(fig)

    def base(self,shot):
        canvas=Image.new('RGB',(1920,1080),BG)
        text(canvas,(76,44),'GR00T N1.7 × UR10e',30,TEAL,True)
        text(canvas,(76,99),shot['caption'],58,WHITE,True,max_width=1770)
        text(canvas,(76,1015),'REAL DATA → RELATIVE EEF → NEW EMBODIMENT → MEASURED EVALUATION',24,MUTED)
        return canvas

    def source_full(self,t):
        return self.videos['real_hook'].at(t) if t<4 else self.videos['real_wrist'].at(t-4)

    def draw_trajectory(self,canvas,box,fraction):
        x,y,w,h=box;panel(canvas,(x,y,x+w,y+h))
        basis=np.array([[.85,-.5,0],[.3,.5,-1.]])
        projected=self.xyz@basis.T;center=(projected.min(0)+projected.max(0))/2
        scale=min((w-110)/(np.ptp(projected[:,0])+.008),(h-170)/(np.ptp(projected[:,1])+.008))
        def point(p):
            z=(np.asarray(p)@basis.T-center)*scale
            return (x+w/2+z[0],y+h/2-z[1]+20)
        k=min(len(self.xyz)-1,max(1,int(fraction*(len(self.xyz)-1))))
        d=ImageDraw.Draw(canvas);d.line([point(p) for p in self.xyz[:k+1]],fill=TEAL,width=5)
        for i,color in [(0,WHITE),(-1,AMBER)]:
            a,b=point(self.xyz[i]);d.ellipse((a-9,b-9,a+9,b+9),outline=color,width=3)
        for j,color in enumerate([(220,105,101),(97,189,135),(100,154,214)]):
            d.line([point(self.xyz[k]),point(self.xyz[k]+self.rot[k,:,j]*.004)],fill=color,width=4)
        text(canvas,(x+28,y+22),'MEASURED SOURCE TRAJECTORY',27,TEAL,True,max_width=w-50)
        text(canvas,(x+28,y+h-66),'Task-normalized terminal EEF frame',24,MUTED,max_width=w-50)
        text(canvas,(x+28,y+h-36),f'{len(self.xyz)} recorded pose samples · no image calibration',19,MUTED,max_width=w-50)

    def box(self,canvas,rect,title,subtitle,color=TEAL):
        panel(canvas,rect);x,y,x2,y2=rect
        text(canvas,(x+25,y+22),title,34,color,True,max_width=x2-x-50)
        text(canvas,(x+25,y+75),subtitle,27,WHITE,max_width=x2-x-50)

    def frame(self,shot,t):
        c=self.base(shot);mode=shot['mode'];duration=shot['end_s']-shot['start_s']
        if mode=='real':
            paste_fit(c,self.videos['real_hook'].at(t),(70,220,970,730))
            paste_fit(c,self.videos['real_right'].at(t),(1150,320,590,500))
            text(c,(1150,226),'REAL FRANKA',42,WHITE,True)
            text(c,(1150,845),'Two recorded camera views',28,MUTED)
            text(c,(86,958),'FIRe public research data · episode 29 · wrist / right',27,MUTED)
        elif mode=='source':
            paste_fit(c,self.videos['real_wrist'].at(t),(70,205,900,760))
            text(c,(1080,260),'Franka peg insertion',48,WHITE,True,max_width=750)
            for i,line in enumerate(['Episode 29 / wrist_view','15 fps original recording','RGB + recorded state + actions','Third-party FIRe research data']):
                text(c,(1080,360+i*80),line,32,MUTED,max_width=730)
            text(c,(1080,770),'Inspect the experience',36,TEAL,True)
        elif mode=='bridge':
            fade=max(0,1-t/4);source=self.videos['real_wrist'].at(3.9).convert('L').convert('RGB')
            source=Image.blend(Image.new('RGB',source.size,BG),source,fade)
            paste_fit(c,source,(80,220,780,680))
            shift=min(1,max(0,(t-2)/3));self.draw_trajectory(c,(1000-900*shift,225,800,635),t/duration)
            if t>4.5:
                paste_fit(c,self.videos['hero'].at(0),(1020,270,820,530))
                text(c,(1050,820),'UR10e / Isaac Sim environment',28,MUTED)
            text(c,(100,918),'Δx   Δy   Δz     ΔRx   ΔRy   ΔRz',44,WHITE,True)
            text(c,(1040,918),'Source motion → feature prior',30,TEAL,True,max_width=790)
        elif mode=='ecosystem':
            text(c,(100,214),'NVIDIA cross-embodiment ecosystem',36,MUTED)
            for i,item in enumerate(self.ecosystem['items']):
                x=100+(i%3)*585;y=320+(i//3)*205
                category='Post-training checkpoint' if 'posttrain' in item['category'] else 'Human video pretraining' if item['label']=='Human video' else 'Base pretraining'
                self.box(c,(x,y,x+550,y+170),item['label'],category)
            text(c,(200,800),'GR00T N1.7',58,WHITE,True)
            text(c,(765,818),'→',70,TEAL,True)
            text(c,(940,800),'NEW_EMBODIMENT',58,WHITE,True)
            text(c,(940,891),'UR10e integration path',33,MUTED)
        elif mode=='target':
            paste_fit(c,self.videos['hero'].at(t),(65,210,1790,740))
            panel(c,(105,785,780,935));text(c,(130,800),'UR10e / ISAAC SIM',36,WHITE,True)
            text(c,(130,852),'EXPERT CONTROL · analogous task',28,TEAL)
            if t>=len(self.videos['hero'].frames)/15: text(c,(130,895),'Terminal frame held',22,MUTED)
        elif mode=='pipeline':
            self.box(c,(90,250,810,405),'REAL FIRe SOURCE','3 episodes → normalized EEF motion')
            self.box(c,(90,470,810,625),'SOURCE FEATURE PRETRAINING','1000 updates → full transfer / backbone-only' if 'source-backbone' in self.tracks else '1000 updates → target head / RNG restored')
            self.box(c,(1080,250,1810,405),'UR10e TARGET DATA','10 → 25 → 50 nested demonstrations')
            self.box(c,(1080,470,1810,625),'CANONICAL LeRobot v2','RGB / state6 / ΔEEF6 / modality.json')
            text(c,(430,410),'↓',50,TEAL,True);text(c,(1410,410),'↓',50,TEAL,True)
            text(c,(830,520),'+',65,TEAL,True)
            self.box(c,(90,745,935,925),'ACT · TRAINED + EVALUATED','Target-only vs two FIRe transfer recipes' if 'source-backbone' in self.tracks else 'Target-only versus FIRe-informed')
            self.box(c,(985,745,1810,925),'GR00T N1.7 · INTEGRATION READY','NEW_EMBODIMENT / compute deferred',AMBER)
        elif mode=='training':
            paste_fit(c,self.loss_plot,(75,240,1190,670))
            panel(c,(1320,240,1835,915));r=self.training
            text(c,(1345,275),'RUN RECEIPT',34,TEAL,True)
            lines=[r['device'].replace('NVIDIA GeForce ',''),'LeRobot ACT '+r['packages']['lerobot'],
                f'{len(r["training_episode_indices"])} target demonstrations',f'{r["steps"]} target updates','Checkpoint saved + reloaded',self.model_hash[:16]]
            for i,line in enumerate(lines):text(c,(1345,360+i*78),line,29,WHITE,max_width=465)
            text(c,(90,940),f'Actual final run: target-only ACT / {self.target_demos} demos',28,MUTED)
        elif mode=='wall':
            text(c,(82,196),f'First eight fixed seeds · target-only ACT / {self.target_demos} demos',31,MUTED)
            cfg=self.primary['environment_config']
            for i,(row,record) in enumerate(zip(self.wall_rows,self.wall,strict=True)):
                video,trace,end=record;x=60+(i%4)*465;y=274+(i//4)*350
                step=min(int(t*15),len(trace)-1);terminal=t>=row['steps']/15
                state=np.array(trace[step]['state']);aligned=np.linalg.norm(state[:2])<=cfg['lateral_tolerance_m'] and np.linalg.norm(state[3:])<=cfg['angular_tolerance_rad']
                phase=('SUCCESS' if row['outcome']=='SUCCESS' else 'FAIL') if terminal else 'APPROACH' if state[2]<-.035 else 'ALIGN' if not aligned else 'APPROACH' if state[2]<0 else 'INSERT'
                panel(c,(x,y,x+440,y+328));paste_fit(c,end if terminal else video.at(t),(x+8,y+40,424,250))
                text(c,(x+16,y+5),f'Seed {row["seed"]}',26,WHITE,True)
                text(c,(x+16,y+293),phase,24,TEAL if phase=='SUCCESS' else AMBER if phase=='FAIL' else MUTED,True)
            text(c,(82,975),'Observe → ACT → ΔEEF → UR10e → observe · 15 Hz / K=1 · terminal frames held',25,MUTED)
        elif mode=='payoff':
            paste_fit(c,self.source_full(t),(65,245,860,665))
            target_t=t;paste_fit(c,self.videos['close'].at(target_t),(985,245,870,665))
            text(c,(85,195),'REAL WORLD / FRANKA SOURCE',31,TEAL,True)
            text(c,(1000,195),'SIMULATION / TARGET-ONLY ACT',31,TEAL,True)
            text(c,(85,925),'Analogous insertion tasks · different embodiments',27,MUTED)
            text(c,(1000,925),f'ACT · {self.target_demos} demos · held-out {self.primary["successes"]}/{self.primary["episodes"]}',33,WHITE,True)
            if target_t>=(self.receipts['close']['frame_range'][1]-1)/15:
                panel(c,(1290,824,1790,900));text(c,(1320,835),'INSERT SUCCESS',37,TEAL,True)
        elif mode=='results':
            paste_fit(c,self.curve,(120,205,1680,660))
            pairs={(r['target_demos'],r['track']):r['success_rate'] for r in self.study['records']}
            gains=[n for n in (10,25,50) for t in self.tracks if t!='target-only' and pairs[n,t]>pairs[n,'target-only']]
            if not gains:
                text(c,(180,885),'No source-data gain from either recipe in this experiment',34,AMBER,True)
            else:
                text(c,(180,885),'Source gain at '+' / '.join(f'{n} demos' for n in sorted(set(gains)))+' · one training seed',34,TEAL,True)
            text(c,(180,942),'P0 smoke evaluation · one training seed · all failures included',29,MUTED)
        elif mode=='diagnosis':
            d=self.diagnosis
            paste_fit(c,d['figure'],(60,200,1000,560))
            text(c,(80,785),'Full transfer: transformer ignores UR10e input',32,AMBER,True,max_width=960)
            text(c,(80,840),'0-step control = target-only, bit-identical',27,MUTED,max_width=960)
            bb=next(r for r in self.study['records'] if r['track']=='source-backbone' and r['target_demos']==50)
            text(c,(80,885),f'Backbone-only · 50 demos: {bb["successes"]}/{bb["episodes"]} · mean lateral {bb["mean_lateral_error_m"]*1000:.2f} mm (1 mm clearance)',27,MUTED,max_width=960)
            for i,(key,label) in enumerate([('target_evaluation','TARGET-ONLY'),('backbone_evaluation','FIRe BACKBONE-ONLY')]):
                video,row=d[key];y=200+i*395
                panel(c,(1100,y,1860,y+370));paste_fit(c,video.at(t),(1110,y+45,740,280))
                text(c,(1120,y+6),f'{label} · 50 demos · seed {d["seed"]}',25,WHITE,True,max_width=720)
                done=t>=row['steps']/15
                text(c,(1120,y+330),f'{row["outcome"]} · final lateral {row["lateral_error_m"]*1000:.2f} mm' if done else 'running (recorded 15 Hz)',24,
                     (TEAL if row['outcome']=='SUCCESS' else AMBER) if done else MUTED,True)
        elif mode=='close':
            text(c,(120,192),'REAL FRANKA / FIRe SOURCE',28,TEAL,True)
            text(c,(1050,192),'SIMULATION / TARGET-ONLY ACT',28,TEAL,True)
            paste_fit(c,self.source_full(7.5),(95,230,710,530))
            paste_fit(c,self.videos['close'].at(10),(1030,230,785,530))
            text(c,(830,400),'ΔEEF',45,TEAL,True)
            text(c,(850,470),'→',80,TEAL,True)
            text(c,(120,810),'REAL DATA → MEASURED EVALUATION',43,WHITE,True)
            text(c,(120,895),'GR00T N1.7 · NEW_EMBODIMENT · INTEGRATION READY',36,TEAL,True,max_width=1690)
        return c

    def previews(self):
        stem=Path(self.m['delivery'].get('output_name','main.mp4')).stem
        folder=self.out/('review' if stem=='main' else f'review_{stem}');folder.mkdir(exist_ok=True)
        for shot in self.m['shots']:
            self.frame(shot,(shot['end_s']-shot['start_s'])/2).save(folder/f'{shot["shot_id"]}.png')
        print('ACTUAL_EVIDENCE_PREVIEW_FRAMES_WRITTEN; NATIVE_REVIEW_NOT_COMPLETED')

    def render(self):
        self.previews();output=self.out/self.m['delivery'].get('output_name','main.mp4')
        if output.exists(): raise FileExistsError('Existing main cut must be reviewed before replacement')
        fps=self.m['delivery']['fps'];frames=round(self.m['delivery']['duration_s']*fps)
        process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),
            '-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','4','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
        try:
            for shot in self.m['shots']:
                for f in range(round((shot['end_s']-shot['start_s'])*fps)):
                    process.stdin.write(self.frame(shot,f/fps).tobytes())
            process.stdin.close()
            if process.wait()!=0: raise RuntimeError('FINAL_VIDEO_ENCODER_FAILED')
        except BaseException:
            if process.poll() is None: process.terminate();process.wait()
            raise
        info=probe(output)
        if (int(info['nb_read_frames']),info['r_frame_rate'],info['width'],info['height'])!=(frames,'30/1',1920,1080):
            raise ValueError('FINAL_DELIVERY_FORMAT_MISMATCH')
        self.m['verification']['assembly']='FORMAT_AND_EVIDENCE_CHECKS_PASS'
        self.m['output']={'path':output.relative_to(ROOT).as_posix(),'sha256':sha(output),'probe':info,
            'script_sha256':sha(Path(__file__)),'playback':'15Hz recorded footage frame-duplicated to30fps; static terminal holds; no audio'}
        MANIFEST.write_text(json.dumps(self.m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        captions=self.out/'captions';captions.mkdir(exist_ok=True)
        def stamp(seconds):return f'{int(seconds)//3600:02d}:{int(seconds)//60%60:02d}:{int(seconds)%60:02d},000'
        (captions/(output.stem+'.srt')).write_text('\n\n'.join(f'{i+1}\n{stamp(s["start_s"])} --> {stamp(s["end_s"])}\n{s["caption"]}' for i,s in enumerate(self.m['shots']))+'\n',encoding='utf-8')
        with (self.out/('shot_list.csv' if output.stem=='main' else f'shot_list_{output.stem}.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['shot_id','start_s','end_s','caption','claim','evidence_refs']);w.writeheader()
            for s in self.m['shots']:w.writerow({k:'; '.join(s[k]) if k=='evidence_refs' else s[k] for k in w.fieldnames})
        (self.out/MANIFEST.name).write_bytes(MANIFEST.read_bytes())
        print(json.dumps(self.m['output'],indent=2))

    def derivatives(self):
        main=self.out/'main.mp4'
        if self.m['verification'].get('main_file_review_sha256')!=sha(main):
            raise ValueError('REVIEW_CURRENT_MAIN_CUT_FIRST')
        hero=self.out/'hero.mp4';poster=self.out/'poster.png'
        if hero.exists() or poster.exists(): raise FileExistsError('Preserve existing derivatives before rebuilding')
        # Recorded clips retain real time. Only the abstract trajectory animation
        # occupies four seconds here instead of eight in the main cut.
        process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30',
            '-i','pipe:0','-an','-c:v','libx264','-preset','fast','-threads','4','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(hero)],stdin=subprocess.PIPE)
        try:
            for i in range(360):
                t=i/30
                shot,u=(self.m['shots'][0],t) if t<3 else (self.m['shots'][2],(t-3)*2) if t<7 else (self.m['shots'][8],t-7)
                c=self.frame(shot,u)
                ImageDraw.Draw(c).rectangle((0,1003,1920,1080),fill=BG)
                text(c,(76,1015),'SOURCE: REAL FIRe DATA     |     TARGET: ISAAC SIM     |     GR00T N1.7: INTEGRATION READY',25,TEAL,True)
                process.stdin.write(c.tobytes())
            process.stdin.close()
            if process.wait()!=0: raise RuntimeError('HERO_ENCODER_FAILED')
        except BaseException:
            if process.poll() is None: process.terminate();process.wait()
            raise
        c=Image.new('RGB',(1920,1080),BG)
        text(c,(78,47),'GR00T N1.7 × UR10e',76,WHITE,True)
        text(c,(82,148),'Real-World Cross-Embodiment Robot Learning',38,TEAL)
        text(c,(100,245),'REAL FRANKA / FIRe SOURCE',30,TEAL,True)
        text(c,(1070,245),'SIMULATION / LEARNED UR10e',30,TEAL,True)
        paste_fit(c,self.source_full(2),(80,310,710,585))
        paste_fit(c,self.videos['evaluation'].at(10),(1040,310,790,585))
        text(c,(825,460),'ΔEEF',50,TEAL,True);text(c,(850,530),'→',78,TEAL,True)
        text(c,(90,930),f'Target-only ACT · {self.target_demos} demos · held-out {self.primary["successes"]}/{self.primary["episodes"]}',36,WHITE,True)
        text(c,(90,1000),'GR00T NEW_EMBODIMENT integration ready · measured 10 / 25 / 50 comparison',28,MUTED)
        c.save(poster)
        info=probe(hero)
        if (int(info['nb_read_frames']),info['r_frame_rate'])!=(360,'30/1'): raise ValueError('HERO_FORMAT_MISMATCH')
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(hero),'-f','null','-'],check=True)
        self.m['derivatives']={'hero':{'path':hero.relative_to(ROOT).as_posix(),'sha256':sha(hero),'probe':info,
            'segments':[{'shot':'S01_REAL','duration_s':3},{'shot':'S03_BRIDGE','duration_s':4,'abstract_animation_rate':2},
                {'shot':'S09_PAYOFF','duration_s':5}],'recorded_motion_speed':'unchanged'},
            'poster':{'path':poster.relative_to(ROOT).as_posix(),'sha256':sha(poster),'size':[1920,1080]},
            'script_sha256':sha(Path(__file__)),'visual_review':'NOT_RUN'}
        credits=self.out/'credits';credits.mkdir(exist_ok=True)
        for name in ('FIRe-source-credit.md','FIRe-Apache-2.0.txt'):
            (credits/name).write_bytes((ROOT/'references'/name).read_bytes())
        MANIFEST.write_text(json.dumps(self.m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        (self.out/'edit_manifest.json').write_bytes(MANIFEST.read_bytes())
        print(json.dumps(self.m['derivatives'],indent=2))

    def package(self):
        if self.m['verification'].get('main_file_review_sha256')!=sha(self.out/'main.mp4'):
            raise ValueError('REVIEW_CURRENT_MAIN_CUT_FIRST')
        folder=self.out/'sim/wall';folder.mkdir(exist_ok=True)
        for row in self.wall_rows:
            for source in (ROOT/row['video']['path'],ROOT/row['raw_artifact_dir']/f"{row['seed']}_end.png"):
                (folder/source.name).write_bytes(source.read_bytes())
        for mode,name in [('bridge','source_trajectory'),('pipeline','architecture'),('results','evaluation')]:
            shot=next(s for s in self.m['shots'] if s['mode']==mode)
            self.frame(shot,(shot['end_s']-shot['start_s'])/2).save(self.overlays/f'{name}.png')
        transition=self.overlays/'task_space_transition.mp4'
        if not transition.exists():
            subprocess.run(['ffmpeg','-v','error','-ss','8','-i',str(self.out/'main.mp4'),'-t','8','-an',
                '-c:v','libx264','-threads','1','-crf','19','-pix_fmt','yuv420p',str(transition)],check=True)
        evidence=self.out/'evidence';evidence.mkdir(exist_ok=True)
        refs=['artifacts/eval/data_efficiency.json','artifacts/datasets/fire_source.json',
            'artifacts/datasets/fire_task_audit.json','artifacts/datasets/fire_motion_prior/manifest.json',
            self.m['selection']['training_run']+'/summary.json',self.m['selection']['training_run']+'/loss.csv']
        for ref in refs:
            destination=evidence/ref;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((ROOT/ref).read_bytes())
        files=[]
        for f in sorted(self.out.rglob('*')):
            if not f.is_file() or 'review' in f.relative_to(self.out).parts or f.name=='edit_manifest.json': continue
            files.append({'path':f.relative_to(self.out).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size})
        self.m['package']={'root':'media/output','files':files,'heavy_files_in_git':False,
            'regeneration':'scripts/local/README.md; media/build.py; media/capture.py',
            'native_desktop_verification':'NOT_COMPLETED'}
        MANIFEST.write_text(json.dumps(self.m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        (self.out/'edit_manifest.json').write_bytes(MANIFEST.read_bytes())
        print('EDIT_PACKAGE_HASHED',len(files))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-source',action='store_true')
    mode.add_argument('--preview',action='store_true')
    mode.add_argument('--render',action='store_true')
    mode.add_argument('--derivatives',action='store_true')
    mode.add_argument('--package',action='store_true')
    parser.add_argument('--manifest',type=Path,default=MANIFEST,help='Edit manifest; v1 default is preserved.')
    args=parser.parse_args();MANIFEST=args.manifest.resolve();manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if args.prepare_source: prepare_source(manifest)
    elif args.preview: Renderer(manifest).previews()
    elif args.derivatives: Renderer(manifest).derivatives()
    elif args.package: Renderer(manifest).package()
    else: Renderer(manifest).render()
