"""Capture one actual expert/learned episode through the canonical evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/local'))
import evaluate
from insertion import InsertionEnv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    specs=json.loads((ROOT/'media/cameras.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=list(specs),required=True)
    parser.add_argument('--policy',choices=['expert','act'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New canonical evaluation receipt directory.')
    args=parser.parse_args()
    if args.policy=='act' and not args.checkpoint: parser.error('ACT checkpoint is required')
    seeds=json.loads((ROOT/'configs/seeds.json').read_text())['held_out']
    if args.seed not in seeds: parser.error('Capture uses an existing fixed held-out seed')
    spec=specs[args.view];width,height=spec['resolution'];instances=[]
    destination=ROOT/'media/output/sim'/args.output.name
    if destination.exists(): raise FileExistsError(destination)

    class CaptureEnv(InsertionEnv):
        def __init__(self,app):
            super().__init__(app);self.frames=[];instances.append(self)

        def reset(self,seed):
            state=super().reset(seed)
            from pxr import Usd
            if spec.get('existing_policy_camera'):
                camera=next(p for p in self.stage.Traverse() if p.GetTypeName()=='Camera')
                self.media_rgb=self.rgb
                position=spec['position_m'];target=spec['look_at_m']
            else:
                position=spec.get('position_m') or (self.hole_p+spec['position_from_hole_m']).tolist()
                target=spec.get('look_at_m') or (self.hole_p+spec['look_at_from_hole_m']).tolist()
                camera=self.rep.functional.create.camera(position=tuple(position),look_at=tuple(target),
                    look_at_up_axis=(0,0,1),focal_length=spec['focal_length_mm'],clipping_range=(.01,100.),name=args.view)
                self.media_product=self.rep.create.render_product(camera,(width,height))
                self.media_rgb=self.rep.annotators.get('rgb');self.media_rgb.attach(self.media_product)
            matrix=self.UG.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            self.camera={'role':args.view,'specification':spec,'position_world_m':position,'look_at_world_m':target,
                'world_transform_matrix_row_major':[[float(matrix[i][j]) for j in range(4)] for i in range(4)],
                'prim_path':str(camera.GetPath()),'world_up_axis':'Z','world_unit':'m'}
            return state

        def image(self):
            state=self.state().copy();time=self.SM.get_simulation_time()
            front=super().image()
            for _ in range(10):
                frame=self.media_rgb.get_data()
                if getattr(frame,'size',0): break
                self.rep.orchestrator.step(rt_subframes=2,delta_time=0.,pause_timeline=False)
            frame=np.asarray(frame)[...,:3].copy()
            if frame.shape!=(height,width,3): raise RuntimeError('MEDIA_RGB_SHAPE_MISMATCH')
            np.testing.assert_allclose(self.state(),state,atol=1e-9,rtol=0)
            if abs(self.SM.get_simulation_time()-time)>1e-9: raise RuntimeError('MEDIA_CAPTURE_ADVANCED_PHYSICS')
            self.frames.append(frame)
            return front

    # Dependency injection changes capture only; the evaluator owns all policy,
    # clipping, stepping, contact and success logic. One episode prevents child spawning.
    sys.argv=[str(ROOT/'scripts/local/evaluate.py'),'--policy',args.policy,'--split','held_out',
        '--seed-offset',str(seeds.index(args.seed)),'--seed-count','1','--record-all','--output',str(args.output)]
    if args.checkpoint: sys.argv+=['--checkpoint',str(args.checkpoint)]
    def save_capture(summary, rows):
        row=rows[0];env=instances[0]
        if summary['execution_status']!='PASS' or len(env.frames)!=row['steps']+1:
            raise RuntimeError('CAPTURE_EPISODE_INCOMPLETE')
        destination.mkdir(parents=True)
        video=destination/f'{args.view.lower()}.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
            '-r','15','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)],
            input=b''.join(f.tobytes() for f in env.frames),check=True)
        from PIL import Image
        for name,i in [('start',0),('middle',len(env.frames)//2),('end',len(env.frames)-1)]:
            Image.fromarray(env.frames[i]).save(destination/f'{name}.png')
        report={'source_type':'PROJECT_SIM_LEARNED' if args.policy=='act' else 'PROJECT_SIM_EXPERT',
            'evaluation':args.output.resolve().relative_to(ROOT).as_posix(),'seed':args.seed,'outcome':row['outcome'],
            'checkpoint_sha256':summary['checkpoint_files_sha256'],'environment_config_hash':summary['environment_config_hash'],
            'asset':summary['asset'],'camera':env.camera,'frame_range':[0,len(env.frames)],'fps':15,
            'sampling':'one pre-action frame per control step plus final terminal frame; unaltered speed',
            'physics_and_state_invariance_checked_every_capture':True,'policy_camera_unchanged':True,
            'video':video.relative_to(ROOT).as_posix(),'video_sha256':sha(video),
            'source_sha256':sha(Path(__file__)),'camera_config_sha256':sha(ROOT/'media/cameras.json'),
            'primary_evaluation_replacement':False,'native_desktop_verification':'NOT_COMPLETED'}
        folder=ROOT/'artifacts/media';folder.mkdir(exist_ok=True)
        (folder/f'{args.output.name}.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
    # Kit closes the Python process; persist media before SimulationApp.close().
    evaluate.main(env_factory=CaptureEnv, on_complete=save_capture)


if __name__=='__main__': main()
