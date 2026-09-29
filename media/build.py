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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-source',action='store_true',required=True)
    args=parser.parse_args()
    prepare_source(json.loads(MANIFEST.read_text(encoding='utf-8')))
