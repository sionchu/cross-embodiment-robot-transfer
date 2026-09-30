"""4:5 LinkedIn cut (1080x1350, 30 fps) built from the same verified evidence as media/build.py.

Reuses build.Renderer for hash-checked assets; adds CC BY 4.0 GR00T demo_data clips
(DROID / BridgeData V2 / Open X-Embodiment Fractal) as labeled ecosystem context only.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import build
from build import ROOT, Video, text, panel, paste_fit, sha, probe, BG, WHITE, MUTED, TEAL, AMBER

W, H, FPS = 1080, 1350, 30
DEMO = ROOT / '.external/Isaac-GR00T/demo_data'
ECOSYSTEM = [  # (label, embodiment, file, license)
    ('DROID', 'Franka · real', 'droid_sample/videos/chunk-000/observation.images.exterior_1_left/episode_000000.mp4', 'CC BY 4.0'),
    ('BridgeData V2', 'WidowX · real', 'simplerenv_bridge_sample/videos/chunk-000/observation.images.image_0/episode_000001.mp4', 'CC BY 4.0'),
    ('Fractal (RT-1)', 'Google Robot · real', 'simplerenv_fractal_sample/videos/chunk-000/observation.images.image/episode_000000.mp4', 'CC BY 4.0'),
]
CREDIT = 'Context clips: DROID, BridgeData V2, Open X-Embodiment/RT-1 (CC BY 4.0) via NVIDIA Isaac-GR00T demo_data; resized. FIRe (Apache-2.0).'
SHOTS = [('hook', 4), ('ecosystem', 7), ('bridge', 6), ('three', 10), ('wall', 6), ('results', 6), ('close', 5)]


class LinkedIn:
    def __init__(self, manifest_path):
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.r = build.Renderer(m)
        self.eco = []
        for label, body, rel, lic in ECOSYSTEM:
            path = DEMO / rel
            self.eco.append((label, body, lic, Video(path), {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}))
        seed = m['selection']['diagnosis']['seed']
        self.three = []
        for key, label in [('act-50-heldout-001', 'TARGET-ONLY ACT'), ('act-fire-50-heldout-001', 'FIRe FULL TRANSFER'),
                           ('act-firebb-50-heldout-001', 'FIRe BACKBONE-ONLY')]:
            row = next(json.loads(l) for l in (ROOT / 'artifacts/eval' / key / 'episodes.jsonl').read_text().splitlines()
                       if json.loads(l)['seed'] == seed)
            if sha(ROOT / row['video']['path']) != row['video']['sha256']: raise ValueError('CLIP_HASH_MISMATCH')
            self.three.append((label, Video(ROOT / row['video']['path']), row))
        self.seed = seed
        self.study = {(x['target_demos'], x['track']): x for x in self.r.study['records']}

    def base(self, title, sub=None):
        c = Image.new('RGB', (W, H), BG)
        text(c, (56, 40), 'GR00T N1.7 × UR10e', 30, TEAL, True)
        text(c, (56, 86), title, 54, WHITE, True, max_width=W - 112)
        if sub: text(c, (58, 156), sub, 28, MUTED, max_width=W - 116)
        text(c, (56, H - 58), 'ACT trained + evaluated · GR00T N1.7 NEW_EMBODIMENT integration-ready (model run deferred)', 21, MUTED, max_width=W - 112)
        return c

    def tile(self, c, img, box, label, sub, color=MUTED):
        x, y, w, h = box
        panel(c, (x, y, x + w, y + h)); paste_fit(c, img, (x + 8, y + 8, w - 16, h - 78))
        text(c, (x + 16, y + h - 66), label, 26, WHITE, True, max_width=w - 32)
        text(c, (x + 16, y + h - 34), sub, 21, color, max_width=w - 32)

    def frame(self, mode, t):
        r = self.r
        if mode == 'hook':
            c = self.base('Real robot experience → a new robot', 'Can real Franka data help a UR10e learn precision insertion?')
            paste_fit(c, r.videos['real_hook'].at(t), (140, 230, 800, 760))
            panel(c, (140, 1020, 940, 1100)); text(c, (170, 1034), 'REAL FRANKA · FIRe public data', 34, TEAL, True)
            text(c, (140, 1140), '→ UR10e · Isaac Sim · closed-loop evaluation', 30, WHITE, True, max_width=800)
        elif mode == 'ecosystem':
            c = self.base('Many robots, one data contract', 'RGB · state · relative EEF action · modality.json')
            for i, (label, body, lic, video, _) in enumerate(self.eco):
                self.tile(c, video.at(t), (40 + i * 340, 215, 320, 330), label, body)
            self.tile(c, r.videos['real_wrist'].at(t), (40, 565, 490, 400), 'FIRe · SOURCE', 'Franka · real · Apache-2.0')
            x, y = 550, 565
            ImageDraw.Draw(c).rounded_rectangle((x - 6, y - 6, x + 496, y + 406), radius=22, outline=TEAL, width=5)
            self.tile(c, r.videos['hero'].at(t), (x, y, 490, 400), 'UR10e · NEW_EMBODIMENT', 'Isaac Sim · this project', TEAL)
            text(c, (56, 1000), 'Top row: GR00T N1.7 demo data, shown for context.', 26, MUTED, max_width=W - 112)
            text(c, (56, 1040), 'Not trained or run in this project.', 26, AMBER, True, max_width=W - 112)
            text(c, (56, 1090), 'My UR10e dataset passes the official GR00T loader.', 30, WHITE, True, max_width=W - 112)
            text(c, (56, H - 100), CREDIT, 19, MUTED, max_width=W - 112)
        elif mode == 'bridge':
            c = self.base('Transfer motion, not joints', 'Franka pose → relative EEF motion → UR10e')
            paste_fit(c, r.videos['real_wrist'].at(t), (40, 200, 470, 420))
            paste_fit(c, r.videos['hero'].at(t), (570, 200, 470, 420))
            text(c, (60, 630), 'REAL FRANKA', 28, TEAL, True); text(c, (590, 630), 'UR10e · ISAAC SIM', 28, TEAL, True)
            canvas = Image.new('RGB', (1920, 1080), BG); r.draw_trajectory(canvas, (40, 40, 1000, 640), t / 6)
            paste_fit(c, canvas.crop((40, 40, 1040, 680)), (40, 690, W - 80, 470))
            text(c, (80, 1180), 'Δx  Δy  Δz   ΔRx  ΔRy  ΔRz', 44, WHITE, True, max_width=W - 160)
        elif mode == 'three':
            c = self.base('Same robot · same seed · 3 policies', f'UR10e · 50 target demos · held-out seed {self.seed} · real-time 15 Hz')
            for i, (label, video, row) in enumerate(self.three):
                y = 205 + i * 350; done = t >= row['steps'] / 15
                panel(c, (40, y, W - 40, y + 330)); paste_fit(c, video.at(t), (50, y + 10, 420, 310))
                text(c, (500, y + 40), label, 34, WHITE, True, max_width=540)
                if done:
                    ok = row['outcome'] == 'SUCCESS'
                    text(c, (500, y + 120), row['outcome'], 52, TEAL if ok else AMBER, True)
                    text(c, (500, y + 200), f'final lateral error {row["lateral_error_m"] * 1000:.2f} mm', 28, WHITE, max_width=540)
                    text(c, (500, y + 245), 'tolerance 0.70 mm', 24, MUTED)
                else:
                    text(c, (500, y + 120), 'running…', 40, MUTED, True)
        elif mode == 'wall':
            c = self.base('Target-only ACT · 20/20 held-out', 'First 8 fixed seeds · 50 target demos · closed loop')
            cfg = r.primary['environment_config']
            for i, (row, (video, trace, end)) in enumerate(zip(r.wall_rows, r.wall, strict=True)):
                x = 40 + (i % 2) * 510; y = 205 + (i // 2) * 265
                step = min(int(t * 15), len(trace) - 1); terminal = t >= row['steps'] / 15
                s = np.array(trace[step]['state'])
                aligned = np.linalg.norm(s[:2]) <= cfg['lateral_tolerance_m'] and np.linalg.norm(s[3:]) <= cfg['angular_tolerance_rad']
                phase = ('SUCCESS' if row['outcome'] == 'SUCCESS' else 'FAIL') if terminal else 'APPROACH' if s[2] < -.035 else 'ALIGN' if not aligned else 'APPROACH' if s[2] < 0 else 'INSERT'
                panel(c, (x, y, x + 490, y + 250)); paste_fit(c, end if terminal else video.at(t), (x + 8, y + 8, 300, 234))
                text(c, (x + 325, y + 60), f'seed {row["seed"]}', 26, WHITE, True)
                text(c, (x + 325, y + 120), phase, 30, TEAL if phase == 'SUCCESS' else AMBER if phase == 'FAIL' else MUTED, True)
        elif mode == 'results':
            c = self.base('Result at 50 UR10e demos', 'Fixed 20 held-out seeds · one training seed')
            rows = [('Target-only ACT', 'target-only', TEAL), ('FIRe full transfer', 'source-informed', AMBER), ('FIRe backbone-only', 'source-backbone', AMBER)]
            for i, (label, key, color) in enumerate(rows):
                rec = self.study[50, key]; y = 215 + i * 230
                panel(c, (40, y, W - 40, y + 205))
                text(c, (75, y + 30), label, 34, WHITE, True)
                mm = rec['mean_lateral_error_m'] * 1000
                text(c, (75, y + 95), f'mean lateral error {mm:.0f} mm' if mm >= 10 else f'mean lateral error {mm:.2f} mm', 30, MUTED)
                text(c, (700, y + 45), f'{rec["successes"]}/{rec["episodes"]}', 90, color, True)
            text(c, (56, 930), 'Full transfer: the transformer stopped', 32, WHITE, True, max_width=W - 112)
            text(c, (56, 975), 'responding to UR10e input (root-caused).', 32, WHITE, True, max_width=W - 112)
            text(c, (56, 1040), 'Backbone-only learned, but missed the', 32, WHITE, True, max_width=W - 112)
            text(c, (56, 1085), '1 mm clearance. No source-data gain here.', 32, AMBER, True, max_width=W - 112)
        else:  # close
            c = self.base('Measured, not assumed', 'Real data → new embodiment → closed-loop evidence')
            paste_fit(c, r.source_full(7.5 + t), (40, 210, W - 80, 420))
            paste_fit(c, r.videos['close'].at(8 + t), (40, 650, W - 80, 420))
            text(c, (60, 1090), 'GR00T N1.7 · NEW_EMBODIMENT contract validated', 30, TEAL, True, max_width=W - 120)
            text(c, (60, 1135), 'Next: GR00T fine-tuning on 40 GB+ compute', 28, WHITE, max_width=W - 120)
            text(c, (56, H - 100), CREDIT, 19, MUTED, max_width=W - 112)
        return c

    def previews(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        for mode, d in SHOTS: self.frame(mode, d * .75).save(folder / f'{mode}.png')

    def render(self, output):
        if output.exists(): raise FileExistsError(output)
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        for mode, d in SHOTS:
            for f in range(d * FPS): p.stdin.write(self.frame(mode, f / FPS).tobytes())
        p.stdin.close()
        if p.wait() != 0: raise RuntimeError('ENCODER_FAILED')
        info = probe(output)
        if (int(info['nb_read_frames']), info['r_frame_rate'], info['width'], info['height']) != (sum(d for _, d in SHOTS) * FPS, '30/1', W, H):
            raise ValueError('FORMAT_MISMATCH')
        receipt = {'output': output.relative_to(ROOT).as_posix(), 'sha256': sha(output), 'probe': info,
            'script_sha256': sha(Path(__file__)), 'shots': SHOTS, 'ecosystem_context': [e[4] | {'label': e[0], 'license': e[2]} for e in self.eco],
            'three_policy_seed': self.seed, 'credit': CREDIT, 'native_desktop_review': 'NOT_RUN'}
        output.with_suffix('.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(receipt['probe']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=ROOT / 'media/edit_manifest_v2.json')
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args(); job = LinkedIn(a.manifest.resolve())
    if a.preview: job.previews(ROOT / 'media/output/review_linkedin')
    else: job.render(ROOT / 'media/output/linkedin_4x5.mp4')
