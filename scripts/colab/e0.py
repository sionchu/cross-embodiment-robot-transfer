"""Run the pinned official E0 CLI; never substitute mocked GPU evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_metrics(log, returncode):
    if returncode != 0:
        raise ValueError('Official inference returned a nonzero exit code')
    rows = re.findall(r'MSE for trajectory (\d+): ([^,\s]+), MAE: ([^\s]+)', log)
    if len(rows) != 2 or {r[0] for r in rows} != {'1', '2'}:
        raise ValueError('Both requested trajectories must have exactly one result')
    results = []
    for episode, mse, mae in rows:
        values = [float(mse), float(mae)]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Nonfinite or negative metric')
        results.append({'episode_id': int(episode), 'mse': values[0], 'mae': values[1]})
    return results


def run(upstream, output):
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous run.
    pins = json.loads((ROOT / 'references/source_manifest.json').read_text())
    report = {
        'status': 'FAILED', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'preflight', 'upstream_commit': pins['upstream']['commit'],
        'dataset': pins['dataset'], 'seed_set': [42], 'episode_ids': [1, 2],
        'steps_max_per_episode': 200, 'execution_horizon': 8,
        'requested_episode_count_per_model': 2, 'evaluation_episode_count': 0, 'runs': [],
    }

    def command(args, cwd=upstream, env=None):
        return subprocess.run(args, cwd=cwd, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def checked(args, cwd=upstream):
        result = command(args, cwd)
        if result.returncode:
            raise RuntimeError(f'Command failed: {args[0]} (exit {result.returncode})')
        return result.stdout.strip()

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    try:
        smi = checked(['nvidia-smi'])
        (output / 'nvidia-smi.txt').write_text(smi + '\n')
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise RuntimeError('E0 requires Linux x86_64')
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError('Use the upstream Python 3.12 environment')
        if checked(['git', 'rev-parse', 'HEAD']) != pins['upstream']['commit']:
            raise RuntimeError('Wrong upstream revision')
        if checked(['git', 'status', '--porcelain', '--untracked-files=no']):
            raise RuntimeError('Upstream tracked files were modified')
        if digest(upstream / 'uv.lock') != pins['upstream']['lock_sha256']:
            raise RuntimeError('Upstream lock hash mismatch')
        submodules = checked(['git', 'submodule', 'status', '--recursive'])
        if any(line.startswith(('-', '+', 'U')) for line in submodules.splitlines()):
            raise RuntimeError('Submodules are missing or not at recorded revisions')
        report['project_commit'] = checked(['git', 'rev-parse', 'HEAD'], ROOT)
        if checked(['git', 'status', '--porcelain'], ROOT):
            raise RuntimeError('Commit the project workflow before recording E0 evidence')
        ffmpeg = checked(['ffmpeg', '-version']).splitlines()[0]
        version = re.search(r'ffmpeg version (\d+)', ffmpeg)
        if not version or int(version[1]) not in pins['environment']['ffmpeg_major']:
            raise RuntimeError('Pinned torchcodec requires FFmpeg 4-7')

        import torch
        import torchcodec  # Exercise the actual video backend's native-library import.
        if torch.__version__.split('+')[0] != pins['environment']['torch']:
            raise RuntimeError('Torch does not match the pinned environment')
        if importlib.metadata.version('torchcodec') != pins['environment']['torchcodec']:
            raise RuntimeError('Torchcodec does not match the pinned environment')
        if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('An actual CUDA A100 is required for this gate')
        if torch.ones(1, device='cuda').sum().item() != 1:
            raise RuntimeError('CUDA compute smoke failed')
        packages = dict(sorted((d.metadata['Name'], d.version)
                              for d in importlib.metadata.distributions()))
        environment = {
            'python': platform.python_version(), 'os': platform.system(),
            'machine': platform.machine(), 'gpu': torch.cuda.get_device_name(0),
            'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch_cuda': torch.version.cuda, 'torch': torch.__version__,
            'packages': packages, 'ffmpeg': ffmpeg,
            'uv': checked(['uv', '--version']), 'git': checked(['git', '--version']),
            'git_lfs': checked(['git', 'lfs', 'version']),
            'submodules': submodules.splitlines(),
            'lock_sha256': digest(upstream / 'uv.lock'),
        }
        save('environment.json', environment)
        report['environment_config_hash'] = digest(output / 'environment.json')
        report['workflow_sha256'] = digest(__file__)
        report['pins_sha256'] = digest(ROOT / 'references/source_manifest.json')

        report['stage'] = 'sample'
        sample = upstream / 'demo_data/droid_sample'
        payload = sorted(p for p in sample.rglob('*') if p.is_file())
        if not payload or not list(sample.rglob('*.parquet')) or not list(sample.rglob('*.mp4')):
            raise RuntimeError('Bundled sample payload missing; perform the scoped LFS pull')
        for path in payload:
            with path.open('rb') as stream:
                if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise RuntimeError('Unresolved LFS pointer in sample')
        report['dataset_files_sha256'] = {
            p.relative_to(sample).as_posix(): digest(p) for p in payload
        }

        report['stage'] = 'gated_access_and_snapshots'
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        backbone = pins['backbone']
        # Downloads a gated file, not just public model metadata.
        hf_hub_download(backbone['id'], 'config.json', revision=backbone['revision'])
        report['gated_config_access'] = 'PASS'
        # Official nested backbone loads use main. Resolve main in an isolated cache,
        # verify it is the pin, then force all inference into offline mode.
        if HfApi().model_info(backbone['id']).sha != backbone['revision']:
            raise RuntimeError('Backbone main moved; review and update pins before running')
        backbone_path = Path(snapshot_download(backbone['id'], revision='main'))
        if backbone_path.name != backbone['revision']:
            raise RuntimeError('Backbone changed during download')
        report['backbone'] = backbone
        snapshots = []
        for model in pins['models']:
            path = Path(snapshot_download(model['id'], revision=model['revision']))
            if path.name != model['revision']:
                raise RuntimeError('Checkpoint revision mismatch')
            snapshots.append((model, path))

        report['stage'] = 'official_inference'
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   MPLBACKEND='Agg', WANDB_MODE='disabled')
        # No authentication is necessary in the offline inference subprocess.
        env.pop('HF_TOKEN', None)
        for model, path in snapshots:
            name = model['id'].split('/')[-1]
            args = [sys.executable, 'scripts/deployment/standalone_inference_script.py',
                    '--model-path', str(path), '--dataset-path', 'demo_data/droid_sample',
                    '--embodiment-tag', 'OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT',
                    '--traj-ids', '1', '2', '--inference-mode', 'pytorch',
                    '--execution-horizon', '8', '--steps', '200', '--seed', '42']
            result = command(args, env=env)
            (output / f'{name}.log').write_text(result.stdout)
            entry = {'model_id': model['id'], 'revision': model['revision'],
                     'exit_code': result.returncode, 'status': 'FAILED'}
            report['runs'].append(entry)
            entry['episodes'] = parse_metrics(result.stdout, result.returncode)
            entry['status'] = 'PASS'
            report['evaluation_episode_count'] += len(entry['episodes'])
            entry['log_sha256'] = digest(output / f'{name}.log')
        report['status'] = 'PASS'
        report['stage'] = 'complete'
    except Exception as exc:
        # Keep error text/tracebacks out of shareable manifests (may contain secrets).
        report['error_type'] = type(exc).__name__
        print(f"E0 FAILED at {report['stage']}: {type(exc).__name__}", file=sys.stderr)
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save('e0.json', report)
    print(f'E0 PASS: {output / "e0.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream.resolve(), args.output.resolve())
