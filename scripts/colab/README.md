# Colab / Cloud GPU adapter: official N1.7 E0

Status: workflow prepared; official inference has not yet completed. E0 requires
Linux x86_64 with a measured NVIDIA CUDA GPU of at least 16 GB-class VRAM; a free
T4 is acceptable if it passes the runner's VRAM gate. Fine-tuning later requires
40 GB+ VRAM. Actual hardware is recorded, never assumed. The owner must approve
Cosmos model access beforehand.
This workflow downloads upstream dependencies and three model snapshots; inspect
available disk/RAM first. It does not start training or publish anything.

Create a portable copy from a clean committed project checkout:

```text
git bundle create project.bundle HEAD
```

Upload that bundle when requested below. A bundle preserves the project commit
without requiring a public push. Run each fenced code block as a separate Colab
cell, in order. Stop on any failed cell; do not skip checks to mark E0 complete.

### Optional: official Colab CLI operator

Google's official `googlecolab/google-colab-cli` can replace most browser lifecycle
work. On Linux/macOS (or Windows via WSL2):

```text
uv tool install google-colab-cli
colab version
colab usage
colab new -s e0 --gpu T4
colab status -s e0
colab stop -s e0
```

The first account command starts Google's OAuth flow. `colab usage` is the preferred
way to record compute-unit balance. `colab new --gpu A100` requests A100 but does not
bypass subscription, regional, quota or capacity restrictions. For this South Korea
account, Google One support reported the bundled AI Pro Colab CCU benefit as currently
available only in Canada, Mexico and the United States. The CLI is therefore an
automation/evidence tool, not an entitlement workaround.

## 1. Record hardware before installation

```python
import subprocess
from pathlib import Path
print(subprocess.check_output(['nvidia-smi'], text=True))
print(subprocess.check_output(['df', '-h', '/content'], text=True))
print(subprocess.check_output(['free', '-h'], text=True))
preflight = Path('/content/e0-preflight')
preflight.mkdir(exist_ok=False)
(preflight / 'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
(preflight / 'disk.txt').write_text(subprocess.check_output(['df', '-h', '/content'], text=True))
(preflight / 'ram.txt').write_text(subprocess.check_output(['free', '-h'], text=True))
```

Confirm that the GPU has at least 16 GB-class VRAM in the output. If a prior
attempt created these directories, use a fresh runtime; this procedure intentionally
refuses to reuse stale results.

## 2. Upload and check out the committed project

```python
from google.colab import files
uploaded = files.upload()  # Select project.bundle only; never upload credentials.
assert set(uploaded) == {'project.bundle'}
subprocess.run(['git', 'clone', '/content/project.bundle', '/content/transfer'], check=True)
subprocess.run(['git', '-C', '/content/transfer', 'status', '--short'], check=True)
```

## 3. Install the pinned upstream environment

```bash
%%bash
set -euo pipefail
cd /content/transfer
apt-get update -qq
apt-get install -y git-lfs ffmpeg
python -m pip install 'uv==0.12.5'
git lfs install
PIN=$(python -c 'import json; print(json.load(open("references/source_manifest.json"))["upstream"]["commit"])')
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git /content/Isaac-GR00T
cd /content/Isaac-GR00T
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$PIN"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
git lfs pull --include='demo_data/droid_sample/**,scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl' --exclude=''
test "$(git rev-parse HEAD)" = "$PIN"
uv sync --locked --python 3.12 2>&1 | tee /content/e0-preflight/uv-sync.log
.venv/bin/python -c 'import gr00t, torch, torchcodec; print(torch.__version__, torch.version.cuda)'
ffmpeg -version
```

The pin uses cu128 PyTorch 2.9.0 and FFmpeg 4-7. The scoped LFS pull intentionally
includes the aarch64 torchcodec wheel even on x86_64 because uv inspects its local-source
metadata during `uv sync --locked`; leaving it as a Git LFS pointer causes an invalid ZIP
header failure before platform markers can exclude it from installation. If the Colab
driver, system libraries, resolver, or disk cannot support the pinned environment, retain
the failure output and stop. Do not silently change torch/CUDA or regenerate uv.lock.
The pipeline above installs the official default dependency set, including GPU
packages; no custom minimal dependency fork is maintained here.

## 4. Use the owner's Colab secret, then run both checkpoints

When using the browser notebook, add an approved read token as the Colab secret
named `HF_TOKEN` and grant this notebook access. Never paste a token into a cell or
print it. The script checks
a gated config download before downloading full snapshots. The token is passed
only through the child environment; no token login file is written by this flow.

```python
import os
import tempfile
from google.colab import userdata
run_env = os.environ.copy()
try:
    token = userdata.get("HF_TOKEN")
except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
    token = None
if not token:
    raise RuntimeError(
        "HF_TOKEN Colab secret is missing or notebook access is disabled."
    )
run_env["HF_TOKEN"] = token
del token
run_env['HF_HOME'] = tempfile.mkdtemp(prefix='e0-hf-', dir='/content')
run_env['HF_HUB_DISABLE_TELEMETRY'] = '1'
run_env['WANDB_MODE'] = 'disabled'
try:
    subprocess.run([
        '/content/Isaac-GR00T/.venv/bin/python',
        '/content/transfer/scripts/colab/e0.py',
        '--upstream', '/content/Isaac-GR00T',
        '--output', '/content/transfer/artifacts/runtime/e0-gpu',
    ], env=run_env, check=True)
finally:
    run_env.pop('HF_TOKEN', None)
```

The official CLI is run sequentially for base and DROID snapshots, with
trajectories 1/2, seed 42, maximum 200 steps each and execution horizon 8.
CLI output is saved in the run directory; no reference benchmark numbers are
copied into results. The script checks both finite per-episode MSE/MAE values
and each process exit status before writing `status: PASS`.

The official nested backbone loader requests `main`. The runner confirms that
`main` still matches the pinned Cosmos SHA, downloads it into a new isolated
cache, then runs inference offline. A moved backbone branch fails closed and
requires a reviewed pin update. GR00T model snapshots are downloaded by SHA and
passed as local snapshot paths. Upstream code is not patched.

## 5. Preserve evidence, including failed attempts

Run this cell even when the previous cell fails:

```python
import json
import shutil
run_dir = Path('/content/transfer/artifacts/runtime/e0-gpu')
if run_dir.exists():
    shutil.copytree('/content/e0-preflight', run_dir / 'preflight', dirs_exist_ok=True)
    report = run_dir / 'e0.json'
    if report.exists():
        print(json.loads(report.read_text())['status'])
    archive = shutil.make_archive('/content/e0-gpu', 'zip', run_dir)
else:
    archive = shutil.make_archive('/content/e0-preflight', 'zip', '/content/e0-preflight')
files.download(archive)
```

Review `e0.json`, `environment.json`, `nvidia-smi.txt` and both model logs together.
The environment hash ties the result to package versions and submodule commits;
the result also hashes every bundled sample file and records project/model pins.
`PASS` means official open-loop inference executed successfully, not closed-loop
transfer performance. GPU nondeterminism remains possible despite the fixed seed.

Raw logs are ignored by Git. After reviewing for paths/secrets, retain compact
JSON under `artifacts/environment/<run-id>/` and update HANDOFF in a separate
results commit. Keep the raw evidence archive externally. Do not mark E0 complete
from a notebook import, parser unit tests, or a run that failed before inference.
