# Isolated Windows data/ACT environment. Does not change Isaac Sim or system PATH.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$previousLfsSmudge = $env:GIT_LFS_SKIP_SMUDGE
Push-Location $projectRoot
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        uv venv --python 3.12 .venv
        if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
    }
    uv pip sync --python .venv/Scripts/python.exe scripts/local/requirements.lock.txt --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match --link-mode copy
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed' }
    $pins = Get-Content references/source_manifest.json -Raw | ConvertFrom-Json
    New-Item -ItemType Directory -Force .external | Out-Null
    $archive = '.external/ffmpeg-7.1.1-shared.7z'
    if (-not (Test-Path $archive)) {
        Invoke-WebRequest -Uri $pins.local.ffmpeg_reader.url -OutFile $archive
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pins.local.ffmpeg_reader.sha256) {
        throw 'FFmpeg archive hash mismatch; do not use this archive'
    }
    $decoderBin = '.external/ffmpeg-7.1.1/ffmpeg-7.1.1-full_build-shared/bin'
    if (-not (Test-Path "$decoderBin/avcodec-61.dll")) {
        $sevenZip = 'C:/Program Files/7-Zip/7z.exe'
        if (-not (Test-Path $sevenZip)) { throw 'Install 7-Zip or extract the verified archive into .external/ffmpeg-7.1.1' }
        & $sevenZip x $archive '-o.external/ffmpeg-7.1.1' -y
        if ($LASTEXITCODE -ne 0) { throw 'FFmpeg extraction failed' }
    }
    if (-not (Test-Path '.external/Isaac-GR00T/.git')) {
        $env:GIT_LFS_SKIP_SMUDGE = '1'
        git clone --filter=blob:none --no-checkout $pins.upstream.url .external/Isaac-GR00T
        if ($LASTEXITCODE -ne 0) { throw 'GR00T clone failed' }
        git -C .external/Isaac-GR00T checkout --detach $pins.upstream.commit
        if ($LASTEXITCODE -ne 0) { throw 'GR00T pinned checkout failed' }
    }
    $actual = git -C .external/Isaac-GR00T rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $actual -ne $pins.upstream.commit) { throw 'Existing GR00T checkout has a different revision' }
    .venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }
} finally {
    $env:GIT_LFS_SKIP_SMUDGE = $previousLfsSmudge
    Pop-Location
}
