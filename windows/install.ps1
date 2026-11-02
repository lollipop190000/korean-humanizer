param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'KoreanHumanizer'),
    [switch]$ToolsOnly,
    [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
$sourceRoot = Split-Path $PSScriptRoot -Parent
$appRoot = Join-Path $InstallRoot 'app'
$toolsRoot = Join-Path $InstallRoot 'tools'
New-Item -ItemType Directory -Force -Path $appRoot, $toolsRoot | Out-Null
$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($InstallRoot).Substring(0,1))
if (-not $ToolsOnly -and $drive.Free -lt 25GB) { throw 'Please free at least 25 GB on the installation drive.' }
Write-Host 'Installing Korean Humanizer. No administrator privileges are needed.'
if ([IO.Path]::GetFullPath($sourceRoot).TrimEnd('\') -ne [IO.Path]::GetFullPath($appRoot).TrimEnd('\')) {
    $excluded = @('.git', '.venv', '.venv-inference', 'artifacts', 'node_modules', 'dist', '__pycache__', '.pytest_cache', '.ruff_cache')
    Get-ChildItem -LiteralPath $sourceRoot -Force | Where-Object { $_.Name -notin $excluded } | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $appRoot -Recurse -Force
    }
}
$uv = Join-Path $toolsRoot 'uv.exe'
if (-not (Test-Path $uv)) {
    Write-Host 'Downloading verified Python installer (uv 0.12.23)...'
    $zip = Join-Path $toolsRoot 'uv.zip'
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip
    $digest = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    if ($digest -ne '75d05de6762778c31ee183398de7dd15093fad0ed90b1f236d8205ea5ec00c90') { throw 'Installer checksum mismatch.' }
    Expand-Archive -LiteralPath $zip -DestinationPath $toolsRoot -Force
    Remove-Item -LiteralPath $zip
}
function Invoke-Uv {
    & $uv @args
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE)." }
}
Push-Location $appRoot
try {
    $env:PYTHONUTF8 = '1'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsRoot 'python'
    Invoke-Uv python install 3.11
    Invoke-Uv venv --python 3.11 --allow-existing .venv-inference
    $python = Join-Path $appRoot '.venv-inference\Scripts\python.exe'
    if ($ToolsOnly) {
        Write-Host 'Tools-only validation mode: inference packages and model weights are not installed.'
        Invoke-Uv pip install --python $python -e .
    } else {
        $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($nvidia) {
            Write-Host 'NVIDIA detected. Installing CUDA inference support...'
            Invoke-Uv pip install --python $python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
            Invoke-Uv pip install --python $python bitsandbytes==0.45.5
        }
        Invoke-Uv pip install --python $python -e . -r requirements/inference.txt
        & $python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('The trained model will be downloaded on first rewrite.')"
        if ($LASTEXITCODE -ne 0) { throw 'Inference environment check failed.' }
        if ($nvidia) {
            & $python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
            if ($LASTEXITCODE -ne 0) { throw 'NVIDIA GPU found but CUDA is unavailable. Update the NVIDIA driver and run Install.cmd again.' }
        }
    }
    $desktop = [Environment]::GetFolderPath('Desktop')
    if ($desktop -and -not $ToolsOnly) {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path $desktop 'Korean Humanizer.lnk'))
        $shortcut.TargetPath = Join-Path $appRoot 'Start.cmd'
        $shortcut.WorkingDirectory = $appRoot
        $shortcut.Description = 'Korean Humanizer - local trained model'
        $shortcut.Save()
    }
    Write-Host "Installed at $appRoot"
    Write-Host 'Next time, open the Korean Humanizer desktop shortcut.'
    if (-not $NoLaunch -and -not $ToolsOnly) {
        Start-Process -FilePath (Join-Path $appRoot 'Start.cmd') -WorkingDirectory $appRoot
    }
} finally { Pop-Location }
