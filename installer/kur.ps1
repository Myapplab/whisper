# STT kurulum adimlari (setup.exe bu betigi calistirir). Tekrar calistirilirsa eksikleri tamamlar.
param([Parameter(Mandatory=$true)][string]$InstDir, [string]$DataDir = "")
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location $InstDir
function Step($m) { Write-Output ""; Write-Output "==> $m" }
function Run($exe, [string[]]$a, $what) {
  # PowerShell 5.1: yerel programin stderr satirlari (pip uyarilari) hata sayilmasin
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $exe @a 2>&1 | ForEach-Object { "$_" }; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
  if ($code -ne 0) { throw "$what basarisiz (kod $code)" }
}
function Quiet([scriptblock]$b) {
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $b } finally { $ErrorActionPreference = $old }
}

# --- 1. Python 3.11
Step "Python 3.11 araniyor"
$py = $null
try { $py = Quiet { (& py -3.11 -c "import sys;print(sys.executable)" 2>$null) } } catch {}
if (-not $py -or -not (Test-Path $py)) {
  $cand = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
  if (Test-Path $cand) { $py = $cand }
}
if (-not $py -or -not (Test-Path $py)) {
  Step "Python 3.11 indiriliyor (python.org, ~25 MB)"
  $inst = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
  Invoke-WebRequest "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $inst -UseBasicParsing
  $p = Start-Process -FilePath $inst -ArgumentList "/quiet InstallAllUsers=0 Include_launcher=1 PrependPath=0 Include_test=0 Shortcuts=0" -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "Python kurulumu basarisiz (kod $($p.ExitCode))" }
  $py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
  if (-not (Test-Path $py)) { throw "Python 3.11 kurulamadi" }
}
Write-Output "Python: $py"

# --- 2. Sanal ortam + kutuphaneler
$venv = Join-Path $InstDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venv)) {
  Step "Sanal ortam olusturuluyor"
  Run $py @("-m", "venv", (Join-Path $InstDir ".venv")) "Sanal ortam"
}
Run $venv @("-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel") "pip guncelleme"
Step "PyTorch + CUDA 12.8 (ilk kurulumda ~3 GB, birkac dakika surer)"
Run $venv @("-m", "pip", "install", "--progress-bar", "off", "torch==2.8.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128") "PyTorch kurulumu"
Step "Uygulama kutuphaneleri (faster-whisper, yt-dlp, pyannote, arayuz)"
Run $venv @("-m", "pip", "install", "--progress-bar", "off", "-c", "constraints.txt", "-e", ".[diarize]") "Kutuphane kurulumu"

# --- 3. ffmpeg ve deno (once winget, olmazsa dogrudan indirme)
$tools = Join-Path $InstDir "tools"
function Have($n) { return [bool](Get-Command $n -ErrorAction SilentlyContinue) -or (Test-Path (Join-Path $tools "$n*")) }
if (-not (Have "ffmpeg")) {
  Step "ffmpeg kuruluyor"
  $ok = $false
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try { Run "winget" @("install", "-e", "--id", "Gyan.FFmpeg", "--silent", "--accept-source-agreements", "--accept-package-agreements") "winget ffmpeg"; $ok = $true } catch { Write-Output "$_" }
  }
  if (-not $ok) {
    New-Item -ItemType Directory -Force $tools | Out-Null
    $z = Join-Path $env:TEMP "ffmpeg.zip"
    Invoke-WebRequest "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $z -UseBasicParsing
    Expand-Archive $z -DestinationPath $tools -Force; Remove-Item $z
  }
}
if (-not (Have "deno")) {
  Step "deno kuruluyor (YouTube icin)"
  $ok = $false
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    try { Run "winget" @("install", "-e", "--id", "DenoLand.Deno", "--silent", "--accept-source-agreements", "--accept-package-agreements") "winget deno"; $ok = $true } catch { Write-Output "$_" }
  }
  if (-not $ok) {
    New-Item -ItemType Directory -Force (Join-Path $tools "deno") | Out-Null
    $z = Join-Path $env:TEMP "deno.zip"
    Invoke-WebRequest "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip" -OutFile $z -UseBasicParsing
    Expand-Archive $z -DestinationPath (Join-Path $tools "deno") -Force; Remove-Item $z
  }
}

# --- 4. Ayar dosyasi (veri seti klasoru)
$toml = Join-Path $InstDir "stt.toml"
if (-not (Test-Path $toml)) {
  Copy-Item (Join-Path $InstDir "stt.toml.example") $toml
  if ($DataDir) {
    $d = $DataDir -replace '\\', '/'
    (Get-Content $toml -Encoding UTF8) -replace '^out_dir = .*', "out_dir = '$d'" | Set-Content $toml -Encoding utf8
  }
}

# --- 5. Whisper large-v3 modeli (~3 GB, bir kez)
Step "Whisper large-v3 modeli indiriliyor (~3 GB, bir kez)"
Run $venv @("-c", "from faster_whisper.utils import download_model; print(download_model('large-v3'))") "Model indirme"

Step "Kurulum tamamlandi"
