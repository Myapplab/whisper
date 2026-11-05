# STT kurulum adımları. STT-Kurulum.exe her adımı ayrı çağırır (-Step); elle çalıştırılırsa hepsini yapar.
# Tekrar çalıştırılırsa biten adımlar atlanır. Ayrıntılı günlük: <kurulum klasörü>\kurulum.log
# NOT: Bu dosya UTF-8 (BOM'lu) kaydedilmelidir; Windows PowerShell 5.1 BOM'suz dosyada Türkçe harfleri bozar.
param(
  [Parameter(Mandatory = $true)][string]$InstDir,
  [string]$DataDir = "",
  [ValidateSet("all", "check", "python", "venv", "torch", "libs", "tools", "config", "model")][string]$Step = "all"
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
# Kurulum penceresi çıktıyı OEM kod sayfasında okur (nsExec /OEM): Türkçe harfler doğru görünsün
try {
  $oem = [Text.Encoding]::GetEncoding([Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage)
  [Console]::OutputEncoding = $oem
} catch {}
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
New-Item -ItemType Directory -Force $InstDir | Out-Null
Set-Location $InstDir
$LogFile = Join-Path $InstDir "kurulum.log"
$venv = Join-Path $InstDir ".venv\Scripts\python.exe"
$tools = Join-Path $InstDir "tools"

function Say([string]$m) {
  Write-Output $m
  Add-Content -Path $LogFile -Value ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) -Encoding UTF8
}
function LogOnly([string]$m) { Add-Content -Path $LogFile -Value $m -Encoding UTF8 }

# Bir programı çalıştırır: çıktının tamamı kurulum.log'a, yalnızca anlamlı satırlar pencereye.
function Run([string]$exe, [string[]]$a, [string]$what, [switch]$Pip) {
  LogOnly ("> {0} {1}" -f $exe, ($a -join " "))
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # pip uyarıları (stderr) hata sayılmasın
  $last = @()
  try {
    & $exe @a 2>&1 | ForEach-Object {
      $line = "$_"
      LogOnly $line
      $last = @($last + $line) | Select-Object -Last 8
      if ($Pip) {
        if ($line -match '^\s*Downloading\s+(\S+)\s+\(([\d.]+)\s*(MB|GB)\)') {
          $size = [double]$Matches[2]
          if ($Matches[3] -eq 'GB' -or $size -ge 5) {
            $name = ($Matches[1] -split '/')[-1] -replace '-(cp\d+|py\d).*$', ''
            Say ("   indiriliyor: {0} ({1} {2})" -f $name, $Matches[2], $Matches[3])
          }
        } elseif ($line -match '^Installing collected packages: (.*)$') {
          Say ("   kuruluyor: {0} paket" -f (($Matches[1] -split ',').Count))
        } elseif ($line -match '^Successfully installed') {
          Say "   tamam."
        } elseif ($line -match '^\s*(ERROR|error):') {
          Say ("   " + $line.Trim())
        }
      }
    }
    $code = $LASTEXITCODE
  } finally { $ErrorActionPreference = $old }
  if ($code -ne 0) {
    foreach ($l in $last) { if ($l.Trim()) { Say ("   | " + $l.Trim()) } }
    throw "$what başarısız oldu (çıkış kodu $code)."
  }
}

function Download([string]$url, [string]$out, [string]$what) {
  for ($i = 1; $i -le 3; $i++) {
    try {
      Invoke-WebRequest $url -OutFile $out -UseBasicParsing
      return
    } catch {
      LogOnly "indirme hatası ($i/3) $url : $_"
      if ($i -eq 3) { throw "$what indirilemedi. İnternet bağlantısını kontrol edin. ($($_.Exception.Message))" }
      Start-Sleep -Seconds (3 * $i)
    }
  }
}

# ------------------------------------------------------------------ adımlar
function Step-Check {
  Say "Ön kontrol"
  $drive = (Get-Item $InstDir).PSDrive
  $freeGB = [math]::Round($drive.Free / 1GB, 1)
  Say ("   {0}: sürücüsünde boş alan: {1} GB" -f $drive.Name, $freeGB)
  if (-not (Test-Path $venv) -and $freeGB -lt 9) {
    throw ("Yeterli boş alan yok: en az 9 GB gerekiyor, {0} GB var. Yer açıp kurulumu tekrar çalıştırın." -f $freeGB)
  }
  try {
    $null = Invoke-WebRequest "https://pypi.org/simple/pip/" -UseBasicParsing -Method Head -TimeoutSec 20
    Say "   internet bağlantısı: var"
  } catch {
    throw "İnternete bağlanılamadı (pypi.org). Bağlantıyı kontrol edip kurulumu tekrar çalıştırın."
  }
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($smi) {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $gpus = & nvidia-smi --query-gpu=name,memory.total,display_active --format=csv,noheader 2>$null
    $ErrorActionPreference = $old
    foreach ($g in $gpus) { Say ("   ekran kartı: " + ($g -replace 'NVIDIA GeForce ', '')) }
  } else {
    Say "   UYARI: NVIDIA ekran kartı bulunamadı. Uygulama işlemciyle (çok daha yavaş) çalışır."
  }
}

function Find-Python {
  $p = $null
  try {
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $p = (& py -3.11 -c "import sys;print(sys.executable)" 2>$null)
    $ErrorActionPreference = $old
  } catch {}
  if ($p -and (Test-Path $p)) { return $p }
  $cand = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
  if (Test-Path $cand) { return $cand }
  return $null
}

function Step-Python {
  if (Test-Path $venv) { Say "Python: sanal ortam zaten var, atlandı."; return }
  $py = Find-Python
  if (-not $py) {
    Say "Python 3.11 indiriliyor (python.org, ~25 MB)"
    $inst = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
    Download "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" $inst "Python"
    Say "Python 3.11 kuruluyor (yönetici izni gerekmez)"
    $p = Start-Process -FilePath $inst -ArgumentList "/quiet InstallAllUsers=0 Include_launcher=1 PrependPath=0 Include_test=0 Shortcuts=0" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Python kurulumu başarısız oldu (kod $($p.ExitCode))." }
    Remove-Item $inst -ErrorAction SilentlyContinue
    $py = Find-Python
    if (-not $py) { throw "Python 3.11 kurulamadı." }
  }
  Say "   Python: $py"
}

function Step-Venv {
  if (-not (Test-Path $venv)) {
    $py = Find-Python
    if (-not $py) { throw "Python 3.11 bulunamadı; kurulumu tekrar çalıştırın." }
    Say "Sanal ortam oluşturuluyor"
    Run $py @("-m", "venv", (Join-Path $InstDir ".venv")) "Sanal ortam oluşturma"
  }
  Say "pip güncelleniyor"
  Run $venv @("-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel") "pip güncelleme"
  Say "   tamam."
}

function Step-Torch {
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  $have = & $venv -c "import torch;print(torch.__version__)" 2>$null
  $ErrorActionPreference = $old
  if ($have -like "2.8.0+cu128*") { Say "PyTorch $have zaten kurulu, atlandı."; return }
  Say "PyTorch + NVIDIA CUDA indiriliyor (~3 GB). Bu adım birkaç dakika sürer; pencere donmadı, lütfen bekleyin."
  Run $venv @("-m", "pip", "install", "--progress-bar", "off", "torch==2.8.0", "torchaudio==2.8.0",
              "--index-url", "https://download.pytorch.org/whl/cu128") "PyTorch kurulumu" -Pip
}

function Step-Libs {
  Say "Uygulama kütüphaneleri kuruluyor (faster-whisper, yt-dlp, pyannote, arayüz)"
  Run $venv @("-m", "pip", "install", "--progress-bar", "off", "-c", "constraints.txt", "-e", ".[diarize]") "Kütüphane kurulumu" -Pip
  # YouTube sık değişir: yt-dlp her kurulumda en yeni sürüme güncellenir
  Say "yt-dlp güncelleniyor (YouTube değişikliklerine karşı)"
  Run $venv @("-m", "pip", "install", "-q", "--upgrade", "yt-dlp[default]") "yt-dlp güncelleme"
  Say "   tamam."
}

function Have-Tool([string]$n) {
  if (Get-Command $n -ErrorAction SilentlyContinue) { return $true }
  return [bool](Get-ChildItem -Path $tools -Recurse -Filter "$n.exe" -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Step-Tools {
  New-Item -ItemType Directory -Force $tools | Out-Null
  if (Have-Tool "ffmpeg") { Say "ffmpeg: var, atlandı." } else {
    Say "ffmpeg indiriliyor (~100 MB)"
    $z = Join-Path $env:TEMP "stt-ffmpeg.zip"
    try {
      Download "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" $z "ffmpeg"
    } catch {
      Say "   gyan.dev olmadı, GitHub'dan deneniyor"
      Download "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" $z "ffmpeg"
    }
    Expand-Archive $z -DestinationPath $tools -Force
    Remove-Item $z -ErrorAction SilentlyContinue
    if (-not (Have-Tool "ffmpeg")) { throw "ffmpeg kurulamadı." }
    Say "   tamam."
  }
  if (Have-Tool "deno") { Say "deno: var, atlandı." } else {
    Say "deno indiriliyor (YouTube için gerekli, ~40 MB)"
    $d = Join-Path $tools "deno"
    New-Item -ItemType Directory -Force $d | Out-Null
    $z = Join-Path $env:TEMP "stt-deno.zip"
    Download "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip" $z "deno"
    Expand-Archive $z -DestinationPath $d -Force
    Remove-Item $z -ErrorAction SilentlyContinue
    Say "   tamam."
  }
  # indirilen araçlar "internetten geldi" işaretli olmasın (Smart App Control)
  Get-ChildItem -Recurse $tools -Include *.exe, *.dll -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
}

function Step-Config {
  $toml = Join-Path $InstDir "stt.toml"
  if (Test-Path $toml) { Say "Ayar dosyası zaten var (stt.toml), korunuyor."; return }
  Copy-Item (Join-Path $InstDir "stt.toml.example") $toml
  if ($DataDir) {
    $dd = $DataDir -replace '\\', '/'
    New-Item -ItemType Directory -Force $DataDir | Out-Null
    $txt = (Get-Content $toml -Encoding UTF8) -replace '^out_dir = .*', "out_dir = '$dd'"
    [IO.File]::WriteAllLines($toml, $txt, (New-Object Text.UTF8Encoding $false))  # BOM'suz UTF-8
    Say "Veri seti klasörü: $DataDir"
  }
}

function Step-Model {
  Say "Whisper large-v3 modeli indiriliyor (~3 GB, yalnızca bir kez). Birkaç dakika sürebilir."
  Run $venv @("-c", "from faster_whisper.utils import download_model; print(download_model('large-v3'))") "Model indirme"
  Say "   tamam."
  Say "Son kontrol"
  Run $venv @("-c", "import faster_whisper, ctranslate2, yt_dlp, webview; print('kütüphaneler tamam, GPU sayısı:', ctranslate2.get_cuda_device_count())") "Kütüphane kontrolü"
  # eski elle kurulumdan kalan dosyalar artık gerekmiyor (her şey uygulamada)
  foreach ($f in @("stt.bat", "install.bat", "linkler_pano.txt")) {
    $p = Join-Path $InstDir $f
    if (Test-Path $p) { Remove-Item $p -Force -ErrorAction SilentlyContinue }
  }
}

# ------------------------------------------------------------------ çalıştır
try {
  LogOnly ""
  LogOnly ("===== {0:yyyy-MM-dd HH:mm:ss}  adım: {1}" -f (Get-Date), $Step)
  switch ($Step) {
    "check"  { Step-Check }
    "python" { Step-Python }
    "venv"   { Step-Venv }
    "torch"  { Step-Torch }
    "libs"   { Step-Libs }
    "tools"  { Step-Tools }
    "config" { Step-Config }
    "model"  { Step-Model }
    "all"    { Step-Check; Step-Python; Step-Venv; Step-Torch; Step-Libs; Step-Tools; Step-Config; Step-Model; Say "Kurulum tamamlandı." }
  }
  exit 0
} catch {
  Say ""
  Say ("HATA: " + $_.Exception.Message)
  Say ("Ayrıntılar: " + $LogFile)
  exit 1
}
