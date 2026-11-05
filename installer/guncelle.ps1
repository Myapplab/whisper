# STT - tek satirla guncelle (yoksa kur). PowerShell'e yapistirin:
#   irm https://raw.githubusercontent.com/Myapplab/whisper/main/installer/guncelle.ps1 | iex
# Dosya diske inmeden bellekte calisir: Smart App Control / "Engellemeyi kaldir" gerekmez.
# Not: bu dosya bilerek yalnizca ASCII (irm | iex kod sayfasi farklarindan etkilenmesin).

function Install-STT {
  $ErrorActionPreference = 'Stop'
  $ProgressPreference = 'SilentlyContinue'
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $repo = "Myapplab/whisper"
  $zipUrl = "https://codeload.github.com/$repo/zip/refs/heads/main"

  function Say($m, $c = "Gray") { Write-Host $m -ForegroundColor $c }

  # 1) kurulum klasoru: kayit defteri -> C:\stt -> LocalAppData\Programs\STT
  $dir = $null
  $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\STT"
  if (Test-Path $key) { $dir = (Get-ItemProperty $key).InstallLocation }
  if (-not $dir -or -not (Test-Path (Join-Path $dir ".venv\Scripts\python.exe"))) {
    $dir = $null
    foreach ($c in @("C:\stt", (Join-Path $env:LOCALAPPDATA "Programs\STT"))) {
      if (Test-Path (Join-Path $c ".venv\Scripts\python.exe")) { $dir = $c; break }
    }
  }
  $fresh = -not $dir
  if ($fresh) { $dir = Join-Path $env:LOCALAPPDATA "Programs\STT"; Say "STT kurulu degil: $dir klasorune kurulacak (ilk kurulum 10-30 dk)." "Yellow" }
  else { Say "STT bulundu: $dir" "Green" }

  # 2) acik STT'yi kapat (uygulama + GPU iscileri)
  $venvDir = (Join-Path $dir ".venv").ToLower()
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath.ToLower().StartsWith($venvDir) } |
    ForEach-Object { Say ("Kapatiliyor: " + $_.Name + " (" + $_.ProcessId + ")"); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2

  # 3) en son surumu indir
  Say "En son surum GitHub'dan indiriliyor..."
  $tmp = Join-Path $env:TEMP ("stt-guncelle-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
  New-Item -ItemType Directory -Force $tmp | Out-Null
  $zip = Join-Path $tmp "stt.zip"
  Invoke-WebRequest $zipUrl -OutFile $zip -UseBasicParsing
  Expand-Archive $zip -DestinationPath $tmp -Force
  $root = Get-ChildItem $tmp -Directory | Where-Object { Test-Path (Join-Path $_.FullName "stt\__init__.py") } | Select-Object -First 1
  if (-not $root) { throw "Indirilen pakette STT dosyalari yok." }
  $root = $root.FullName

  if ($fresh) {
    # 4a) ilk kurulum: dosyalari kopyala, kur.ps1 ile Python/PyTorch/model/araclari kur
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($n in @("stt", "pyproject.toml", "constraints.txt", "requirements.txt", "README.md", "BENI_OKU.txt", "stt.toml.example")) {
      $s = Join-Path $root $n
      if (Test-Path $s) { Copy-Item $s $dir -Recurse -Force }
    }
    Copy-Item (Join-Path $root "installer\kur.ps1") (Join-Path $dir "kur.ps1") -Force
    $data = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "STT Veri Seti"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir "kur.ps1") -InstDir $dir -DataDir $data -Step all
    if ($LASTEXITCODE -ne 0) { throw "Kurulum tamamlanamadi. Ayrintilar: $dir\kurulum.log (komutu tekrar calistirin; biten adimlar atlanir)" }
  } else {
    # 4b) guncelleme: uygulamanin kendi guncelleyicisi (once dener, hata olursa eski hali geri koyar)
    $py = Join-Path $dir ".venv\Scripts\python.exe"
    $code = "import sys,pathlib; sys.path.insert(0, sys.argv[1]); from stt import updater; " +
            "r = updater.apply(lambda s, t: print(t, flush=True), tool_dir=pathlib.Path(sys.argv[2]), data=pathlib.Path(sys.argv[3]).read_bytes()); " +
            "print(r); sys.exit(0 if r.get('ok') else 1)"
    $env:PYTHONIOENCODING = "utf-8"
    & $py -c $code $root $dir $zip
    if ($LASTEXITCODE -ne 0) { throw "Guncelleme yapilamadi (eski surum yerinde duruyor)." }
  }

  # 5) kisayollar (masaustu + Baslat menusu) ve uygulamayi ac
  $pyw = Join-Path $dir ".venv\Scripts\pythonw.exe"
  $ico = Join-Path $dir "stt\assets\stt.ico"
  $sh = New-Object -ComObject WScript.Shell
  foreach ($d in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    $l = $sh.CreateShortcut((Join-Path $d "STT.lnk"))
    $l.TargetPath = $pyw; $l.Arguments = "-m stt.gui"; $l.WorkingDirectory = $dir; $l.IconLocation = $ico
    $l.Description = "STT - Basin Odasi"; $l.Save()
  }
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Say "Tamam. STT aciliyor... (bundan sonra guncellemeler uygulamanin icinden yapilir)" "Green"
  Start-Process $pyw -ArgumentList "-m stt.gui" -WorkingDirectory $dir
}

try { Install-STT } catch { Write-Host ("HATA: " + $_.Exception.Message) -ForegroundColor Red }
