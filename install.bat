@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "CUDA_VISIBLE_DEVICES="
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
echo ============================================
echo   stt kurulumu - bu pencereyi kapatmayin
echo   (ilk kurulum 10-20 dakika surebilir)
echo ============================================
echo.

rem ---- Python 3.11 bul, yoksa kur
set "PYEXE="
for /f "delims=" %%i in ('py -3.11 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE (
  echo [1/6] Python 3.11 kuruluyor...
  winget install -e --id Python.Python.3.11 --scope user --silent --accept-source-agreements --accept-package-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
if not defined PYEXE (
  echo [HATA] Python 3.11 kurulamadi. https://www.python.org/downloads/release/python-3119/ adresinden kurup tekrar deneyin.
  pause & exit /b 1
)
echo [1/6] Python: %PYEXE%

if not exist ".venv\Scripts\python.exe" (
  echo [2/6] Sanal ortam olusturuluyor
  "%PYEXE%" -m venv .venv || (pause & exit /b 1)
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install -q --upgrade pip setuptools wheel || (pause & exit /b 1)

echo [3/6] PyTorch CUDA 12.8 indiriliyor (~3 GB)
"%PY%" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || (pause & exit /b 1)

echo [4/6] stt ve bagimliliklari
"%PY%" -m pip install -c constraints.txt -e ".[diarize,test]" || (pause & exit /b 1)

echo [5/6] ffmpeg ve deno
where ffmpeg >nul 2>&1 || winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
where deno >nul 2>&1 || winget install -e --id DenoLand.Deno --accept-source-agreements --accept-package-agreements

rem yeni kurulan ffmpeg/deno bu pencerede de gorunsun
for /d %%d in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do for /d %%e in ("%%d\ffmpeg-*") do set "PATH=%%e\bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if exist "%USERPROFILE%\.deno\bin" set "PATH=%USERPROFILE%\.deno\bin;%PATH%"

rem masaustu ve Baslat menusu kisayolu: STT uygulamasi (konsolsuz)
powershell -NoProfile -Command "foreach($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))){ $s=(New-Object -ComObject WScript.Shell).CreateShortcut($d+'\STT.lnk'); $s.TargetPath='%~dp0.venv\Scripts\pythonw.exe'; $s.Arguments='-m stt.gui'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0stt\assets\stt.ico'; $s.Description='STT - Basin Odasi'; $s.Save() }" >nul 2>&1

if not exist "stt.toml" copy /y "stt.toml.example" "stt.toml" >nul

echo [6/6] Whisper large-v3 modeli indiriliyor (~3 GB) ve kontrol
"%PY%" -c "from faster_whisper.utils import download_model; print(download_model('large-v3'))"
"%PY%" -m stt kontrol
echo.
echo ============================================
echo   KURULUM BITTI.
echo   Kullanmak icin: masaustundeki STT simgesine cift tiklayin.
echo   (ffmpeg yeni kurulduysa once bilgisayari yeniden baslatin
echo    ya da bu pencereyi kapatip stt.bat'i acin.)
echo ============================================
pause
endlocal
