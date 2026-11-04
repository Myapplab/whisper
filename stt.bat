@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "STT_DIR=%~dp0"
rem iki kartin da gorunmesi icin: baska programlarin koydugu kisitlamayi bu pencerede kaldir
set "CUDA_VISIBLE_DEVICES="
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "PY=%STT_DIR%.venv\Scripts\python.exe"
rem winget ile yeni kurulan ffmpeg/deno PATH'e henuz girmemis olabilir
for /d %%d in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") do for /d %%e in ("%%d\ffmpeg-*") do set "PATH=%%e\bin;%PATH%"
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if exist "%USERPROFILE%\.deno\bin" set "PATH=%USERPROFILE%\.deno\bin;%PATH%"

if not exist "%PY%" if exist "C:\stt\.venv\Scripts\python.exe" if /i not "%STT_DIR%"=="C:\stt\" (
  call "C:\stt\stt.bat" %*
  exit /b %ERRORLEVEL%
)
if not exist "%PY%" (
  echo [stt] Kurulum bulunamadi: %STT_DIR%
  echo   - Zip'in ICINDEN calistirdiysaniz: once zip'i C:\stt'ye cikarin, sonra C:\stt\stt.bat'i acin.
  echo   - Hic kurmadiysaniz: install.bat dosyasina cift tiklayin.
  pause
  exit /b 1
)
if not "%~1"=="" (
  "%PY%" -m stt %*
  exit /b %ERRORLEVEL%
)

rem ---- cift tiklama: soru sorarak calis
cd /d "%STT_DIR%"
:dongu
cls
echo ==================================================
echo   stt - video / ses / link  ^>  metin
echo ==================================================
echo.
echo TEK video: linki yapistirin (sag tik) ya da dosyayi / klasoru buraya surukleyin, Enter.
echo COK link: linklerin hepsini bir yerden KOPYALAYIN (Ctrl+C), sonra burada sadece Enter'a basin.
set "GIRDI="
set "LISTE="
set /p "GIRDI=> "
if defined GIRDI goto tek
set "LISTE=%STT_DIR%linkler_pano.txt"
powershell -NoProfile -Command "Get-Clipboard | Out-File -Encoding utf8 '%LISTE%'" >nul 2>&1
findstr /i /c:"http" "%LISTE%" >nul 2>&1 && goto liste_hazir
rem panoda link yok -> Not Defteri
set "LISTE=%STT_DIR%linkler.txt"
if not exist "%LISTE%" (
  > "%LISTE%" echo # Her satira bir link yapistirin. FPL icin linkin yanina:  11 ARS
)
start "" notepad "%LISTE%"
echo.
echo Panoda link bulunamadi; Not Defteri acildi.
echo Linkleri yapistirin, KAYDEDIN (Ctrl+S), sonra bu pencereye donup bir tusa basin.
pause >nul
findstr /i /c:"http" "%LISTE%" >nul 2>&1 && goto liste_hazir
echo.
echo   Listede link yok (kaydetmeyi unuttunuz mu?). Tekrar deneyin.
pause
goto dongu
:liste_hazir
for /f %%n in ('findstr /i /c:"http" "%LISTE%" ^| find /c /v ""') do echo   %%n satir link bulundu.
set "GIRDI=%LISTE%"
goto sorular
:tek
set "GIRDI=%GIRDI:"=%"
:sorular

set "EK="
echo.
echo Videoda KONUSULAN dil? (Arteta, Slot vb. icin: en; listede karisiksa bos birakin)
set "DIL="
set /p "DIL=en / tr / az  (bos = otomatik algila): "
if defined DIL set "EK=%EK% --lang %DIL%"

set "KON="
set /p "KON=Konusmaci ayrilsin mi (Teknik direktor / Muhabir)? e/h: "
if /i not "%KON%"=="e" goto fpl_sor
if defined HF_TOKEN goto token_var
echo   Bunun icin ucretsiz bir HuggingFace token gerekir (README: "Konusmaci ayirma").
echo   Token'iniz yoksa Enter'a basin; konusmaci ayirma olmadan devam edilir.
set /p "HF_TOKEN=  HF token: "
if not defined HF_TOKEN (
  echo   Token girilmedi: konusmaci ayirma olmadan devam ediliyor.
  goto fpl_sor
)
setx HF_TOKEN "%HF_TOKEN%" >nul
:token_var
set "EK=%EK% --diarize"

:fpl_sor
set "GW="
set "TAKIM="
if defined LISTE goto calistir
set /p "GW=FPL GW numarasi (bos = FPL yok): "
if not defined GW goto calistir
set /p "TAKIM=Takim kodu (or. ARS): "
if not defined TAKIM goto calistir
if defined FPL_REPO if exist "%FPL_REPO%\presser.py" goto fpl_tamam
echo   FPL repo klasoru = presser.py dosyasinin bulundugu klasor.
set "FPL_REPO="
set /p "FPL_REPO=  FPL repo klasoru (bilmiyorsaniz bos birakin = FPL yok): "
if not defined FPL_REPO goto fpl_iptal
set "FPL_REPO=%FPL_REPO:"=%"
if not exist "%FPL_REPO%\presser.py" goto fpl_iptal
setx FPL_REPO "%FPL_REPO%" >nul
goto fpl_tamam
:fpl_iptal
echo   presser.py bulunamadi: FPL adimi atlandi (metin yine de yazilir).
reg delete HKCU\Environment /v FPL_REPO /f >nul 2>&1
set "FPL_REPO="
goto calistir
:fpl_tamam
set "EK=%EK% --fpl %GW% %TAKIM%"

:calistir
echo.
"%PY%" -m stt "%GIRDI%" %EK%
echo.
echo Ciktilar: %STT_DIR%cikti
start "" "%STT_DIR%cikti" 2>nul
echo.
set "TEKRAR="
set /p "TEKRAR=Baska bir sey? (Enter = evet, h = cik): "
if /i not "%TEKRAR%"=="h" goto dongu
endlocal
