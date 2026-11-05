; STT - Basin Odasi : Windows kurulum programi (NSIS 3)
Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

!define APPNAME "STT - Basın Odası"
!define VERSION "1.2.0"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\STT"

Name "${APPNAME}"
OutFile "STT-Kurulum.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\STT"
ShowInstDetails show
ShowUninstDetails show
SetCompressor /SOLID lzma
BrandingText "STT ${VERSION}"

VIProductVersion "1.2.0.0"
VIAddVersionKey /LANG=1055 "ProductName" "${APPNAME}"
VIAddVersionKey /LANG=1055 "FileDescription" "${APPNAME} kurulumu"
VIAddVersionKey /LANG=1055 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1055 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1055 "LegalCopyright" "STT"

!define MUI_ICON "staged\stt\assets\stt.ico"
!define MUI_UNICON "staged\stt\assets\stt.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "STT - Basın Odası kurulumu"
!define MUI_WELCOMEPAGE_TEXT "Basın toplantısı videolarını metne çeviren STT kurulacak.$\r$\n$\r$\nKurulum internetten şunları indirir (ilk kurulumda toplam ~6 GB, 10-20 dakika):$\r$\n  • Python 3.11 (yoksa)$\r$\n  • PyTorch + NVIDIA CUDA kütüphaneleri$\r$\n  • Whisper large-v3 modeli$\r$\n  • ffmpeg ve deno$\r$\n$\r$\nYönetici izni gerekmez. Kurulum yarıda kalırsa tekrar çalıştırın; biten adımlar atlanır.$\r$\nKurulumdan sonra yalnızca masaüstündeki STT simgesini kullanın; başka dosya gerekmez."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "STT kuruldu"
!define MUI_FINISHPAGE_TEXT "Masaüstündeki ve Başlat menüsündeki STT simgesiyle açabilirsiniz.$\r$\n$\r$\nBir sorun olursa uygulamadaki Kontrol düğmesine basın."
!define MUI_FINISHPAGE_RUN "$INSTDIR\.venv\Scripts\pythonw.exe"
!define MUI_FINISHPAGE_RUN_PARAMETERS "-m stt.gui"
!define MUI_FINISHPAGE_RUN_TEXT "STT'yi şimdi aç"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Turkish"

Function .onInit
  ; önceki elle kurulum (C:\stt) varsa onu güncelle: hazır ortam ve veriler korunur
  ${If} ${FileExists} "C:\stt\.venv\Scripts\python.exe"
    StrCpy $INSTDIR "C:\stt"
  ${EndIf}
FunctionEnd

Section "STT" SecMain
  SetOutPath "$INSTDIR"
  File /r "staged\*.*"

  ; veri seti klasörü: önceki 'cikti' varsa o, yoksa Belgeler\STT Veri Seti
  StrCpy $0 "$DOCUMENTS\STT Veri Seti"
  ${If} ${FileExists} "$INSTDIR\cikti\*.*"
    StrCpy $0 ""
  ${EndIf}

  DetailPrint "Bileşenler indirilip kuruluyor. Bu pencere kapanmadan bekleyin..."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\kur.ps1" -InstDir "$INSTDIR" -DataDir "$0"'
  Pop $1
  ${If} $1 != 0
    MessageBox MB_ICONSTOP "Kurulum tamamlanamadı (kod $1).$\r$\n$\r$\nPenceredeki son satırlarda sebep yazıyor. İnternet bağlantısını kontrol edip kurulumu tekrar çalıştırın; biten adımlar atlanır."
    Abort
  ${EndIf}

  CreateShortcut "$DESKTOP\STT.lnk" "$INSTDIR\.venv\Scripts\pythonw.exe" "-m stt.gui" "$INSTDIR\stt\assets\stt.ico" 0 SW_SHOWNORMAL "" "${APPNAME}"
  CreateShortcut "$SMPROGRAMS\STT.lnk" "$INSTDIR\.venv\Scripts\pythonw.exe" "-m stt.gui" "$INSTDIR\stt\assets\stt.ico" 0 SW_SHOWNORMAL "" "${APPNAME}"

  WriteUninstaller "$INSTDIR\Kaldir.exe"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "STT"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\stt\assets\stt.ico"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Kaldir.exe"'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\STT.lnk"
  Delete "$SMPROGRAMS\STT.lnk"
  ; yalnızca uygulama dosyaları silinir; veri seti (cikti / Belgeler\STT Veri Seti) ve ayarlar kalır
  RMDir /r "$INSTDIR\.venv"
  RMDir /r "$INSTDIR\stt"
  RMDir /r "$INSTDIR\tests"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\stt.egg-info"
  Delete "$INSTDIR\kur.ps1"
  Delete "$INSTDIR\stt.bat"
  Delete "$INSTDIR\install.bat"
  Delete "$INSTDIR\pyproject.toml"
  Delete "$INSTDIR\constraints.txt"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\BENI_OKU.txt"
  Delete "$INSTDIR\stt.toml.example"
  Delete "$INSTDIR\.gitignore"
  Delete "$INSTDIR\Kaldir.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
SectionEnd
