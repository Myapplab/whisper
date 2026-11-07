; STT - Basın Odası : Windows kurulum programı (NSIS 3, Unicode)
; Derleme: installer/build.sh  ->  dist/STT-Kurulum.exe
Unicode true
ManifestDPIAware true          ; yüksek DPI ekranlarda yazılar bulanık görünmesin
ManifestSupportedOS all
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "WinMessages.nsh"

!define APPNAME "STT - Basın Odası"
!define VERSION "1.9.24"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\STT"
!define STEPS 9

Name "${APPNAME}"
Caption "${APPNAME} ${VERSION} Kurulumu"
OutFile "STT-Kurulum.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\STT"
ShowInstDetails show
ShowUninstDetails show
SetCompressor /SOLID lzma
BrandingText "STT ${VERSION}"
SetFont "Segoe UI" 9

VIProductVersion "1.9.24.0"
VIAddVersionKey /LANG=1055 "ProductName" "${APPNAME}"
VIAddVersionKey /LANG=1055 "FileDescription" "${APPNAME} kurulumu"
VIAddVersionKey /LANG=1055 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1055 "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=1055 "LegalCopyright" "STT"

; ---------------------------------------------------------------- görünüm
!define MUI_ICON "staged\stt\assets\stt.ico"
!define MUI_UNICON "staged\stt\assets\stt.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "header.bmp"
!define MUI_HEADERIMAGE_BITMAP_STRETCH "FitControl"
!define MUI_ABORTWARNING
!define MUI_ABORTWARNING_TEXT "Kurulum yarıda kesilsin mi?$\r$\n$\r$\nDaha sonra tekrar çalıştırırsanız biten adımlar atlanır."

; ---------------------------------------------------------------- sayfalar
!define MUI_WELCOMEPAGE_TITLE "STT - Basın Odası$\r$\nkurulumuna hoş geldiniz"
!define MUI_WELCOMEPAGE_TEXT "STT, basın toplantısı videolarını (YouTube, X, Instagram, kulüp siteleri ya da bilgisayarınızdaki dosyalar) soru-cevap biçiminde metne çevirir.$\r$\n$\r$\nKurulum gerekli bileşenleri internetten indirir:$\r$\n$\r$\n     -  Python 3.11 (bilgisayarda yoksa)$\r$\n     -  PyTorch ve NVIDIA CUDA kütüphaneleri (~3 GB)$\r$\n     -  Whisper large-v3 konuşma tanıma modeli (~3 GB)$\r$\n     -  ffmpeg ve deno (video indirme için)$\r$\n$\r$\nİlk kurulum, internet hızına göre 10-30 dakika sürer ve en az 9 GB boş alan ister. Yönetici izni gerekmez.$\r$\n$\r$\nKurulum yarıda kalırsa tekrar çalıştırın; biten adımlar atlanır."
!insertmacro MUI_PAGE_WELCOME

!define MUI_DIRECTORYPAGE_TEXT_TOP "STT bu klasöre kurulacak. Varsayılan klasörü değiştirmeniz gerekmez.$\r$\n$\r$\nMetinler (veri seti) ayrıca Belgeler\STT Veri Seti klasörüne kaydedilir; uygulamayı kaldırsanız da silinmez."
!insertmacro MUI_PAGE_DIRECTORY

!define MUI_PAGE_HEADER_TEXT "Kuruluyor"
!define MUI_PAGE_HEADER_SUBTEXT "Bileşenler indirilip kuruluyor. Lütfen pencereyi kapatmayın."
!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Kurulum tamamlandı"
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "Bütün bileşenler kuruldu."
!define MUI_INSTFILESPAGE_ABORTHEADER_TEXT "Kurulum tamamlanamadı"
!define MUI_INSTFILESPAGE_ABORTHEADER_SUBTEXT "Sebep, listenin son satırlarında yazıyor."
!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_TITLE "STT kuruldu"
!define MUI_FINISHPAGE_TEXT "Masaüstündeki ya da Başlat menüsündeki STT simgesiyle açabilirsiniz.$\r$\n$\r$\nİlk açılışta model ekran kartına yüklenir (1-2 dakika).$\r$\n$\r$\nBir sorun olursa uygulamadaki Kontrol düğmesine basın."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "STT'yi şimdi aç"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\BENI_OKU.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Kısa kullanım kılavuzunu göster"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!insertmacro MUI_PAGE_FINISH

!define MUI_UNCONFIRMPAGE_TEXT_TOP "STT uygulaması kaldırılacak.$\r$\n$\r$\nVeri setiniz (metinler) ve ayarlarınız SİLİNMEZ."
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Turkish"

; ---------------------------------------------------------------- yardımcılar
Var DataDir
Var PBar

Function .onInit
  ; önceki elle kurulum (C:\stt) varsa onu güncelle: hazır ortam ve veriler korunur
  ${If} ${FileExists} "C:\stt\.venv\Scripts\python.exe"
    StrCpy $INSTDIR "C:\stt"
  ${EndIf}
FunctionEnd

Function LaunchApp
  ; uygulama, kurulumun eski ortam değişkenleriyle değil, kendi klasöründe açılsın
  SetOutPath "$INSTDIR"
  Exec '"$INSTDIR\.venv\Scripts\pythonw.exe" -m stt.gui'
FunctionEnd

; ilerleme çubuğunu adımın ağırlığına göre ayarla (NSIS aralığı 0..30000)
!macro SetBar PCT
  FindWindow $0 "#32770" "" $HWNDPARENT
  GetDlgItem $PBar $0 1004
  IntOp $1 ${PCT} * 300
  SendMessage $PBar ${PBM_SETPOS} $1 0
!macroend

; bir kurulum adımı: üstte "Adım N/9: ..." yazar, kur.ps1'i çalıştırır, hata olursa durur
!macro RunStep NUM PCT NAME LABEL
  SetDetailsPrint both
  DetailPrint " "
  DetailPrint "Adım ${NUM}/${STEPS}: ${LABEL}"
  SetDetailsPrint listonly
  !insertmacro SetBar ${PCT}
  nsExec::ExecToLog /OEM 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\kur.ps1" -InstDir "$INSTDIR" -DataDir "$DataDir" -Step ${NAME}'
  Pop $1
  ${If} $1 != 0
    SetDetailsPrint both
    DetailPrint "Adım ${NUM} tamamlanamadı."
    MessageBox MB_ICONSTOP|MB_OK "Kurulum tamamlanamadı (adım ${NUM}: ${LABEL}).$\r$\n$\r$\nSebep, penceredeki listenin son satırlarında yazıyor. Ayrıntılı günlük:$\r$\n$INSTDIR\kurulum.log$\r$\n$\r$\nİnternet bağlantısını kontrol edip kurulumu tekrar çalıştırın; biten adımlar atlanır."
    Abort
  ${EndIf}
!macroend

; ---------------------------------------------------------------- kurulum
Section "STT" SecMain
  SetDetailsPrint both
  DetailPrint "Uygulama dosyaları kopyalanıyor"
  SetDetailsPrint listonly
  SetOutPath "$INSTDIR"
  File /r "staged\*.*"
  SetDetailsPrint both

  ; veri seti klasörü: önceki 'cikti' varsa o kalır, yoksa Belgeler\STT Veri Seti
  StrCpy $DataDir "$DOCUMENTS\STT Veri Seti"
  ${If} ${FileExists} "$INSTDIR\cikti\*.*"
    StrCpy $DataDir ""
  ${EndIf}

  !insertmacro RunStep 1 3  check  "Ön kontrol (boş alan, internet, ekran kartı)"
  !insertmacro RunStep 2 6  python "Python 3.11"
  !insertmacro RunStep 3 10 venv   "Sanal ortam"
  !insertmacro RunStep 4 15 torch  "PyTorch + NVIDIA CUDA (~3 GB, birkaç dakika)"
  !insertmacro RunStep 5 50 libs   "Uygulama kütüphaneleri"
  ; paketteki kod derlendiği günkü sürüm: en son sürüme geç (eski exe yeni kurulumu eski koda döndürmesin)
  !insertmacro RunStep 6 62 update "En son sürüm (GitHub)"
  !insertmacro RunStep 7 65 tools  "ffmpeg ve deno"
  !insertmacro RunStep 8 70 config "Ayarlar"
  !insertmacro RunStep 9 72 model  "Whisper large-v3 modeli (~3 GB)"
  !insertmacro SetBar 98

  DetailPrint "Kısayollar oluşturuluyor"
  SetOutPath "$INSTDIR"   ; kısayolların çalışma klasörü
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
  DetailPrint "Kurulum tamamlandı."
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\STT.lnk"
  Delete "$SMPROGRAMS\STT.lnk"
  ; yalnızca uygulama dosyaları silinir; veri seti (cikti / Belgeler\STT Veri Seti) ve stt.toml kalır
  RMDir /r "$INSTDIR\.venv"
  RMDir /r "$INSTDIR\stt"
  RMDir /r "$INSTDIR\tests"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\stt.egg-info"
  Delete "$INSTDIR\kur.ps1"
  Delete "$INSTDIR\kurulum.log"
  Delete "$INSTDIR\stt.bat"
  Delete "$INSTDIR\install.bat"
  Delete "$INSTDIR\pyproject.toml"
  Delete "$INSTDIR\constraints.txt"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\BENI_OKU.txt"
  Delete "$INSTDIR\CLAUDE.md"
  Delete "$INSTDIR\stt.toml.example"
  Delete "$INSTDIR\.gitignore"
  Delete "$INSTDIR\Kaldir.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
SectionEnd
