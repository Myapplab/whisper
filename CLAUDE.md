# STT — Basın Odası (proje notları / devir)

Kullanıcı Türkçe konuşur, teknik olmayan bir kullanıcıdır: adımları kısa, kopyala-yapıştır
PowerShell komutlarıyla ver; sonuçları ekran görüntüsüyle paylaşır. Yanıtları Türkçe yaz.

## Ne bu?
Basın toplantısı videolarını (YouTube, X, Instagram, kulüp siteleri, yerel dosyalar) metne çeviren
Windows aracı. Amaç: **Fantasy Premier League modeli için veri seti** (takım › maç › basın toplantısı).

- Motor: faster-whisper large-v3 (float16, beam 5, vad_filter, word_timestamps, toplu/batched mod),
  iki GPU paralel (her GPU'ya bir işçi süreci, ortak kuyruk), uzun dosyalar Silero VAD ile sessizlikten bölünür.
- İsteğe bağlı konuşmacı ayırma: pyannote/speaker-diarization-3.1 (HF token) → en uzun konuşan teknik
  direktör, diğerleri muhabir → `.txt` soru-cevap biçiminde.
- Çıktı: `.txt` (soru-cevap), `.srt`, `.json` (kelime zamanı + konuşmacı), `ozet.md`, `mac.json`,
  `index.csv` / `index.jsonl`.
- Veri seti düzeni: `<out>/Premier League/<Takım>/<maç tarihi> GWxx <Ev> vs <Dep>/<tarih> mac-oncesi - <TD>.txt`.
  Maç + GW, FPL API fikstürlerinden (fantasy.premierleague.com/api/bootstrap-static, /fixtures) eşleşir
  (`stt/dataset.py`), önbellek `<out>/.stt/fpl_cache.json`.
- Takım/TD adı başlık + YouTube kanal adından (`stt/match.py`; Türkçe/Azerice kalıplar dahil).
- FPL: `presser.py` (kullanıcının FPL reposu) → `inputs/pressers/GWxx/TAKIM.txt` yazılır ve
  `python3 presser.py <dosya>` çalışır; PL maçı bulunursa GW/takım kodu otomatik.
- Atlama: dosya SHA-256 / normalize link / `extractor:id`; sürdürme: biten parçalar `<out>/.stt/is/<key>/`.
- Yedek motor: `--engine openai` (varsayılan model `gpt-transcribe`, OPENAI_API_KEY).
- WER: `stt olc [klasör] [--ornek]` (jiwer + whisper normalizer, hedef < %8).

## Arayüzler
- **Uygulama:** `pythonw -m stt.gui` (pywebview + Edge WebView2; `stt/gui.py` + `stt/gui.html`).
  Python ↔ JS: `App` sınıfının `_` ile başlamayan metotları JS'e açıktır (iç alanlar `_` ile başlamalı!).
  Olaylar `Pipeline.events` → `window.__stt.onEvent(...)`.
- **CLI:** `stt.bat <link|dosya|klasör|liste.txt> [--lang en] [--diarize] [--fpl 11 ARS]`,
  alt komutlar: `kontrol [--derin]`, `olc`, `izle KLASOR`, `durum`. Çift tıklanınca etkileşimli
  (panodaki linkleri okur).
- **Kurulum:** `installer/stt.nsi` (NSIS) + `installer/kur.ps1`; `installer/build.sh` → `dist/STT-Kurulum.exe`.
  Eski yol: `install.bat` (venv + torch cu128 + `pip install -c constraints.txt -e .[diarize,test]`).

## Kullanıcının makinesi (önemli)
- Windows 11, RTX 3090 (24 GB) + RTX 3080 (10 GB), sürücü 610.88. Kurulum `C:\stt` (venv `C:\stt\.venv`),
  Python 3.11.9, torch 2.8.0+cu128, ctranslate2 4.8.5, pyannote.audio 3.4.0, faster-whisper 1.2.1.
- **Smart App Control açık**: internetten inen .bat/.exe engellenir → `Unblock-File` gerekir.
  Zip içinden .bat çalıştırınca "Kurulum bulunamadı" hatası alır (geçici klasör) → `C:\stt`'ye çıkarılmalı.
- Sistemde `CUDA_VISIBLE_DEVICES` vardı (3080 görünmüyordu) → kod bunu kaldırıyor;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` (cuda:0 = 3080, cuda:1 = 3090, nvidia-smi sırası).
- İki model aynı anda yüklenince `mkl_malloc: failed to allocate memory` (RAM) → modeller sırayla
  yükleniyor (`WorkerPool.load_lock`), işçiler torch import etmiyor.
- HF token kullanıcının ortam değişkeninde (`HF_TOKEN`) — **repoya asla yazma** (`stt.toml` .gitignore'da).

## Bilinen sınırlar / yapılacaklar
- Windows'ta henüz doğrulanmadı: GUI (pywebview) ilk açılış, setup.exe + SAC davranışı, pyannote ile
  gerçek konuşmacı ayırma (kullanıcıda "ayırmıyor" şikâyeti vardı; `stt kontrol` artık modeli
  yükleyip dener — sonucunu kullanıcıdan iste), FPL API ile gerçek eşleşme, gerçek WER (large-v3).
- Yerel dosyalarda kanal adı yok → başlıkta yalnızca "Everton vs Coventry City" varsa TD'nin takımı
  bilinemez (Lampard → Everton'a düştü). Fikir: GUI'de takımı elle düzeltme, ya da TD→takım eşleştirmesi.
- `MANAGER_HINTS` / `TEAMS` (`stt/match.py`) elle tutulan listeler; sezon değişince güncellenmeli.
- setup.exe imzasız; paylaşım için kod imzalama (ör. Azure Trusted Signing) gerekir.
- Kullanıcı ileride uygulamayı paylaşmak istiyor (tek kurulum dosyası).

## Geliştirme (bulut/Linux)
- Testler: `python -m pytest -q tests` (26 test).
- GPU yok ve HuggingFace engelliyse test modeli: GitHub `rhasspy/models` `asr_faster-whisper-small-int8`
  + `dev/make_tokenizer.py` (openai-whisper tiktoken → tokenizer.json) + `dev/add_alignment_heads.py`.
  `--model <klasör> --gpus cpu,cpu` ile uçtan uca çalışır.
- GUI testi: `dev/gui_bridge.py` (pywebview yerine HTTP köprüsü) + Playwright (`dev/gui_drive.js`,
  `dev/gui_screens.js`). Ayrıntı: `dev/README.md`.
- Commit/PR mesajlarında model adı yazma.
