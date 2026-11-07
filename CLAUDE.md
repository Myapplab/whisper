# STT — Basın Odası

Kullanıcı Türkçe konuşur, teknik değildir: yanıtlar Türkçe, kısa; komutlar kopyala-yapıştır PowerShell;
sonuçları ekran görüntüsüyle paylaşır.

**Ne:** basın toplantısı / podcast / film videolarını metne çeviren Windows uygulaması (faster-whisper large-v3 +
pyannote konuşmacı ayırma). Amaç: Fantasy Premier League için veri seti (takım › maç › basın toplantısı).

## İki profil, önce kalite (önce bunu oku)
- Kullanıcı iki Claude hesabıyla sırayla çalışır. Ortak hafıza **`docs/YAPILACAKLAR.md`**: oturum başında oku,
  sonunda "Son durum"u güncelle. İşi her zaman güncel `main`'den başlat
  (`git fetch origin main && git checkout -B <dal> origin/main`), bitince dala ve `main`'e push et.
- **Öncelik kalite; kullanım (token/limit) ikinci planda.** Kullanıcının kararı (2026-11), kendisi geri alana kadar geçerli.
  Eskiden burada yazan "dosyaları bütün okuma, parça oku, alt ajan açma" kuralı kaldırıldı:
  - Değiştireceğin kodu, onu çağıran ve onun çağırdığı yerleri tam oku; gerekirse dosyanın tamamını. Tahminle değil,
    okuyarak karar ver. Büyük dosyalar: `stt/pipeline.py` (~1500 satır), `stt/gui.py` (~1100), `stt/gui.html` (~1000),
    `stt/fetch.py`, `stt/engines.py` (~850).
  - Kütüphane davranışından (faster-whisper, pyannote, torch, yt-dlp...) emin değilsen kaynağına ya da belgesine bak.
  - Hatayı önce testle göster, sonra düzelt; her davranış değişikliğine test ekle; bütün testler geçmeden push etme.
  - Bitirmeden önce değişikliği kenar durumlar için baştan gözden geçir (iptal, GPU hatası, boş/bozuk girdi, Windows).
  - Denemediğini açıkça yaz ("Windows'ta / ekran kartında denenmedi"); doğrulamadığın şeye "çalışıyor" deme.
  - Alt ajan: bağımsız ikinci göz (kod incelemesi) ya da paralel araştırma kaliteyi artıracaksa açılabilir.
- Ayrıntılı mimari/tarihçe: `docs/GELISTIRME.md`. Konuşmacı ayırma planı: `docs/KONUSMACI_AYIRMA.md`.
  İndirme araştırması: `docs/ARASTIRMA.md`.
- Dış denetim raporu (Gemini): `docs/GEMINI_4_ARGON_BULGULAR.md`. Konuşmacı ayırma işinden önce oku, ama raporun bir
  kısmı yanlış: ör. "var olmayan model" dediği `claude-haiku-5-5` / `claude-sonnet-5-5` / `gpt-5-mini` /
  `gpt-transcribe` gerçek modellerdir, değiştirilmez. Her maddenin doğrulama sonucu raporun başındaki
  **"0. Claude doğrulaması"** bölümünde; yanlış (❌) maddeleri uygulama, ❓ olanları ölçümle karar ver.

## Yayınlama
- Yeni sürüm = `stt/__init__.py` sürümünü artır + `main`'e push. Kullanıcıya "uygulamada Güncelle'ye basın" de;
  zip/exe/PowerShell gönderme. (`pyproject.toml` değişirse güncellemede pip çalışır; gereksiz değiştirme.)
- Testler: `pip install pytest numpy jiwer && python -m pytest -q tests` (592 test, ~12 sn; arayüz betiği için node varsa sözdizimi testi de çalışır). Commit mesajında model adı yazma.

## Haritası
- `pipeline.py` kuyruk/işçi yönetimi, iptal (`cancel`, `cancel_all`), bitirme · `workers.py` işçi süreçleri ·
  `engines.py` Whisper/Diarizer/GPU bilgisi · `merge.py` kelime→konuşmacı · `outputs.py` dosyalar ·
  `fetch.py` indirme zinciri · `gui.py`+`gui.html` arayüz (pywebview) · `config.py` ayarlar (`stt.toml`) ·
  `updater.py` uygulama içi güncelleme · `match.py`/`dataset.py`/`fpl.py` takım-maç-FPL · `llm.py` yapay zekâ ·
  `live.py` canlı · `channel.py` kanal/liste · `doctor.py` "Kontrol" · `voices.py` ses kütüphanesi / konuşmacı düzeltme · `governor.py` gece modu (ısıya göre görev arası mola) · `power.py` güç / saat sınırı · `tune.py` konuşmacı ayırmayı düzeltilmiş videolarla eğitme (`stt egit`) ·
  `reclassify.py` veri setini metni yeniden çevirmeden yeniden sınıflandırma ("Veri seti > Yeniden sınıflandır",
  `stt yeniden-siniflandir`). Sezonluk elle tutulan listeler: `match.PL_MANAGERS` + `MANAGER_SEASON`,
  `dataset.PL_FALLBACK`, `fpl.TEAM_NAMES` (her sezon başında güncelle).
- GUI kuralı: `App`'in `_` ile başlamayan metotları JS'e açık; olaylar yalnızca `App._push` ile (evaluate_js'i
  başka yerden çağırma, WebView2 takılıyordu).

## Değişmez kurallar
- Monitörlü karta (RTX 3080) kendiliğinden yük verme; konuşmacı ayırma yalnızca GPU'da (`diar_cpu`=false),
  ayrı süreçte. Kullanıcıda işlemci zayıf (i3-10100), fan şikâyeti var.
- Anahtar/token (HF_TOKEN, API anahtarları) repoya asla yazılmaz (`stt.toml` .gitignore'da). Depo herkese açık.
- Üçüncü taraf "downloader" sitelerine link gönderilmez; korsan/DRM'li sitelerin aşılmasına yardım edilmez.
- `installer/kur.ps1` UTF-8 BOM'lu kalmalı. `installer/guncelle.ps1` yalnız ASCII.

## Kullanıcının makinesi
Windows 11, RTX 3090 24 GB (cuda:1, işler burada) + RTX 3080 10 GB (cuda:0, monitörlü), i3-10100, 24 GB RAM,
kurulum `C:\stt` (venv `.venv`, Python 3.11, torch 2.8.0+cu128, pyannote.audio 4.0.7 (community-1; 1.7.0'dan beri),
faster-whisper 1.2.1).
Smart App Control açık. Günlük: `%LOCALAPPDATA%\stt\stt_app.log`.
