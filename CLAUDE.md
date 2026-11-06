# STT — Basın Odası

Kullanıcı Türkçe konuşur, teknik değildir: yanıtlar Türkçe, kısa; komutlar kopyala-yapıştır PowerShell;
sonuçları ekran görüntüsüyle paylaşır.

**Ne:** basın toplantısı / podcast / film videolarını metne çeviren Windows uygulaması (faster-whisper large-v3 +
pyannote konuşmacı ayırma). Amaç: Fantasy Premier League için veri seti (takım › maç › basın toplantısı).

## İki profil, az kullanım (önce bunu oku)
- Kullanıcı iki Claude hesabıyla sırayla çalışır. Ortak hafıza **`docs/YAPILACAKLAR.md`**: oturum başında oku,
  sonunda "Son durum"u güncelle. İşi her zaman güncel `main`'den başlat
  (`git fetch origin main && git checkout -B <dal> origin/main`), bitince dala ve `main`'e push et.
- Kullanım limiti dar: dosyaları bütün okuma, `grep -n` ile bul, `sed -n a,bp` ile parça oku. Büyükler:
  `stt/pipeline.py` (~1250 satır), `stt/gui.html` (~900), `stt/gui.py`, `stt/fetch.py`. Alt ajan açma.
- Ayrıntılı mimari/tarihçe: `docs/GELISTIRME.md` (yalnızca gereken başlığı oku). Konuşmacı ayırma planı:
  `docs/KONUSMACI_AYIRMA.md`. İndirme araştırması: `docs/ARASTIRMA.md`.

## Yayınlama
- Yeni sürüm = `stt/__init__.py` sürümünü artır + `main`'e push. Kullanıcıya "uygulamada Güncelle'ye basın" de;
  zip/exe/PowerShell gönderme. (`pyproject.toml` değişirse güncellemede pip çalışır; gereksiz değiştirme.)
- Testler: `pip install pytest numpy jiwer && python -m pytest -q tests` (126 test, ~3 sn). Commit mesajında model adı yazma.

## Haritası
- `pipeline.py` kuyruk/işçi yönetimi, iptal (`cancel`, `cancel_all`), bitirme · `workers.py` işçi süreçleri ·
  `engines.py` Whisper/Diarizer/GPU bilgisi · `merge.py` kelime→konuşmacı · `outputs.py` dosyalar ·
  `fetch.py` indirme zinciri · `gui.py`+`gui.html` arayüz (pywebview) · `config.py` ayarlar (`stt.toml`) ·
  `updater.py` uygulama içi güncelleme · `match.py`/`dataset.py`/`fpl.py` takım-maç-FPL · `llm.py` yapay zekâ ·
  `live.py` canlı · `channel.py` kanal/liste · `doctor.py` "Kontrol" · `voices.py` ses kütüphanesi / konuşmacı düzeltme.
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
kurulum `C:\stt` (venv `.venv`, Python 3.11, torch 2.8.0+cu128, pyannote.audio 3.4.0, faster-whisper 1.2.1).
Smart App Control açık. Günlük: `%LOCALAPPDATA%\stt\stt_app.log`.
