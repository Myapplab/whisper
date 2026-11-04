# stt — basın toplantısı / video / ses → metin

Linki ya da dosyayı verirsiniz; `stt` sesi indirir, iki ekran kartında paralel olarak
**faster-whisper large-v3** ile metne çevirir, isterseniz konuşmacıları ayırır
(teknik direktör / muhabir) ve `.txt`, `.srt`, `.json`, `ozet.md` dosyalarını yazar.
`--fpl` verirseniz metni FPL reposuna koyar ve `presser.py`'yi çalıştırır.

```bat
stt.bat "https://www.youtube.com/watch?v=XXXX" --fpl 11 ARS
```

## Uygulama (önerilen)

Masaüstündeki **STT** simgesine çift tıklayın (terminal açılmaz).

- **Yeni iş:** linkleri yapıştırın (20 tane de olur; yazı arasına karışık da olsa linkler bulunur),
  ya da *Dosya ekle / Klasör ekle* / pencereye sürükle-bırak. Dil, *Soru–cevap olarak ayır* ve
  *FPL presser.py* seçenekleri, sonra **Başlat**.
- **Kuyruk:** her video için durum, ilerleme çubuğu, bulunan takım / maç / GW / maç öncesi-sonrası,
  hata olursa sebebi. *Metni göster*, *Klasörü aç*.
- **Üst şerit:** her ekran kartının anlık durumu (boşta / çeviriyor / konuşmacı ayırıyor). Masaüstü için boş
  bırakılan kart gri görünür.
- **Durdur:** sıradaki ve çalışan bütün işleri durdurur; biten parçalar saklanır, aynı linki tekrar eklerseniz
  kaldığı yerden devam eder.
- **Veri seti:** Premier League › takım › maç ağacı, basın toplantısı/takım/maç sayıları, `index.csv`.
- **Ayarlar:** veri seti klasörü, HuggingFace token, FPL repo klasörü, çerez tarayıcısı, ekran kartı kullanımı
  (*Önerilen*: monitörün bağlı olmadığı kart — bilgisayar akıcı kalır; *Hepsi*: en hızlı).
- Uygulama açılmazsa günlük: `%LOCALAPPDATA%\stt\stt_app.log`, ya da konsolda
  `C:\stt\.venv\Scripts\python.exe -m stt.gui`.

## Veri seti düzeni

```
cikti\
  Premier League\
    Arsenal\
      2026-11-08 GW11 Arsenal vs Chelsea\
        mac.json                                    fikstür bilgisi (FPL); "ek_bilgiler" alanına maça ait başka veri eklenebilir
        2026-11-07 mac-oncesi - Mikel Arteta.txt    soru-cevap metni
        2026-11-07 mac-oncesi - Mikel Arteta.json   kelime zamanları + konuşmacı
        2026-11-07 mac-oncesi - Mikel Arteta.srt
        2026-11-07 mac-oncesi - Mikel Arteta - ozet.md
      2026-11-04 Champions League Arsenal vs Slavia Prague\
  Diger Takimlar\
    Galatasaray\ ...
  index.csv / index.jsonl                           her basın toplantısı bir satır (model eğitimi için)
```

- Takım, YouTube kanal adından (ör. "Arsenal") ya da başlıktan bulunur; teknik direktör başlıktan.
- **Maç ve GW** FPL'nin herkese açık fikstür verisinden otomatik eşleştirilir: takımın, basın
  toplantısı tarihinden 2 gün önce ile 6 gün sonrası arasındaki maçı. Maçtan önce ise
  `mac-oncesi`, sonra ise `mac-sonrasi`.
- Rakip PL takımı değilse (Şampiyonlar Ligi vb.) turnuva başlıktan okunur, tarih = yayın tarihi.
- `FPL presser.py` açıksa ve FPL repo ayarlıysa GW ve takım kodu (ARS…) otomatik verilir.
- `index.csv` sütunları: key, presser_date, phase, league, team, opponent, competition, gw,
  match_date, home, away, manager, language, duration_s, questions, words, diarized, url, txt, json.
- Yerel dosyalarda kanal adı olmadığı için takım dosya adından tahmin edilir. Dosya adında yalnızca
  "Everton vs Coventry City" yazıyorsa teknik direktörün hangi takımda olduğu bilinemez;
  YouTube linki vermek ya da dosya adına takımı yazmak en güvenlisi.

---

## İçindekiler

1. [Neler yapar](#neler-yapar)
2. [Kurulum (Windows 11)](#kurulum-windows-11)
3. [Kullanım](#kullanım)
4. [Çıktılar](#çıktılar)
5. [Konuşmacı ayırma](#konuşmacı-ayırma)
6. [FPL entegrasyonu](#fpl-entegrasyonu)
7. [Atlama ve kaldığı yerden devam](#atlama-ve-kaldığı-yerden-devam)
8. [İki GPU nasıl kullanılıyor](#iki-gpu-nasıl-kullanılıyor)
9. [OpenAI yedek motoru](#openai-yedek-motoru)
10. [Kelime hata oranı (WER) ölçümü](#kelime-hata-oranı-wer-ölçümü)
11. [Tüm seçenekler](#tüm-seçenekler)
12. [Sorun giderme](#sorun-giderme)
13. [Proje yapısı](#proje-yapısı)

---

## Neler yapar

| Girdi | Örnek |
|---|---|
| yt-dlp'nin desteklediği her site | YouTube, X, Instagram, TikTok, Facebook, Vimeo, kulüp siteleri… |
| Yerel video/ses dosyası | `mp4 mkv mov webm avi mp3 m4a wav flac ogg opus …` |
| Klasör (alt klasörler dahil) | `stt.bat D:\basin` |
| Link listesi | `stt.bat linkler.txt` (satır başına bir link ya da dosya yolu, `#` yorum) |
| İzleme klasörü | `stt.bat izle D:\gelen` (yeni gelen dosya otomatik işlenir) |

Akış:

1. **İndirme:** yt-dlp ile yalnızca ses alınır. Önce çerezsiz denenir. Giriş/bot doğrulaması
   isteyen sitelerde `--cookies-from-browser chrome` ile tekrar denenir. Instagram/Facebook'ta
   doğrudan çerezle başlanır. O da olmazsa sayfadaki `m3u8`/`mp4` linki bulunur
   (`<video>`, `og:video`, JS içindeki kaçışlı JSON) ve ffmpeg ile çekilir.
   **DRM korumalı** içerik (Widevine/FairPlay/PlayReady, SAMPLE-AES) tespit edilir ve atlanır.
2. **Dönüştürme:** ffmpeg ile **16 kHz mono WAV**.
3. **Bölme:** 8 dakikadan uzun dosyalar Silero VAD ile **sessizlik noktalarından** ~5 dakikalık
   parçalara ayrılır. Parçalar zaman çizgisini boşluksuz kaplar, yani ses kaybolmaz.
4. **Tanıma:** her GPU'da bir işçi süreci, ortak kuyruk. Ayarlar: `large-v3`, `float16`,
   `beam_size=5`, `vad_filter=True`, `word_timestamps=True`. Dil `--lang en|tr|az` ile verilir,
   verilmezse otomatik algılanır.
5. **Konuşmacı ayırma (isteğe bağlı):** `pyannote/speaker-diarization-3.1`. En uzun konuşan
   **Teknik direktör**, diğerleri **Muhabir** olarak etiketlenir.
6. **Çıktılar:** `.txt`, `.srt`, `.json` (kelime zamanları + konuşmacı), `ozet.md`.
7. **FPL:** `inputs/pressers/GWxx/TAKIM.txt` yazılır, ardından `python3 presser.py <dosya>` çalışır.

---

## Kurulum (Windows 11)

**Önerilen:** `dist/STT-Kurulum.exe` dosyasını çalıştırın. Yönetici izni istemez; Python, PyTorch, ffmpeg,
deno ve modeli kendisi indirir (8 adım, ilerleme çubuğu ve Türkçe açıklamalarla). Yarıda kalırsa tekrar
çalıştırın, biten adımlar atlanır. Ayrıntılı günlük: kurulum klasöründe `kurulum.log`.
Smart App Control imzasız dosyayı engellerse: dosyaya sağ tık › Özellikler › *Engellemeyi kaldır*.

**Elle kurulum (geliştirici):**

Gerekenler: NVIDIA sürücüsü (CUDA 12.8 destekleyen, 570+), Python 3.11, internet.

```bat
winget install Python.Python.3.11
git clone <bu repo> C:\araclar\stt
cd C:\araclar\stt
install.bat
```

`install.bat` şunları yapar:

- `.venv` adında bir Python 3.11 sanal ortamı oluşturur.
- **PyTorch 2.8 + CUDA 12.8** kurar. cuBLAS/cuDNN DLL'leri de bununla gelir;
  faster-whisper (CTranslate2) bu DLL'leri kullanır, ayrıca CUDA Toolkit kurmanız gerekmez.
- faster-whisper, yt-dlp (+ yt-dlp-ejs), pyannote.audio 3.4, jiwer, openai paketlerini kurar.
- ffmpeg ve **deno** (yt-dlp'nin YouTube için istediği JS çalışma zamanı) yoksa `winget` ile kurar.
- `stt.toml.example` dosyasını `stt.toml` olarak kopyalar.
- Son olarak `stt kontrol` çalıştırır.

> winget yeni bir şey kurduysa PATH'in yenilenmesi için **yeni bir terminal** açıp
> `stt.bat kontrol` komutunu tekrar çalıştırın.

Her yerden `stt` yazabilmek için klasörü PATH'e ekleyin:

```bat
setx PATH "%PATH%;C:\araclar\stt"
```

### Kurulum kontrolü

```bat
stt.bat kontrol          :: Python, ffmpeg, yt-dlp, deno, CUDA, GPU'lar, model, pyannote, OpenAI, FPL
stt.bat kontrol --derin  :: modeli her GPU'ya yükleyip kısa bir tanıma yapar (cuDNN/cuBLAS testi)
```

Beklenen çıktıdan bir kesit:

```
[ OK ] torch              2.8.0+cu128, CUDA 12.8, cuDNN 91002
[ OK ] cuda:0             NVIDIA GeForce RTX 3090, 24.0 GB
[ OK ] cuda:1             NVIDIA GeForce RTX 3080, 10.0 GB
[ OK ] ctranslate2        4.x, 2 GPU, float16=True
```

İlk çalıştırmada `large-v3` modeli (~3 GB) HuggingFace önbelleğine iner. İki işçi aynı anda
indirmeye çalışmasın diye model ana süreçte bir kez indirilir.

---

## Kullanım

```bat
:: tek link
stt.bat "https://www.youtube.com/watch?v=XXXX"

:: dili sabitle (en / tr / az, yoksa otomatik)
stt.bat "https://x.com/Arsenal/status/123" --lang en

:: yerel video
stt.bat "D:\videolar\basin toplantisi.mp4" --lang tr

:: klasör (alt klasörler dahil)
stt.bat D:\basin --lang en

:: link listesi
stt.bat linkler.txt

:: konuşmacı ayırma
stt.bat "https://www.youtube.com/watch?v=XXXX" --diarize

:: FPL: inputs\pressers\GW11\ARS.txt yaz + presser.py çalıştır
stt.bat "https://www.youtube.com/watch?v=XXXX" --fpl 11 ARS

:: izleme klasörü (Ctrl+C ile durur)
stt.bat izle D:\gelen --lang en --diarize

:: işlenenlerin listesi
stt.bat durum
```

> **cmd ipucu:** İçinde `&` geçen linkleri (ör. `...watch?v=X&t=30`) mutlaka **tırnak** içine alın.

### Toplu link (ör. 20 video)

`stt.bat`'a çift tıklayın, link sorusunda **hiçbir şey yazmadan Enter**'a basın. Not Defteri'nde
`linkler.txt` açılır: her satıra bir link yapıştırın, kaydedin, pencereye dönüp bir tuşa basın.
Hepsi sırayla indirilir ve iki ekran kartında işlenir. Daha önce işlenenler atlanır.
FPL için linkin yanına GW ve takım yazılabilir: `https://youtube.com/... 11 ARS`.

### Dosya adları

Başlık ve kanal adından maç ve teknik direktör bulunur:
`PRESS CONFERENCE | Mikel Arteta previews Slavia Prague` (kanal: Arsenal) →
`cikti\Arsenal vs Slavia Prague - Mikel Arteta (2026-11-03)\` ve içindeki
`... .txt / .srt / .json`, `ozet.md`. Bulunamazsa video başlığı kullanılır.
Teknik direktörün adı bulunursa metinde "Teknik direktör" yerine adı yazılır.

### Soru-cevap metni

Konuşmacı ayırma açıksa `.txt` şöyle olur:

```
Arsenal vs Slavia Prague - Mikel Arteta | basın toplantısı
Tarih: 2026-11-03

Açılış - Mikel Arteta:
...

Soru 1 - Muhabir:
How is Saka?

Cevap - Mikel Arteta:
He trained today...
```

Konuşmacı ayırma yapılamazsa sebebi `.txt`'nin başında, `ozet.md`'de ve işlem sonundaki özette
yazar (ör. "HuggingFace model koşulları kabul edilmemiş"). `stt.bat kontrol` modeli gerçekten
yükleyip dener.

### Hız

- faster-whisper'ın toplu (batched) modu varsayılan olarak açık: 3090'da 16, 3080'de 8 parça
  aynı anda. GPU belleği yetmezse toplu boyut kendiliğinden yarıya iner. Kapatmak için `--batch 0`.
- İki kart aynı anda çalışır, indirmeler GPU'yu beklemeden 3'er 3'er yapılır.

`linkler.txt` örneği:

```
# GW11 basın toplantıları
https://www.youtube.com/watch?v=AAAA
https://www.instagram.com/reel/BBBB/
https://www.arsenal.com/news/...
D:\videolar\ekstra.mp4
```

İzleme modunda yarım dosyalar (`.part`, `.crdownload`, `.tmp`) beklenir. Dosya boyutu iki
taramada değişmezse işlenir. Klasöre bırakılan `.txt` dosyası link listesi olarak işlenir.

---

## Çıktılar

Her kayıt için `cikti\<başlık>\` klasörü oluşur (yer `--out` ya da `stt.toml` ile değişir):

| Dosya | İçerik |
|---|---|
| `<başlık>.txt` | Düz metin. Konuşmacı ayrıldıysa `Teknik direktör: …` / `Muhabir: …` paragrafları |
| `<başlık>.srt` | Altyazı: satır başı ≤42 karakter, en çok 2 satır, ≤6 sn. Konuşmacı değişince `[Muhabir]` öneki |
| `<başlık>.json` | Segmentler: `start`, `end`, `speaker`, `speaker_id`, `text`, `words[]` (kelime zamanı + olasılık), ayrıca kaynak bilgisi, dil, model, konuşmacı süreleri |
| `ozet.md` | Kaynak/süre/dil tablosu, konuşmacı payları, muhabir soruları (zamanlı), anahtar kelimeler, teknik direktörden öne çıkan cümleler |

`json` örneği:

```json
{"id": 12, "start": 63.42, "end": 68.1, "speaker": "Teknik direktör", "speaker_id": "SPEAKER_00",
 "text": "He trained today and he is available.",
 "words": [{"start": 63.42, "end": 63.6, "word": " He", "prob": 0.99}, "..."]}
```

---

## Konuşmacı ayırma

1. https://huggingface.co adresinde hesap açın, **Read** yetkili bir token oluşturun.
2. Şu iki sayfada kullanım koşullarını kabul edin (yoksa model inmez):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Token'ı verin. Üç yol var:
   - `setx HF_TOKEN hf_xxx` (önerilen)
   - `stt.toml` içine `hf_token = "hf_xxx"`
   - komut satırında `--hf-token hf_xxx`
4. Çalıştırın: `stt.bat <link> --diarize`. Konuşmacı sayısını biliyorsanız `--speakers 3` ekleyin.

Nasıl çalışır:

- Ayırma işi ortak kuyruğa ayrı bir görev olarak girer. Bir GPU ayırmayı yaparken diğeri
  tanımaya devam eder.
- Her **kelime**, zaman olarak en çok örtüştüğü konuşmacıya atanır. Whisper bir soruyla cevabı
  tek segmentte verirse segment konuşmacı değişim noktasından bölünür. Tek kelimelik sıçramalar
  yumuşatılır.
- En uzun toplam süre konuşan = `Teknik direktör`, diğerleri = `Muhabir`. Etiketleri
  `--td-label` / `--reporter-label` (ya da `stt.toml`) ile değiştirebilirsiniz.
- Ayırma başarısız olursa (token yok, model koşulları kabul edilmemiş…) metin yine yazılır,
  yalnızca konuşmacı bilgisi olmaz.

---

## FPL entegrasyonu

```bat
setx FPL_REPO D:\kod\fpl
stt.bat "https://www.youtube.com/watch?v=XXXX" --fpl 11 ARS
```

- Metin `D:\kod\fpl\inputs\pressers\GW11\ARS.txt` dosyasına yazılır. GW iki haneye tamamlanır
  (`--fpl 5 LIV` → `GW05\LIV.txt`).
- Ardından repo klasöründe `python3 presser.py inputs/pressers/GW11/ARS.txt` çalışır.
  Windows'ta `python3` yoksa ya da Microsoft Store kısayoluysa sırayla `py -3` ve stt'nin kendi
  Python'u denenir. Belirli bir komut için: `--fpl-python "D:\kod\fpl\.venv\Scripts\python.exe"`
  ya da `stt.toml` içinde `fpl_python`.
- `presser.py` çıktısı ekrana ve `cikti\<başlık>\fpl_presser.log` dosyasına yazılır.
- Takım kodu biliniyorsa (ARS, LIV, MCI…) Whisper'a `Arsenal press conference, Premier League.`
  gibi bir başlangıç istemi verilir; bu, isimlerin doğru yazılmasına yardım eder.
  Kendi isteminizi `--prompt "Arteta, Saka, Ødegaard, Rice"` ile verebilirsiniz.
- Aynı video daha önce işlendiyse yeniden tanıma yapılmaz; yalnızca FPL adımı çalışır.
  Önceki `presser.py` hata verdiyse bu da kapsanır.
- `--fpl` tek bir girdiyle kullanılır.

---

## Atlama ve kaldığı yerden devam

- **Dosyalar** içerik hash'i (SHA-256) ile tanınır. Aynı video başka bir adla gelse de atlanır.
- **Linkler** normalize edilir: `youtu.be/X`, `youtube.com/watch?v=X&si=…`, `shorts/X` hepsi
  aynı sayılır, takip parametreleri atılır. İndirmeden sonra `site:video-id` ile ikinci bir
  kontrol yapılır.
- Kayıtlar `cikti\.stt\durum.sqlite` dosyasında tutulur. `stt.bat durum` hepsini listeler.
  `--force` ile yeniden işlenir.
- **Yarıda kalırsa** (Ctrl+C, elektrik, çökme): hazırlanmış WAV ve biten her parça
  `cikti\.stt\is\<hash>\` altında durur. Aynı komutu tekrar çalıştırınca yalnızca eksik parçalar
  işlenir. yt-dlp yarım indirmeleri de devam ettirir. İş bitince bu ara klasör silinir
  (`--keep-audio` ile kalır).
- Bir işçi çökerse (ör. CUDA hatası) üzerindeki görev diğer GPU'ya geri verilir.

---

## İki GPU nasıl kullanılıyor

```
            hazırlık iş parçacıkları (indir → wav → VAD ile böl)
                            │
                            ▼
                 ortak görev kuyruğu (parçalar + ayırma görevleri)
                    │                        │
          ┌─────────┴─────────┐    ┌─────────┴─────────┐
          │ işçi cuda:0 (3090)│    │ işçi cuda:1 (3080)│   ← her biri kendi süreci,
          │ large-v3 float16  │    │ large-v3 float16  │     kendi modeli
          └─────────┬─────────┘    └─────────┬─────────┘
                    └──────────┬─────────────┘
                               ▼
                 birleştir → konuşmacı ata → txt/srt/json/ozet.md → FPL
```

- Görevler boşta işçi oldukça dağıtılır. Hızlı olan 3090 doğal olarak daha çok parça alır.
- İndirme ve dönüştürme GPU'ları beklemeden sırada yapılır; bir sonraki video, önceki tanınırken iner.
- Hata veren bir işin kuyrukta bekleyen parçaları iptal edilir.
- `large-v3` float16 yaklaşık 3–4 GB VRAM kullanır. pyannote ile birlikte 3080'in 10 GB'ı yeter.
- GPU seçimi: varsayılan `auto` = monitörün bağlı olmadığı kart(lar) (bilgisayar, YouTube ve oyunlar akıcı kalır; tek kart varsa düşük yükle o kart). `--gpus all` = bütün kartlar (en hızlı), `--gpus 0,1`, `--gpus 0`, `--gpus cpu`. Tanıma süreçleri düşük öncelikte çalışır.
  Kart numaraları `nvidia-smi` ile aynıdır (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Hangi kartın kullanıldığı ve
  hangisinin boş bırakıldığı günlüğün başında yazar.
- Takılan bir kart (görev çok uzun sürerse) ya da çöken bir işçi otomatik yeniden başlatılır; görev tekrar denenir.

---

## OpenAI yedek motoru

```bat
setx OPENAI_API_KEY sk-...
stt.bat "<link>" --engine openai                         :: varsayılan model: gpt-transcribe
stt.bat "<link>" --engine openai --openai-model whisper-1
```

- API anahtarı yalnızca `OPENAI_API_KEY` ortam değişkeninden okunur.
- Ses VAD ile ~30 saniyelik parçalara bölünür ve 4 parça paralel gönderilir
  (`stt.toml` → `openai_workers`). Önce kelime zamanlı `verbose_json` istenir. Model bunu
  desteklemiyorsa düz metne düşülür; zaman damgaları o zaman parça sınırlarından gelir.
  `whisper-1` için ≤10 dakikalık parçalar kullanılır (25 MB API sınırı).
- `--diarize` bu motorla da çalışır: pyannote yerelde, GPU varsa GPU'da koşar.

---

## Kelime hata oranı (WER) ölçümü

```bat
:: 3 İngilizce örnek kaydı (LibriSpeech, referanslı) indirip ölçer, hedef < %8
stt.bat olc --ornek

:: kendi kayıtlarınızla (önerilen): klasöre ses + aynı adlı .txt referans koyun
::   D:\olcum\arteta_gw10.mp3   D:\olcum\arteta_gw10.txt
::   D:\olcum\slot_gw10.mp4     D:\olcum\slot_gw10.txt
::   D:\olcum\amorim_gw10.m4a   D:\olcum\amorim_gw10.txt
stt.bat olc D:\olcum --lang en --hedef 8
```

- Ölçüm normal akışla aynı ayarları kullanır: iki GPU, large-v3, beam 5, VAD.
- Metinler Whisper'ın İngilizce normalleştiricisiyle karşılaştırılır (büyük/küçük harf,
  noktalama, `colour/color`, sayılar). Türkçe ve Azerice için temel normalleştirici kullanılır.
- Sonuç ekrana, `olcum.md` ve `olcum.json` dosyalarına yazılır. Her dosya için S/D/I
  (yer değiştirme/silme/ekleme) sayıları ve hatalı cümlelerde REF/HYP karşılaştırması verilir.
- Hedefin altındaysa çıkış kodu 0, üstündeyse 1 olur.
- **Basın toplantısı için gerçekçi ölçüm:** LibriSpeech temiz, okunmuş konuşmadır; sonuç iyimser
  çıkar. 3 gerçek basın toplantısından 2–3 dakikalık kesitleri elle yazıya dökün
  (ya da resmî transkript kullanın) ve `stt olc D:\olcum` ile ölçün.

Referans dosyası yerine klasörde `trans.txt` (`dosya_adı<TAB>metin` satırları) da olabilir.

---

## Tüm seçenekler

| Seçenek | Açıklama | Varsayılan |
|---|---|---|
| `--lang en\|tr\|az\|auto` | Dil | otomatik |
| `--engine whisper\|openai` | Motor | whisper |
| `--model` | faster-whisper modeli ya da yerel klasör | large-v3 |
| `--compute-type` | float16 / int8_float16 / int8 | float16 |
| `--beam-size` | Işın genişliği | 5 |
| `--batch N` | Toplu boyut (0 = kapalı) | karta göre (16 / 8) |
| `--gpus` | `auto` (monitörsüz kart), `all`, `0,1`, `0`, `cpu` | auto |
| `--prompt` | Başlangıç istemi (isimler/terimler) | FPL takımına göre |
| `--no-context` | Önceki metne koşullamayı kapat (tekrar döngüsü olursa) | kapalı |
| `--split-min` / `--chunk-min` | Bölme eşiği / hedef parça (dk) | 8 / 5 |
| `--diarize` | Konuşmacı ayırma | kapalı |
| `--hf-token`, `--speakers N` | HF token, sabit konuşmacı sayısı | |
| `--td-label`, `--reporter-label` | Etiketler | Teknik direktör / Muhabir |
| `--cookies-browser` | `chrome`, `firefox`, `edge`, `none` | chrome |
| `--cookies cookies.txt` | Netscape biçiminde çerez dosyası | |
| `-o, --out` | Çıktı klasörü | `cikti` |
| `--force` | İşlenmiş olsa da yeniden işle | |
| `--keep-audio` | Ara WAV/parçaları silme | |
| `--fpl GW TAKIM`, `--fpl-repo`, `--fpl-python` | FPL entegrasyonu | |
| `--openai-model` | OpenAI modeli | gpt-transcribe |
| `-v` | Ayrıntılı günlük | |

Alt komutlar: `izle KLASOR [--aralik 5]`, `kontrol [--derin]`, `olc [KLASOR] [--ornek] [--hedef 8]`, `durum`.

Ayarların öncelik sırası: varsayılan < `stt.toml` < ortam değişkenleri (`HF_TOKEN`, `FPL_REPO`,
`STT_OUT`, `STT_GPUS`, `STT_MODEL`, `STT_COOKIES_BROWSER`) < komut satırı.

---

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| Çeviri sırasında bilgisayar donuyor, YouTube/tarayıcı kapanıyor | Monitörün bağlı olduğu ekran kartı tam yükte kalıyordu. Artık varsayılan olarak o kart boş bırakılır (Ayarlar › Ekran kartı kullanımı › *Önerilen*). *Hepsi* seçiliyse *Önerilen*'e alın. |
| Uygulama penceresi donuyor | 1.2'de düzeltildi: pencere takılsa da iş arka planda sürer. Günlük: `%LOCALAPPDATA%\stt\stt_app.log`. |
| `Could not copy Chrome cookie database` / DPAPI hatası | Windows'ta Chrome açıkken ve yeni sürümlerdeki uygulamaya bağlı şifreleme yüzünden çerezler okunamayabilir. Chrome'u tamamen kapatın, ya da `--cookies-browser firefox` / `edge` deneyin, ya da bir uzantıyla `cookies.txt` dışa aktarıp `--cookies cookies.txt` verin. |
| YouTube: `Sign in to confirm you're not a bot` | Çerezle tekrar denenir. Olmazsa yt-dlp'yi güncelleyin: `.venv\Scripts\python -m pip install -U "yt-dlp[default]"`. Deno kurulu mu bakın (`stt kontrol`). |
| YouTube: `n challenge` / JS uyarısı | Kurulumu (STT-Kurulum.exe) tekrar çalıştırın: deno'yu `tools\deno` klasörüne indirir. |
| `cublas64_12.dll` / `cudnn_ops64_9.dll not found` | `install.bat`'ı yeniden çalıştırın (torch cu128 DLL'leri getirir). `stt kontrol --derin` ile doğrulayın. |
| `CUDA out of memory` (3080) | `--compute-type int8_float16` ya da `--gpus 0`. |
| Aynı cümle tekrar tekrar yazılıyor | `--no-context`. |
| Azerice Türkçe sanılıyor | `--lang az`. |
| DRM korumalı | Bilerek atlanır (Widevine/FairPlay/PlayReady). Bu içerikler indirilemez. |
| `presser.py` bulunamadı | `--fpl-repo` / `FPL_REPO` doğru klasörü mü gösteriyor? |

---

## Proje yapısı

```
stt.bat / install.bat        Windows girişleri
stt.toml.example             ayar örneği
stt/cli.py                   komut satırı, alt komutlar
stt/sources.py               girdileri genişletme, link normalizasyonu, hash
stt/fetch.py                 yt-dlp (çerez denemeleri), m3u8/mp4 yedeği, DRM tespiti, ffmpeg → 16 kHz wav
stt/audio.py                 wav okuma, VAD ile sessizlikten bölme
stt/engines.py               faster-whisper, OpenAI, pyannote
stt/workers.py               GPU başına süreç, ortak kuyruk protokolü
stt/pipeline.py              orkestrasyon, sürdürme, çökme kurtarma, çıktı
stt/merge.py                 parça birleştirme, kelime düzeyinde konuşmacı atama
stt/outputs.py               txt / srt / json / ozet.md
stt/fpl.py                   FPL dosyası + presser.py
stt/watch.py                 izleme klasörü
stt/doctor.py                kurulum kontrolü
stt/bench.py                 WER ölçümü (jiwer)
tests/                       birim testleri (pytest)
```

Testler: `.venv\Scripts\python -m pytest -q tests`
