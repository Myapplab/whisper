# STT — geliştirme ayrıntıları (CLAUDE.md'den taşındı)

Her oturumda otomatik yüklenmesin diye burada. Gerektiğinde yalnızca ilgili başlığı oku
(`grep -n "^- \*\*" docs/GELISTIRME.md` ile başlıkları gör).

## Ne bu?
Basın toplantısı videolarını (YouTube, X, Instagram, kulüp siteleri, yerel dosyalar) metne çeviren
Windows aracı. Amaç: **Fantasy Premier League modeli için veri seti** (takım › maç › basın toplantısı).

- Motor: faster-whisper large-v3 (float16, beam 5, vad_filter, word_timestamps, toplu/batched mod),
  GPU başına bir işçi süreci (ortak kuyruk), uzun dosyalar Silero VAD ile sessizlikten bölünür.
- **GPU politikası (1.2):** `gpus="auto"` = monitörün bağlı OLMADIĞI kart(lar) (`nvidia-smi display_active`);
  hepsi monitörlüyse / tek kartsa en büyük tek kart, toplu boyut `display_batch` (4) ile sınırlı. `gpus="all"` =
  hepsi. İşçiler ve ffmpeg BELOW_NORMAL öncelikte. Sebep: kullanıcıda ekran kartı tam yükte kalınca bilgisayar
  donuyor, YouTube/tarayıcı çöküyordu. Takılan görev (`_task_timeout`) ya da çöken işçi yeniden başlatılır.
- **1.6.0 ekran kartı koruması:** monitörlü kart `engines.gpu_info` = nvidia-smi `display_attached/display_mode/
  display_active` + Windows `EnumDisplayDevices` (`mark_display`; bazı sürücüler monitörlü kartta da "Disabled"
  diyordu → 3080 tam yük, ekran kararıp geliyordu = TDR). Her işçi `workers.isolate_gpu` ile yalnızca kendi kartını
  görür (`CUDA_VISIBLE_DEVICES`, süreç içinde `cuda:0`; günlükte gerçek ad `label`). Ana süreç CUDA başlatmaz
  (`cuda_device_count` nvidia-smi). `Pipeline._diar_device` monitörlü karta asla kendiliğinden gitmez. WebView2
  `--disable-gpu` (`ui_gpu=false`). gpu_info 60 sn önbellekli.
- **1.9.9 "yardımcı kart" (`gpus = "yardim"`, Ayarlar > Ekran kartı kullanımı):** kullanıcı 100+ videoluk kuyrukta
  3090'ın konuşmacı ayırma sırasında metin çevirisini beklettiğini gördü, 3080'in de "YouTube'a yer bırakarak"
  çalışmasını istedi. Ana kartlar = `auto`; kalanlar (`pipeline.helper_devices`) ayrı havuz `Pipeline.hpool` (ayrı
  kuyruk, model yükleme kilidi ana havuzla ortak). `_helper_dispatch`: yardımcıya görev yalnız bekleyen iş, ana
  işçilerin alabileceğinden fazlaysa (3090 konuşmacı ayırırken ana işçi sayılmaz: asıl kazanç bu), tek tek, her
  görevden sonra `helper_rest_s` (2 sn) dinlenme, sonra `engines.gpu_load` (nvidia-smi, arka plan iş parçacığı)
  ölçümü (1.9.11: yüzde ölçümü ve dinlenme kaldırıldı, aşağıya bakın): kart başka programla %`helper_busy_pct` (50) üstü meşgulse ya da `helper_min_free_gb` (1.5) altı boş bellek
  varsa 30 sn bekler; 2 dk sürerse yardımcı süreç kapatılır (`_h_parked`, model belleği boşalır), kart boşalınca
  (+5 GB) yeniden açılır. Yardımcı baştan KAPALI açılır (model ilk iş fazlasında yüklenir), 5 dk iş fazlası olmazsa
  yine kapanır; görev sırasında bellek yetmezse (oyun) parça deneme sayılmadan sıraya döner, yardımcı kapanır;
  model yüklenirken bellek yetmezse yeniden açma döngüsü yerine kapalı bekler. Yardımcıda toplu boyut ≤
  `display_batch` (4), süreç model yüklendikten sonra IDLE önceliğinde (ortak yükleme kilidinde 3090'ı bekletmesin).
  Yalnız TEK yardımcı kart (en büyüğü). **Kuyruk kilidi hatası (eskiden de vardı):** boştaki işçi `task_q.get()`
  içinde kilidi tutarak bekler; `terminate()` edilirse kilit hiç bırakılmaz, yeniden açılan süreç bir daha görev
  alamaz (iş sonsuza dek bekler). Konuşmacı ayırma süreci bellek / CUDA hatasından sonra tam böyle yeniden açılıyordu.
  Artık `WorkerPool.stop/respawn(idle=True)` tek işçili havuzda kapanma işareti (None) gönderir
  (`test_real_queue_idle_worker_stop_keeps_queue_usable`, gerçek süreçlerle). Konuşmacı ayırma
  yardımcıya gitmez (`_diar_device` yalnız ana kartlar; 1.9.13'ten beri yardımcıda İKİNCİ ayırma işçisi var, aşağı). Yardımcı kalıcı düşerse kuyruğundaki görev ana karta döner
  (`_helper_gone`). Arayüz: kart şeridinde "yardımcı" etiketi, "bekliyor · sebep". Windows'ta / gerçek kartlarda DENENMEDİ.
- **1.9.11 yardımcı kart = "ayrılan bellek dışında tam güç":** kullanıcı istedi: 3080'de birkaç YouTube + ~10 sekme
  için bellek ayrılsın, gerisi işe verilsin. `helper_reserve_gb` (Ayarlar'da, varsayılan 2 GB, hemen geçerli).
  Kapalı yardımcı, boş bellek − ayrılan ≥ `HELPER_FOOT_GB[b]` olan en büyük toplu boyutla açılır (8: 6.4, 4: 5.6,
  2: 5.2 GB; faster-whisper README large-v2 fp16 beam 5: tekli 4.5, toplu 8 6.1 GB + pay); sığmazsa bekler. Açıkken
  boş bellek ayrılanın yarısının altına inerse bekler, 60 sn sürerse kapanır; kart boşalınca yine sığan en büyük
  boyutla açılır (başka programın yüzünden küçülmez: inceleme ajanı bu "kalıcı küçülme" hatasını buldu). Kendi payı
  ilk parçadan sonra ölçülür (`_helper_measure`: açılıştaki boş − şimdiki boş); tahmini 0.5 GB'tan çok aşarsa o boyutun
  belleği öğrenilir (`_h_foot`), sonraki açılış ona göre (aç-kapa döngüsü yok). Görevde / yüklemede bellek yetmezse
  bir küçük boyut (`_helper_shrink`). Küçültme ve öğrenilen değerler 30 dk sonra varsayılana döner. Kullanım yüzdesi ölçümü ve dinlenme
  kaldırıldı (`helper_rest_s` = 0; kendi yükümüz ölçüme karışıyordu). nvidia-smi açık yardımcıda en çok 15 sn'de bir.
  GPU hesap gücü GeForce'ta bölünemez (Windows zaman paylaşımı); YouTube videoyu ayrı çözücü birimde (NVDEC) açar.
- **1.9.13 yardımcı kartta konuşmacı ayırma:** kullanıcının ekranı: metinler bitmiş (3/3), videolar ayırma bekliyor →
  darboğaz 3090'daki tek ayırma süreci. Yardımcı kartta ikinci ayırma işçisi `hdpool` (wid `diar2` = `HELPER_DIAR`),
  ana "diar"ın alamadığı ayırma işi varsa (`_helper_diar_dispatch`). Kart aynı anda TEK iş tutar (metin + ayırma
  belleğe sığmaz; aynı kartta torch + CTranslate2 da sırayla çalışmak zorunda): `_w_on` / `_d_on`; öbür işin ihtiyacı
  `HELPER_SWITCH_S` (20 sn) sürdükçe ve bu iş o kadar işsiz kaldıkça geçilir (aç-kapa döngüsü olmasın). Açılış:
  boş − ayrılan ≥ `HELPER_DIAR_GB` (3 GB, tahmin); torch bellek sınırı o boş yere göre. diar2'de bellek yetmezse
  görev deneme sayılmadan `dbacklog` başına döner, işçi kapanır; CUDA hatasında yalnız o süreç yeniden açılır
  (`_respawn_diar`); "Durdur" hangi ayırma süreci o işi yapıyorsa onu keser; kalıcı düşerse kuyruğu `dbacklog`'a döner.
  Windows'ta / gerçek kartlarda DENENMEDİ.
- **1.9.14:** (1) Oynatma listesi (`channel.is_playlist_url`: YouTube `list=` (Mix RD.. hariç), Dailymotion /playlist/,
  Vimeo showcase/album) = kullanıcının seçtiği videolar: konu süzgeci yok, işlenmemişlerin hepsi seçili gelir; kanal
  linkinde süzgeç kalır. "basin" konusuna "manager's preview" eklendi (Man City adlandırması). (2) **Ekran kararması**
  (kullanıcıda 2 kez, "HDMI sinyali yok"): `stt/gpuguard.py` Windows olay günlüğünden nvlddmkm / Display 4101 /
  LiveKernelEvent 141-117 / Kernel-Power 41 / 6008 okur (PowerShell Get-WinEvent); Kontrol'de "Ekran sürücüsü" satırı.
  Yardımcı kart ilk açıldığında `.stt/yardimci_acik.json`; bundan sonra çökme varsa (açılışta + yardımcı moddayken
  10 dk'da bir) yardımcı kapatılır, `gpus=auto`, arayüzde uyarı (`Pipeline.disable_helper`). Şüpheliler: monitörlü
  karta yük, MSI Afterburner hız aşırtması (kullanıcıda 3090 çekirdek +143, bellek +1441), güç kaynağı. Windows'ta
  DENENMEDİ (PowerShell sorgusu dahil). (3) Ayırma toplu boyutu karta göre (`engines.diar_batch_size`: 24 GB 32,
  10-12 GB 16; `diar_batch` = 0 otomatik); pyannote parçaları ayrı hesaplar, sonuç değişmez. Hız kazancı ölçülmedi.
  (4) **Kanalda basın toplantısı bulucu (deneme, varsayılan kapalı):** `channel.find_pressers` kanal içi YouTube
  araması (`/@kanal/search?query=`; yt-dlp 2026.10.26 testinde var) + adında basın toplantısı geçen oynatma
  listeleri; Kanal · liste'de "Kanalın tamamında basın toplantılarını ara (deneme)" kutusu.
- **1.9.14 indirme / kuyruk:** `fetch._QuietLogger` ek argümanları kabul eder (yt-dlp çerez okuyucu
  `warning(msg, only_once=True)`; eskiden TypeError ile bütün tarayıcı çerezi okuması düşüyordu). yt-dlp'ye ve
  curl_cffi'ye User-Agent verilmez (taklidin kendi kimliği); ffmpeg / requests yolları `user_agent()` (yt-dlp'nin
  güncel listesi). `explain_error` sırası: bölge > ... > giriş > ... > Cloudflare > taklit eksik > "unable to
  extract" > çerez okunamadı (dar) ...; siteye özgü: ok.ru "unable to extract player", Rutube "options JSON 404",
  VK "available to followers". `ytdlp_download` `playlist_items: "1"`. `site_url_variants` + `_ytdlp_to_wav`:
  biçimler sırayla, ağ hatasında durur, en açıklayıcı hata ("unable to extract" olmayan) gösterilir. `find_embeds`:
  mail.ru gömme / meta, bilinen oynatıcı olan og:video yt-dlp'ye. `Pipeline._wait_prep_turn`: en çok `PREP_AHEAD`
  (6) iş hazırlanmış / metni bitmemiş olabilir (metni bitip ayırma bekleyen sayılmaz). İlerleme olayında
  `diar_wait` / `started`; arayüz işçi "busy" olayından işin kartını günceller.
- **Konuşmacı ayırma ayrı süreçte** (`Pipeline.dpool`, wid `diar`, ilk ayırma görevinde açılır, belleği en büyük
  kartta): aynı süreçte torch + CTranslate2 → `invalid device ordinal` ve 1.2.0'da `CUDNN_STATUS_EXECUTION_FAILED`
  + `MemoryError` görüldü. `Diarizer.run`: cuDNN hatasında cuDNN kapalı tekrar dener; işlemciye geçmez (1.6.2, `diar_cpu`).
- **Bellek:** `util.memory_status()` (Windows GlobalMemoryStatusEx: RAM + sanal bellek/commit). Kontrol'de "Bellek"
  satırı; RAM hatalarında `pipeline.ram_hint()` anlık durumu ve sanal bellek (pagefile) çözümünü yazar.
- **Dil (1.2.1):** otomatik algılamada `WhisperEngine.detect_language` 3 pencerenin olasılıklarını toplar ve
  izinli diller (`DEFAULT_LANGS` / `languages`) arasından seçer: large-v3 İngiliz aksanını Galce (cy) sanıp
  bütün metni Galce yazıyordu. Dili yanlış ya da konuşmacısız eski çıktılar link tekrar eklenince yeniden işlenir
  (`_redo_reason`). Aynı cümlenin tekrar döngüsü `merge.collapse_repeats` ile teke iner.
- **Konuşmacı biçimi (1.5.0):** `speaker_mode` = `auto` | `basin` (en uzun konuşan TD, diğerleri muhabir,
  soru-cevap) | `herkes` (podcast/film: `Konuşmacı 1..N` ilk konuşma sırasına göre, `[dk] Konuşmacı N:` diyalog).
  `auto`: `match.looks_like_presser` (başlık/açıklamada press conference / basın toplantısı / mətbuat konfransı…
  ya da `MANAGER_HINTS`'te bilinen TD; `find_manager` "Golden State" gibi her büyük harfli diziyi aday sayar, ona
  güvenme; 1.9.8'den beri kulüp kanalı + sözlükte rakip de evet, özet/antrenman/kadın takımı hariç).
  `herkes` çıktıları `<out>/Podcast ve Film/<başlık>/`, FPL/mac.json yok, index `league="Podcast ve Film"`.
  GUI'de "Konuşmacıları ayır" altında tür seçimi + "Kişi sayısı" (pyannote min=max). Eski (biçim kaydı olmayan)
  soru-cevap çıktısı podcast'e benziyorsa link tekrar eklenince yeniden işlenir (`_redo_reason`). Kullanıcı
  YouTube podcast'inde "sesleri ayırmıyor" demişti: sebep herkesin TD/Muhabir diye iki etikete inmesiydi.
- **NVIDIA Nemotron (2026-11 değerlendirmesi):** Nemotron 3.5 ASR (0.6B, akış/canlı odaklı, 40 dil-yerel, NeMo)
  Whisper large-v3'ün yerine geçmedi: bizim iş toplu ve doğruluk odaklı, AZ desteği belirsiz. İleride denenecekse
  ayrı motor olarak (`--engine`) ekle ve `stt olc` ile WER karşılaştır; Sortformer/multitalker konuşmacı ayırma da aday.
- Windows/pythonw: işçilerde stdout/stderr UTF-8 olmalı (`PYTHONUTF8=1`); cp1252 'ş' yazamayıp pyannote'u
  UnicodeEncodeError ile düşürüyordu.
- İsteğe bağlı konuşmacı ayırma: pyannote/speaker-diarization-3.1 (HF token) → en uzun konuşan teknik
  direktör, diğerleri muhabir → `.txt` soru-cevap biçiminde.
- Çıktı: `.txt` (soru-cevap), `.srt`, `.json` (kelime zamanı + konuşmacı), `ozet.md`, `mac.json`,
  `index.csv` / `index.jsonl`.
- Veri seti düzeni: `<out>/Premier League/<Takım>/<maç tarihi> GWxx <Ev> vs <Dep>/<tarih> mac-oncesi - <TD>.txt`.
  Maç + GW, FPL API fikstürlerinden (fantasy.premierleague.com/api/bootstrap-static, /fixtures) eşleşir
  (`stt/dataset.py`), önbellek `<out>/.stt/fpl_cache.json`.
- Takım/TD adı başlık + YouTube kanal adından (`stt/match.py`; Türkçe/Azerice kalıplar dahil).
- **Eşleştirme (1.9.8, `docs/TARAMA_2026-11-06.md` A):** ortak sözcük olan takım adları (derby, nice, Sporting,
  sabah, om) yalnız niteleyiciyle ya da başka takıma bağlıyken; hoca → takım (`match.PL_MANAGERS`, yalnız
  `MANAGER_SEASON` tarihli videoda ve tarafsız kanalda); maç öncesi/sonrası başlıktan (`dataset.phase_hint`, saat
  bilinmiyorsa `presser_time` "yalnız gün"); sözlükte olmayan rakipte rakipsiz ±3 gün yalnız maçın rakibi o adla
  örtüşürse (`_same_club`; Wycombe = kupa maçı), sözlükteki PL dışı rakip PL maçına yazılmaz; açıklamadaki kupa PL'yi
  ezmez. Kulüp kanalı kuralı yalnız kulübün kendi kanalında (`match.club_of_channel`). Eski çıktılar "Veri seti > Yeniden sınıflandır" / `stt
  yeniden-siniflandir` (`stt/reclassify.py`) ile metin yeniden çevrilmeden taşınır; mac.json skoru da yenilenir.
- FPL: `presser.py` (kullanıcının FPL reposu) → `inputs/pressers/GWxx/TAKIM.txt` yazılır ve
  `python3 presser.py <dosya>` çalışır; PL maçı bulunursa GW/takım kodu otomatik.
- Atlama: dosya SHA-256 / normalize link / `extractor:id`; sürdürme: biten parçalar `<out>/.stt/is/<key>/`.
- Yedek motor: `--engine openai` (varsayılan model `gpt-transcribe`, OPENAI_API_KEY).
- WER: `stt olc [klasör] [--ornek]` (jiwer + whisper normalizer, hedef < %8).

- **1.6.0 sağlamlık:** kalıcı kuyruk (`State.queue_*`; `Pipeline.submit` ekler, `emit` done/failed/skipped siler;
  GUI açılışta "Devam et"). İndirme geçici ağ hatasında tekrar (`is_transient`, `download_retries`). İşçi yeniden
  başlatma geri çekilmeli (`_respawn_due`, `worker_restarts`=6, başarılı görev sayacı sıfırlar); RAM/sürücü
  kaynaklı `fatal` da yeniden denenir. `_dispatch` boşta sayımında yalnız ana havuz (eskiden "diar" de sayılıyordu).
  `_maybe_finalize` kilitli tek giriş (canlı + normal).
- **Ses:** `audio.agc` (en çok +20 dB, gürültü kapısı) tanımaya ve ayırmaya; VAD eşiği 0.4. Ayrılamazsa
  `llm.label_qa` ile metinden soru-cevap (`speaker_source`).
- **İndirme zinciri (`fetch.download_url`):** yt-dlp (düz → `impersonate=chrome` → tarayıcı çerezleri
  `cookie_browsers("auto")` Firefox önce, `browser_cookie_file` önbellek → cookies.txt) → sayfadaki gömülü
  oynatıcılar `find_embeds` (yt-dlp'ye) → sayfadaki m3u8/mp4 → `browser.capture_media` (Playwright + kurulu Edge,
  ağ isteklerini yakalar, reklamı eler, ffprobe ile en uzun). yt-dlp günde bir `start_ytdlp_update` (GUI açılışı,
  yt_dlp içe aktarılmadan önce; `.git` varsa yapılmaz). Canlı yayın `match_filter` ile algılanır → `LiveStream`.
  Üçüncü taraf "downloader" sitelerine link GÖNDERİLMEZ (bkz. docs/ARASTIRMA.md).
- **Kanal / liste (`stt/channel.py`):** `list_videos` (YouTube kanal kökü → /videos + /streams, düz liste),
  `select` (TOPICS çok dilli kalıplar, ek/hariç kelime, isteğe bağlı LLM), `Follows` (`.stt/takip.json`, GUI
  `_follow_loop` 10 dk'da bir, `follow_interval_h`), `add_to_collection` → `Koleksiyonlar/<ad>/liste.csv,
  tum_metinler.txt`. CLI `stt kanal`.
- **Canlı (`stt/live.py`):** `LiveRecorder` ffmpeg s16le → ~`live_chunk_s` sessiz noktadan kesim → parça wav,
  görev `offset` ile (işçi `shift_payload`), backlog'un önüne. Dil ilk ~60 sn oylamayla (`_live_vote`), bitişte
  farklı dildeki parçalar yeniden çevrilir (`:r` görevleri). Metin `Canli Yayinlar/*.txt`; bitince tam çıktı.
  Temiz EOF / `was_live` = yayın bitti (VOD'u baştan kaydetme).
- **Yapay zekâ (`stt/llm.py`):** gemini (REST, model boşsa `pick_gemini` en yeni kararlı flash), claude
  (resmi `anthropic` SDK, varsayılan `claude-haiku-5-5`, effort low; sonnet-5-5'te `fallbacks="default"`), openai.
  `Usage` aylık tahmini $ (`llm_budget_usd`). `write_news` → `- haber.md/.json`. Anahtarlar stt.toml'da ya da
  ortam değişkeninde; **repoya yazma**.
- **Dış programlar:** `stt/api.py` yerel HTTP API (127.0.0.1, `api_port`, `api_token`), `webhook_url`, `stt serve`.

- **1.6.1:** kartta tek iş "Durdur" (`Pipeline.cancel(uid)`, `App.cancel`; indirme `fetch.CANCELLED` +
  yt-dlp `progress_hooks` → `Cancelled`; çalışan ayırma görevi diar süreci yeniden başlatılarak kesilir).
  Bellek sınırları `gpu_limit_pct`=90 / `display_gpu_limit_pct`=70 (yalnız torch/ayırma:
  `set_per_process_memory_fraction`; CTranslate2'de sınır API'si yok, orada toplu boyut), `ram_limit_pct`=70
  (`Diarizer.wait_memory`: boş RAM'in %70'i yetene kadar bekler), `diar_batch`=8 (pyannote toplu boyutu; RAM
  hatasında /4). Ayırmada RAM hatası → diar süreci kapatılır, `ddelayed` ile o işin metin çevirisi bitince en çok
  2 kez tekrar. Film/müzik altında VAD konuşmayı kaçırıyordu (metnin ~%30'u yok): `fill_gaps` — 12 sn'den uzun
  metinsiz aralık VAD'siz tekrar çevrilir, `keep_gap_segment` uydurmaları eler.

- **1.6.2:** konuşmacı ayırma ASLA kendiliğinden işlemciye geçmez (`diar_cpu`=false; kullanıcı: "işlemci fanı çok
  dönüyor, yalnızca GPU"; i3-10100). Eski davranış: bir cuDNN hatasından sonra diar süreci CPU'da kalıyor, sonraki
  bütün videolar CPU'da (görünmeden) ayrılıyordu. Şimdi: cuDNN kapalı tekrar → olmazsa hata + diar süreci yeniden
  başlatılır (sonraki video temiz CUDA). `torch.set_num_threads(2)`. İşçi günlükleri dosyaya gitmez → bitişte ana
  süreç `[diar] konuşmacı ayırma bitti (cihaz, sn)` yazar.

## Arayüzler
- **Uygulama:** `pythonw -m stt.gui` (pywebview + Edge WebView2; `stt/gui.py` + `stt/gui.html`). Pencere
  `window_geometry` (ekranın ~%60'ı, en çok 1100×720, boyut `%LOCALAPPDATA%\stt\pencere.json`'da), dar pencerede
  `#views` sekmeleri (<1250 px veri seti sekmede, <860 px tek sütun). Yeni iş sekmeleri: Linkler / Kanal · liste / Canlı.
  Python ↔ JS: `App` sınıfının `_` ile başlamayan metotları JS'e açıktır (iç alanlar `_` ile başlamalı!).
  Olaylar `Pipeline.events` → `App._push` (yalnızca kuyruğa koyar, ASLA beklemez) → `_sender` iş parçacığı
  toplu olarak `window.__stt.onEvents([...])`. evaluate_js'i başka yerden çağırma: WebView2 takılınca bütün
  motor kilitleniyordu. Üstteki "Durdur" = `cancel_all`, karttaki = `cancel(uid)` (biten parçalar saklanır). Tek kopya kilidi (mutex),
  pencere kapanınca `_shutdown` → `os._exit`.
- **CLI (gelişmiş):** `C:\stt\.venv\Scripts\stt.exe <link|dosya|klasör|liste.txt> [--lang en] [--diarize] [--fpl 11 ARS]`,
  alt komutlar: `kontrol [--derin]`, `olc`, `izle KLASOR`, `durum`. (`stt.bat` / `install.bat` 1.2.0'da kaldırıldı:
  kullanıcı yalnızca setup.exe + masaüstü simgesi kullanır. `BENI_OKU.txt` kurulumun bitiş sayfasında açılır.)
- **Ortam hazırlığı** `stt/util.py: prepare_env()` (GUI + CLI açılışında): `tools\ffmpeg*\bin`, `tools\deno`,
  kayıt defterindeki güncel PATH, winget yolları, `CUDA_VISIBLE_DEVICES` kaldırma, `CUDA_DEVICE_ORDER`, UTF-8.
- **Kurulum:** `installer/stt.nsi` (NSIS) + `installer/kur.ps1`; `installer/build.sh` → `dist/STT-Kurulum.exe`.
  NSIS her adımı ayrı çağırır (`kur.ps1 -Step check|python|venv|torch|libs|tools|config|model`), çıktı
  `nsExec::ExecToLog /OEM` ile okunur → kur.ps1 **UTF-8 BOM'lu** kalmalı ve çıktıyı OEM kod sayfasında yazar.
  Tam günlük `<kurulum>\kurulum.log`. ffmpeg/deno `tools\` klasörüne doğrudan iner (winget yok).
  Görseller: `installer/make_images.py` (Pillow) → welcome.bmp / header.bmp. `ManifestDPIAware true`.
  Eski bat dosyaları (stt.bat/install.bat) kur.ps1 tarafından silinir.
- **Güncelleme (1.4.0):** `stt/updater.py` — açılışta `raw.githubusercontent.com/Myapplab/whisper/main/stt/__init__.py`
  sürümünü okur; yeniyse GUI'de yeşil şerit. `apply()`: codeload zip → geçici klasörde içe aktarma testi →
  yedek (`<kurulum>\guncelleme-yedek`) → `stt/` + kök dosyaları yaz → `pyproject/constraints` değiştiyse pip →
  hata olursa geri yükle → `relaunch()` (yeni süreç tek kopya kilidini 30 sn bekler, `STT_RESTART=1`).
  `.git` olan klasörü güncellemez. **Depo herkese açık olmalı** (kullanıcı onayladı). Kurtarma / sıfırdan kurulum:
  `irm https://raw.githubusercontent.com/Myapplab/whisper/main/installer/guncelle.ps1 | iex` (yalnız ASCII,
  bellekte çalışır → SAC/MOTW yok). **Yeni sürüm yayınlamak = `stt/__init__.py` sürümünü artırıp `main`'e
  push**; kullanıcıya artık zip/exe/PowerShell gönderme, "uygulamada Güncelle'ye basın" de.
- **GUI'de Kontrol düğmesi** `stt kontrol`'ü ayrı süreçte çalıştırıp sonucu gösterir; kullanıcıdan bunun ekran görüntüsünü iste.
  Günlük: `%LOCALAPPDATA%\stt\stt_app.log`.

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

- **1.6.3:** konuşmacı ataması cümle düzeyinde (`merge.assign_speakers`): kelime → tur (örtüşmesiz,
  `exclusive_turns`: örtüşen yerde uzun tur kazanır; 4.x'te `exclusive_speaker_diarization` önce), sonra cümle
  (`. ? ! …` ya da >0.6 sn sessizlik) başına süre ağırlıklı çoğunluk oyu; cümle ortasında yalnızca iki konuşmacı da
  ≥1.5 sn ve ≥4 kelime konuşmuşsa bölünür. Şikâyet: "kelimeler farklı konuşmacılara yazılıyor".

## Sürüm geçmişi (kısa)
- 1.6.0: kullanıcı şikâyetleri: "birkaç videoyu aynı anda yapamıyor", "muhabiri ayıramıyor", "3080 çok kullanılıyor,
  açınca ekranlar gidip geliyor", "pencere çok büyük", "YouTube dışı siteler (Dailymotion, ok.ru...) inmiyor".
  Eklenenler: kanal/liste + konu süzgeci + takip, canlı yayın, yapay zekâ (Gemini şimdi, Claude ileride),
  yerel API/webhook. Windows'ta henüz denenmedi (özellikle EnumDisplayDevices, Playwright+Edge, WebView2 --disable-gpu).
- 1.2.0 (dal optimistic-edison) ve 1.2.1 (dal inspiring-ritchie) ayrı oturumlarda paralel geliştirildi; 1.3.0'da
  `main`'de birleştirildi. **Yeni işi her zaman güncel `main`'den başlat** ki dallar yine ayrışmasın.
- 1.4.0: uygulama içi güncelleme + `installer/guncelle.ps1` (kullanıcı zip indirip PowerShell'le uğraşırken
  takılıyordu; "kökten çöz, bir daha olmasın" dedi).
- 1.3.0: birleştirme + konuşmacı ayırma cuDNN/CPU geri çekilmesi + bellek teşhisi. Kullanıcı 1.2.0'da
  "Soru-cevap ayrılamadı: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED" ve "MemoryError" görmüştü; 1.3.0 Windows'ta
  henüz denenmedi.

## Bilinen sınırlar / yapılacaklar
- Windows'ta henüz doğrulanmadı: GUI (pywebview) ilk açılış, setup.exe + SAC davranışı, pyannote ile
  gerçek konuşmacı ayırma (kullanıcıda "ayırmıyor" şikâyeti vardı; `stt kontrol` artık modeli
  yükleyip dener — sonucunu kullanıcıdan iste), FPL API ile gerçek eşleşme, gerçek WER (large-v3).
- Yerel dosyalarda kanal adı yok → başlıkta yalnızca "Everton vs Coventry City" varsa TD'nin takımı
  bilinemez (Lampard → Everton'a düştü). 1.9.8: TD→takım eşleştirmesi (`PL_MANAGERS`) bunu çözüyor; listede
  olmayan hocada sorun sürer (fikir: GUI'de takımı elle düzeltme).
- Elle tutulan listeler, **her sezon başında** (ve hoca değişince) güncellenmeli: `match.PL_MANAGERS` +
  `MANAGER_SEASON` (tarih aralığını da kaydır), `match.TEAMS`, `dataset.PL_FALLBACK` (FPL'ye ulaşılamazsa),
  `fpl.TEAM_NAMES`. Son güncelleme 2026-11-06 (2026-27; web aramasıyla doğrulandı).
- setup.exe imzasız; paylaşım için kod imzalama (ör. Azure Trusted Signing) gerekir.
- Kullanıcı ileride uygulamayı paylaşmak istiyor (tek kurulum dosyası).

## Geliştirme (bulut/Linux)
- Testler: `python -m pytest -q tests` (104 test; `test_v161.py` = tek iş durdurma/bellek/boşluk; `test_v16.py` = 1.6.0 özellikleri; `test_speakers.py` = podcast/film modu; `test_updater.py` = güncelleme/geri alma; `test_stability.py` = GPU seçimi, işçi yeniden başlatma, iptal,
  bloklamayan arayüz, cuDNN geri çekilmesi; `test_app.py` = ortam hazırlığı, ayarlar, hata mesajları).
- GPU yok ve HuggingFace engelliyse test modeli: GitHub `rhasspy/models` `asr_faster-whisper-small-int8`
  + `dev/make_tokenizer.py` (openai-whisper tiktoken → tokenizer.json) + `dev/add_alignment_heads.py`.
  `--model <klasör> --gpus cpu,cpu` ile uçtan uca çalışır.
- GUI testi: `dev/gui_bridge.py` (pywebview yerine HTTP köprüsü) + Playwright (`dev/gui_drive.js`,
  `dev/gui_screens.js`). Ayrıntı: `dev/README.md`.
- Gizli tarayıcı testi: yerel `python -m http.server` + JS ile yüklenen HLS sayfası (`docs/ARASTIRMA.md`); Linux'ta
  Playwright `/opt/pw-browsers` Chromium'unu bulur (`browser._launch`). Canlı testi: `.m3u8` ya da dosya +
  `live_readrate`.
- Commit/PR mesajlarında model adı yazma.
