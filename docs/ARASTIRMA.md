# Araştırma notları (1.6.0)

## 1. Neden YouTube iyi, diğer siteler kötü çalışıyordu?

| Sebep | Belirti | 1.6.0 çözümü |
|---|---|---|
| Çerez tarayıcısı varsayılanı **Chrome**'du. Chrome 127+ (ve Edge) Windows'ta çerezleri "uygulamaya bağlı şifreleme" ile saklıyor; yt-dlp çoğu zaman okuyamıyor. | Instagram, Facebook, X (hassas içerik), yaş sınırlı videolar "giriş gerekli" hatası | `cookies_browser = "auto"`: kurulu tarayıcılar sırayla denenir (Firefox önce). Çerezler bir kez okunup önbelleğe alınır (aynı anda iki indirme "database is locked" vermez). İsteğe bağlı `cookies.txt`. |
| yt-dlp kurulumda bir kez güncelleniyordu; siteler (Dailymotion, ok.ru, Instagram...) sık değişiyor. | "Unable to extract", "No video formats found" | yt-dlp **günde bir kez** arka planda güncellenir (uygulama açılışında, indirmelerden önce). |
| Tarayıcı taklidi yalnızca curl_cffi kuruluysa ve yt-dlp istediğinde devreye giriyordu. | Dailymotion / Cloudflare'li siteler 403 | İkinci deneme her zaman `impersonate=chrome` ile yapılır; sayfa indirme de curl_cffi ile. |
| Kulüp siteleri videoyu **gömülü oynatıcıyla** (YouTube, Brightcove, JW Player, Vimeo, ok.ru...) gösteriyor; yt-dlp'nin genel çıkarıcısı her zaman bulamıyor. | "Unsupported URL" | Sayfadaki iframe / Brightcove `data-account`+`data-video-id` bulunur, gömülü adres yt-dlp'ye verilir. |
| Video adresi HTML'de değil, JavaScript ile yükleniyor (çoğu modern site). | "Sayfada m3u8/mp4 linki bulunamadı" | **Gizli tarayıcıyla yakalama**: sayfa Edge ile görünmeden açılır, oynat düğmesine basılır, ağdaki `.m3u8/.mpd/.mp4` istekleri toplanır; reklamlar elenir (en uzun medya seçilir), çerez/Referer başlıklarıyla ffmpeg'e verilir. |

### "X downloader" siteleri neden kullanılmadı?
savefrom, ssyoutube, dmdownloader gibi siteler de aslında yukarıdaki işi (sayfayı açıp asıl video adresini bulmak)
yapar. Kendi bilgisayarımızda yapmak daha sağlam: bu siteler sık kapanır, captcha/reklam pencereleri açar,
indirme adresini kendi sunucularından geçirir (yavaş, gizlilik riski) ve otomasyonu engeller. Gizli tarayıcı
yöntemi aynı sonucu reklamsız ve linkimizi üçüncü bir siteye göndermeden verir.

DRM'li (Widevine / PlayReady) yayınlar (ör. ücretli platformlar) hiçbir yöntemle alınamaz ve alınmamalı; STT
bunları "DRM" diye atlar.

## 2. Canlı yayını doğrudan çevirme

Seçenekler:

| Yöntem | Gecikme | Doğruluk | Not |
|---|---|---|---|
| **Parçalı Whisper (seçilen)**: ffmpeg yayını okur, ~20 sn'de bir sessiz bir noktadan keser, parça GPU'da large-v3 ile çevrilir | 20-30 sn | Toplu işle aynı (large-v3) | Ek bağımlılık yok, aynı GPU işçileri; kopunca yeniden bağlanır; bitince tam çıktı + konuşmacı ayrımı |
| whisper_streaming (ÚFAL, LocalAgreement) / WhisperLive (Collabora) | 3-5 sn | Biraz düşük (kısa bağlam) | Ayrı sunucu, sürekli GPU yükü; ekran kartı korumasıyla çelişir |
| NVIDIA Nemotron / Parakeet akış ASR (NeMo) | <1 sn | İngilizce iyi, AZ/TR belirsiz | Ayrı motor; ileride `--engine` olarak denenebilir |
| Bulut (Gemini Live, Deepgram, AssemblyAI) | <1 sn | İyi | Ücretli, dakika başı; internet kesintisinde durur |

Haber/veri seti işi için birkaç saniyelik gecikme önemli değil, doğruluk önemli; bu yüzden parçalı large-v3
seçildi. Dil, ilk ~60 saniyenin oylamasıyla sabitlenir; bitişte farklı dilde kalan parçalar yeniden çevrilir.

## 3. Muhabir sesi neden ayrılmıyordu?

Basın toplantısında teknik direktör yaka mikrofonunda (-15..-20 dB), muhabirler salonda, el mikrofonuyla ya da
hiç mikrofonsuz (-35..-45 dB) konuşur. Sonuç: Whisper'ın VAD'ı soruları sessizlik sanıp atlıyor, pyannote da
çok az ses gördüğü için muhabirleri ayrı konuşmacı olarak kümeleyemiyordu.

1.6.0: otomatik kazanç (AGC, en çok +20 dB, gürültü kapısı) hem tanımaya hem ayırmaya uygulanır; VAD eşiği
0.5 → 0.4. Ses yine yetmezse (ya da pyannote hata verirse) ve yapay zekâ ayarlıysa soru-cevap **metinden**
ayrılır (çıktıda "konuşmacı kaynağı: yapay zekâ (metinden)" yazar).

## 4. Ekranın kararıp gelmesi

Sebep büyük olasılıkla Windows TDR (ekran sürücüsünün sıfırlanması): monitörün bağlı olduğu karta (3080) ağır
CUDA yükü binince Windows sürücüyü sıfırlıyor. 1.5'te monitör algılama yalnızca `nvidia-smi display_active`'e
bakıyordu; bazı sürücüler monitörlü kartta da "Disabled" diyor → iki kart da "boş" sanılıp 3080 tam yükte
kullanılıyordu. 1.6.0:
- Windows `EnumDisplayDevices` ile masaüstünü çizen kart adla bulunur (+ nvidia-smi `display_attached/display_mode`).
- Her işçi süreci yalnızca kendi kartını görür (`CUDA_VISIBLE_DEVICES`); ana süreç hiç CUDA başlatmaz.
- Konuşmacı ayırma monitörlü karta asla kendiliğinden gitmez.
- Arayüz (WebView2) ekran kartı hızlandırması kapalı (`ui_gpu = false`).
- Ayarlar'da kart tek tek seçilebilir ("Yalnızca RTX 3090").

## 5. Yol haritası (öneriler)

1. **Haber radarı entegrasyonu**: yerel API (`/api/jobs`, `/api/results`) + webhook hazır; radar programı yeni
   linkleri POST eder, biten metinleri webhook'tan alır.
2. **Claude Max'e geçince**: Ayarlar > Yapay zekâ > Claude; varsayılan Haiku 5.5 (çok ucuz), yazım kalitesi için
   `claude-sonnet-5-5`. Toplu ve acil olmayan işler için Message Batches (%50 ucuz) eklenebilir.
3. **Oyuncu adları sözlüğü**: FPL oyuncu listesinden Whisper `initial_prompt`/`hotwords` (Ødegaard, Gyökeres
   gibi adlar daha doğru yazılır).
4. **Sakatlık/kadro çıkarımı → FPL modeli**: haber paketindeki `sakatlik_kadro` alanı yapılandırılmış veri olarak
   `index.jsonl`'a eklenebilir.
5. **Konuşmacı tanıma**: aynı TD'nin ses izi (embedding) saklanıp sonraki toplantılarda adıyla etiketlenebilir.
6. **Kod imzalama** (Azure Trusted Signing) — paylaşım için SmartScreen/SAC uyarılarını kaldırır.
