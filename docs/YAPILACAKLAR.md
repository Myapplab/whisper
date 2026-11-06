# Yapılacaklar / oturum devri

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## Son durum
- Sürüm 1.8.4: Memphis videosunda ayırma "CUDA error: unknown error" ile düştü, video konuşmacısız bitti. Artık ayırmada
  ekran kartı hatası olunca süreç yeniden açılıp görev 2 kez daha deneniyor (2.'de müzik ayıklamasız);
  `_vocals` CUDA hatasını yutmuyor ("müzik ayıklama sırasında ..." ile hemen bildiriyor → müzik ayıklamasız tekrar).
  3090'da CUDA hataları (invalid device ordinal, cuBLAS failed, unknown error) sık: sürücü / güç / ısı olabilir;
  sürerse NVIDIA sürücüsü (Studio) güncellemesi ve `nvidia-smi -q -d PERFORMANCE` istensin.
- Sürüm 1.8.3 (senaryo denemeleri, `tests/test_sim.py`): ayırma süreci bir daha açılmazsa bekleyen ayırmalar hata
  sayılır (`_drop_dead_diar`; eskiden iş + sıralı çalışma yüzünden bütün kuyruk sonsuza dek bekliyordu); işçi
  mesajındaki beklenmedik hata yalnız o videoyu düşürür (`_safe_msg`; eskiden motor duruyordu); model yüklerken
  30 dk hazır olmayan işçi yeniden açılır (`_loading_since`); ekran kartında metin parçası takılma sınırı
  max(5 dk, 3 x parça) (eskiden ~36 dk); ffmpeg her http girdisinde `-rw_timeout` 30 sn + 2 saat süreç sınırı.
- Sürüm 1.8.2: Windows'ta doğrulandı: `(stt-diar)` günlüğü çalışıyor, community-1 yükleniyor, müzik ayıklama 40 dk
  seste 33 sn, ayırma 36 sn. Yeni sorun: "bellek bekleniyor ... boş RAM 6.5 GB, sanal bellek 1.5 GB" → 20 dk
  bekliyordu. `Diarizer.wait_memory`: RAM yetiyorsa sanal bellek az olsa da 60 sn sonra deniyor; en çok 10 dk.
  Kullanıcıda sayfa dosyası küçük olabilir (commit_free düşük); sorun sürerse sayfa dosyası büyütülsün.
  PowerShell 5'te günlük okurken `-Encoding UTF8` ekle (yoksa Türkçe harfler bozuk görünüyor).
- Sürüm 1.8.1: `invalid device ordinal` konuşmacı ayırma boştayken de çıktı (NBA videosu, parça 0–3 tamam, t:0004 üç
  kez) → eşzamanlılık değil, İÇERİĞE BAĞLI CTranslate2 hatası (büyük olasılıkla kelime zamanı hizalama / toplu çeviri).
  `WhisperEngine._fallback`: toplu olmadan, sonra kelime zamansız tekrar (`engines.odd_cuda_error`); bu hata artık
  işçiyi yeniden açtırmıyor. Yine olmazsa parça atlanıyor (`job.info["skipped_chunks"]`, en çok parçaların 1/4'ü),
  iş bitiyor. Görev hatasının izi (trace) günlüğe yazılıyor: kullanıcıdan "PARÇA ATLANDI" / trace satırları istensin.
- Sürüm 1.8.0: konuşmacı ayırmayı referans videoyla eğitme ("Konuşmacılar" → "Kaydet ve eğit", `stt/tune.py`,
  `stt egit`; ayrıntı `docs/KONUSMACI_AYIRMA.md` adım 3 ve 6). Windows'ta denenmedi: ilk denemede günlükte
  "[eğitim]" ve "Konuşmacı ayarı eğitildi" satırları istensin. 134 test.
- Sürüm 1.7.3: 1.7.2'de metin işçisi yeniden açılsa da t:0000 üç kez `invalid device ordinal` verdi (konuşmacı
  ayırma + müzik ayıklama aynı anda 3090'daydı). Artık aynı kartta sırayla: `Pipeline._shared_gpu/_diar_turn`
  (işin metni bitince ayırma araya girer, o sırada metin görevi verilmez). Kök neden kesin değil; tekrar ederse
  `pip list` (ctranslate2/numpy/torch) ve günlükteki `(stt-diar)` satırları istensin.
- Sürüm 1.7.2 (`main`): metin işçisinde ekran kartı hatası (invalid device ordinal, cuBLAS failed; `workers.gpu_fault`)
  olunca işçi kendini kapatıyor, `_health` temiz açıyor, parça 3 kez deneniyor (eskiden 2. hatada iş başarısızdı).
  Güncelleme denetimi commit SHA ile (GitHub API): raw.githubusercontent dal adresi 5 dk önbellekteydi, yeni sürüm
  hemen görünmüyordu. 128 test.
- Sürüm 1.7.1 (`main`): 1.7.0 Windows'ta denendi; konuşmacı ayırma %0'da takıldı (büyük olasılıkla HF Xet indirmesi
  donması; community-1 "Agree" sonradan kabul edildi). Düzeltme: `util.net_env()` (HF_HUB_DISABLE_XET=1, indirme
  süre sınırları, PYANNOTE_METRICS_ENABLED=false, HF telemetri kapalı); ayırma işçisi `STT_LOG` ile günlüğe yazıyor
  (`(stt-diar)` önekli satırlar: model yükleme/hazır, müzik ayıklama, başlıyor); işçi ilerledikçe `beat` gönderiyor,
  takılma süresi son sinyalden sayılıyor (en az 15 dk; eskiden 3 x video süresi ≈ 2,5 saat). 126 test geçiyor.
- 1.7.0: konuşmacı ayırma community-1 (pyannote.audio 4.0.7; olmazsa 3.1), müzik ayıklama (HDemucs), ses kütüphanesi,
  "Konuşmacılar" düzeltme ekranı. Ayrıntı: `docs/KONUSMACI_AYIRMA.md` "1.7.0'da yapılanlar".
- Not: yalnız yeni işlenen videolar etkilenir; eski çıktılarda ses imzası yok (düzeltme ekranı çalışır ama öğrenmez).

## Sıradaki işler
0. 1.7.1 sonrası: takılan işte günlükte `(stt-diar)` satırlarının son hâli istensin (hangi adımda durdu).
   "cuBLAS failed" / "invalid device ordinal": 1.7.2'de işçi yeniden açılıyor; yine de sık oluyorsa madde 4.
   1.7.0 Windows'ta doğrulansın: güncelleme sonrası "Kontrol" (pyannote 4.0.7, konuşmacı modeli community-1),
   günlükte "müzik ayıklandı" (film/podcast), "Ses kütüphanesinden tanındı"; okuyucuda "Konuşmacılar" ekranı.
   Güncelleme pip'te hata verirse günlüğü iste (torchcodec / Smart App Control olabilir).
1. Kullanıcıdan 1.6.3 sonrası bir basın toplantısının ekran görüntüsü: konuşmacılar hâlâ karışıyor mu?
2. Konuşmacı ayırma: `docs/KONUSMACI_AYIRMA.md` adım 3 (DER ölçümü; referans = `<out>/.stt/duzeltme/*.json`)
   → adım 6 (ayar optimizasyonu) → gerekirse 7 (ince ayar) / Sortformer karşılaştırması.
3. 1.6.1/1.6.2 Windows'ta doğrulansın: kartta "Durdur", film boşluk doldurma, bellek beklemesi,
   günlükte `[diar] konuşmacı ayırma bitti (cuda:..., N sn)`.
4. Metin parçasında bir kez `cudaErrorInvalidDevice: invalid device ordinal` görüldü (ikinci denemede düzeldi);
   tekrar ederse `workers.isolate_gpu` / `CUDA_DEVICE_ORDER` incelensin.
