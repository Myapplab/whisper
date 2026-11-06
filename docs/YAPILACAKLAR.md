# Yapılacaklar / oturum devri

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## Son durum
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
   "cuBLAS failed" / "invalid device ordinal" ekran kartı takılmasından; tekrar ederse madde 4.
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
