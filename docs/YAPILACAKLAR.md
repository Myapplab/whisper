# Yapılacaklar / oturum devri

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## Son durum
- Sürüm 1.7.0 (`main`): konuşmacı ayırma yeni model (community-1, pyannote.audio 4.0.7; olmazsa 3.1), podcast/filmde
  müzik ayıklama (HDemucs), ses kütüphanesi (bilinen kişiler adıyla), okuyucuda "Konuşmacılar" düzeltme ekranı.
  Ayrıntı: `docs/KONUSMACI_AYIRMA.md` "1.7.0'da yapılanlar". 122 test geçiyor. Windows'ta henüz denenmedi.
- Güncellemede pip çalışır (pyannote 3.4 → 4.0.7). Kullanıcı HF'de `pyannote/speaker-diarization-community-1`
  sayfasında "Agree" demeli; demezse eski model kullanılır.
- Not: yalnız yeni işlenen videolar etkilenir; eski çıktılarda ses imzası yok (düzeltme ekranı çalışır ama öğrenmez).

## Sıradaki işler
0. 1.7.0 Windows'ta doğrulansın: güncelleme sonrası "Kontrol" (pyannote 4.0.7, konuşmacı modeli community-1),
   günlükte "müzik ayıklandı" (film/podcast), "Ses kütüphanesinden tanındı"; okuyucuda "Konuşmacılar" ekranı.
   Güncelleme pip'te hata verirse günlüğü iste (torchcodec / Smart App Control olabilir).
1. Kullanıcıdan 1.6.3 sonrası bir basın toplantısının ekran görüntüsü: konuşmacılar hâlâ karışıyor mu?
2. Konuşmacı ayırma: `docs/KONUSMACI_AYIRMA.md` adım 3 (DER ölçümü; referans = `<out>/.stt/duzeltme/*.json`)
   → adım 6 (ayar optimizasyonu) → gerekirse 7 (ince ayar) / Sortformer karşılaştırması.
3. 1.6.1/1.6.2 Windows'ta doğrulansın: kartta "Durdur", film boşluk doldurma, bellek beklemesi,
   günlükte `[diar] konuşmacı ayırma bitti (cuda:..., N sn)`.
4. Metin parçasında bir kez `cudaErrorInvalidDevice: invalid device ordinal` görüldü (ikinci denemede düzeldi);
   tekrar ederse `workers.isolate_gpu` / `CUDA_DEVICE_ORDER` incelensin.
