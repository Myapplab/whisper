# Yapılacaklar / oturum devri

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## Son durum
- Sürüm 1.6.2 (`main`). Kullanıcı Windows'ta güncelledi; konuşmacı ayırma artık yalnız GPU'da.
- Kullanıcı şikâyeti: konuşmacılar kelime kelime karışıyor → plan `docs/KONUSMACI_AYIRMA.md`.

## Sıradaki işler
1. Konuşmacı ayırma: `docs/KONUSMACI_AYIRMA.md` adım 1 (cümle düzeyinde atama) → adım 2 → 3 → 4 → 5.
2. 1.6.1/1.6.2 Windows'ta doğrulansın: kartta "Durdur", film boşluk doldurma, bellek beklemesi,
   günlükte `[diar] konuşmacı ayırma bitti (cuda:..., N sn)`.
3. Metin parçasında bir kez `cudaErrorInvalidDevice: invalid device ordinal` görüldü (ikinci denemede düzeldi);
   tekrar ederse `workers.isolate_gpu` / `CUDA_DEVICE_ORDER` incelensin.
