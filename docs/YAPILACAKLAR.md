# Yapılacaklar / oturum devri

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## Son durum
- Sürüm 1.6.3 (`main`): konuşmacı ataması cümle düzeyinde (plan adım 1) + örtüşen turlar tek konuşmacıya
  indiriliyor (adım 2'nin 3.x yaklaşığı). 109 test geçiyor. Kullanıcı henüz Windows'ta denemedi.
- Not: yalnız yeni işlenen videolar etkilenir; eski çıktılar yeniden işlenmedikçe aynı kalır.

## Sıradaki işler
1. Kullanıcıdan 1.6.3 sonrası bir basın toplantısının ekran görüntüsü: konuşmacılar hâlâ karışıyor mu?
2. Konuşmacı ayırma: `docs/KONUSMACI_AYIRMA.md` adım 2'nin kalanı (pyannote 4.x + community-1, `diar_model`
   ile seçilebilir) → adım 3 (DER ölçümü) → 4 (düzeltme arayüzü) → 5 (ses imzası).
3. 1.6.1/1.6.2 Windows'ta doğrulansın: kartta "Durdur", film boşluk doldurma, bellek beklemesi,
   günlükte `[diar] konuşmacı ayırma bitti (cuda:..., N sn)`.
4. Metin parçasında bir kez `cudaErrorInvalidDevice: invalid device ordinal` görüldü (ikinci denemede düzeldi);
   tekrar ederse `workers.isolate_gpu` / `CUDA_DEVICE_ORDER` incelensin.
