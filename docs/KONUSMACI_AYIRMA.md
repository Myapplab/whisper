# Konuşmacı ayırma: neden karışıyor, nasıl iyileşir, nasıl "eğitilir"

Kullanıcı şikâyeti (1.6.2): "bazı kelimeler aynı konuşmacıya, bazıları farklı konuşmacılara yazılıyor".
Bu dosya bir sonraki oturumun uygulayacağı plandır. Sırayla yap; her adımdan sonra ölç (adım 3).

## Şu anki akış
1. `engines.Diarizer.run` → pyannote 3.1 → `turns = [[başla, bitir, SPEAKER_xx], ...]` (örtüşen konuşma VAR).
2. `merge.assign_speakers`: her kelimeye, zaman aralığıyla en çok örtüşen turu verir (`_speaker_at`),
   `_smooth(min_run=2)` yalnızca TEK kelimelik sıçramaları düzeltir, sonra segmenti konuşmacı değişiminden böler.

## Sorunun sebepleri (en olasıdan aza)
1. **Kelime zamanları kaba:** Whisper kelime zamanı ±0.2–0.5 sn kayar; konuşmacı değişiminin yanındaki 2–4 kelime
   yanlış tura düşer. `_smooth` 1 kelimeyi düzeltir, 2–3 kelimelik sıçramalar kalır → cümle ortasında konuşmacı
   değişiyor gibi görünür.
2. **Örtüşen turlar:** pyannote aynı anda iki konuşmacı verebilir; "en çok örtüşen" seçimi kelime kelime
   değişebilir.
3. **Kümeleme hatası:** aynı kişi iki kümeye (SPEAKER_01/03) bölünür ya da muhabirler tek kümede birleşir.
   Kişi sayısı biliniyorsa (`Kişi sayısı` alanı) büyük ölçüde düzelir.

## 1.7.0'da yapılanlar (özet)
- **community-1:** `diar_model` varsayılanı `pyannote/speaker-diarization-community-1` (pyannote.audio 4.0.7,
  `pyproject.toml` `[diarize]`; `constraints.txt` `torchcodec==0.7.0` = torch 2.8 uyumlu; ses bellekten verildiği
  için torchcodec yüklenemese de çalışır). Yüklenemezse (HF koşulu kabul edilmemiş, pyannote 3.x, ağ)
  `engines.FALLBACK_DIAR_MODEL` = 3.1'e düşülür; günlükte "Konuşmacı modeli: ..." ve "Kontrol"de uyarı.
  4.x'te `exclusive_speaker_diarization` kullanılır.
- **Müzik ayıklama:** `Diarizer._vocals` = torchaudio `HDEMUCS_HIGH_MUSDB_PLUS` (ek paket yok; ilk seferde
  download.pytorch.org'dan ~300 MB). 16 kHz → 44.1 kHz, 10 sn parça + 1 sn bağlam (`engines.chunk_spans`), yalnız
  "vocals" kaynağı, yalnız ekran kartında; hata olursa özgün ses. `diar_music` = auto (podcast/film, yani
  `speaker_mode`=herkes) | on | off. Yalnızca konuşmacı ayırmaya gider; Whisper özgün sesi kullanır.
- **Ses imzası:** pyannote konuşmacı başına ortalama gömme verir (3.x `return_embeddings=True`, 4.x
  `speaker_embeddings`) → `konusmaci.json` `emb` → `<out>/.stt/ses/<key>.json`. Kütüphane `<out>/.stt/sesler.json`
  (`stt/voices.py`, ad başına en çok 30 imza, model başına ayrı; farklı modellerin imzaları karşılaştırılmaz).
  `_finalize` → `Pipeline._voice_names` → kosinüs ≥ `voice_threshold` (0.6) olan konuşmacıya ad; basında kütüphanede
  "td" işaretli ad bulunursa o konuşmacı TD olur. `voice_match`=false kapatır.
- **Düzeltme arayüzü:** okuyucuda "Konuşmacılar" (`App.speakers_load/speakers_save/voices_list/voice_delete`):
  ad verme (kütüphanedeki adlar önerilir), satır / bölüm taşıma, "+ Yeni kişi", basında TD seçimi. Kaydedince
  txt/srt/json/özet yeniden yazılır, adı konan sesler kütüphaneye eklenir, düzeltilmiş turlar
  `<out>/.stt/duzeltme/<key>.json`'a (DER ölçümü / ince ayar verisi) yazılır.
- **Sortformer yapılmadı:** NVIDIA NeMo gerektirir (çok büyük, Windows'ta resmî destek zayıf, en çok 4 konuşmacı).
  DER ölçümü (adım 3) gelince community-1 ile karşılaştırılmaya değer olursa ayrı süreçte denenmeli.

## Yapılacaklar (öncelik sırası)
1. ✅ **(1.6.3 yapıldı)** **Cümle düzeyinde atama** (`stt/merge.py`, ucuz, en görünür kazanç):
   - Segmenti önce cümlelere böl (`. ? !` ve 0.6 sn'den uzun kelime arası).
   - Her cümleye, kelime sürelerinin toplamıyla **çoğunluk oyu** ver.
   - Cümle ortasında ancak öbür konuşmacı ≥1.5 sn VE ≥4 kelime sürüyorsa böl.
   - `_smooth`'u kelime sayısı yerine süreye (ör. <1.0 sn) göre yap.
   - Test: `tests/test_speakers.py`'ye sınırda kayan kelimeler örneği.
2. ✅ **(1.7.0)** **Örtüşmesiz (exclusive) ayırma:** (1.6.3: 3.x yaklaşığı `merge.exclusive_turns` ve 4.x çıktısını okuma yapıldı;
   kalan: 4.x + community-1 modelinin kurulumu) pyannote.audio 4.x + `pyannote/speaker-diarization-community-1`;
   çıktıdaki `exclusive_speaker_diarization` tam bu iş (metinle hizalama) için. Kodda 4.x desteği zaten var
   (`getattr(out, "speaker_diarization", out)` → `exclusive_speaker_diarization` önce denenmeli).
   Kurulumda sürüm (torch 2.8 ile uyum) ve HF koşul sayfası kontrol edilmeli; `diar_model` ayarı ile seçilebilir yap,
   eskisi yedek kalsın. 3.x'te yaklaşık eşdeğer: örtüşen turlarda kısa olanı kırp.
3. **Ölçüm (DER):** kullanıcı 2–3 videoda doğru konuşmacıları düzeltsin (adım 4), `pyannote.metrics`
   `DiarizationErrorRate` ile `stt olc --konusmaci` ekle. Her değişiklik bu sayıyla karşılaştırılır.
4. ✅ **(1.7.0)** **Düzeltme arayüzü = eğitim verisi:** "Metni göster" okuyucusunda satırın konuşmacısına tıklayıp değiştirme;
   düzeltmeler `<çıktı>/<ad>.duzeltme.json` (başla, bitir, doğru konuşmacı) olarak saklanır. Bu hem ölçüm verisi
   hem eğitim verisidir.
5. ✅ **(1.7.0, `stt/voices.py`)** **Ses imzası (eğitimsiz "öğrenme"):** düzeltilmiş/onaylı çıktılardan her bilinen kişi (ör. her TD) için
   pyannote gömme vektörü (`pyannote/wespeaker-voxceleb-resnet34-LM`, `Inference(window="whole")`) ortalaması
   `<out>/.stt/sesler/<ad>.npy`. Yeni videoda kümeler bu imzalarla kosinüs benzerliğine göre adlandırılır
   (eşik ~0.6) → aynı TD her videoda aynı ad; "en uzun konuşan = TD" tahmini gereksizleşir.
6. **Ayar optimizasyonu (az veriyle):** 1–2 saat düzeltilmiş veriyle `pyannote.pipeline.Optimizer` ile
   `clustering.threshold` ve `segmentation.min_duration_off` aranır, sonuç `stt.toml`'a yazılır.
7. **Gerçek ince ayar (çok veri, en son):** ~10+ saat etiketli veriyle `pyannote/segmentation-3.0` ince ayarı
   (pyannote.audio eğitim tarifi, `pyannote.database` protokolü, RTX 3090 yeter). Ancak 3–6 yetmezse.
8. Alternatif model: NVIDIA Sortformer (`docs/GELISTIRME.md` Nemotron notu) — adım 3'teki DER ile karşılaştır.

## Kullanıcıya söylenecekler (kısa)
- Kişi sayısını biliyorsanız "Kişi sayısı"na yazın (en büyük etki, hemen).
- Eğitmek = birkaç videoda konuşmacıları düzeltmek (adım 4 gelince); program sesleri ve ayarları bundan öğrenir.
