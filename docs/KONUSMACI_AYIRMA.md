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

## 1.9.6'da yapılanlar (ilk cümle yutulması + Konuşmacılar satır düzenleme/bölme/aynı anda konuşma)
- **İlk cümlenin sonraki konuşmacının çok cümleli bloğuna yutulması (`stt/merge.py`):**
  1. `_smooth`: cümle başında (`k == 0`, `left is None`) konuşmacı koşusu kendisinden kısa/eşit bir kuyruk kaymasına
     (`rsize <= cur`) asla ezdirilmez; ayrıca cümle başındaki `≥ 2 kelime` ve `≥ 0.8 sn` (`HEAD_MIN_WORDS/HEAD_MIN_SEC`)
     koşu korunur (eskiden 0.9 sn'lik ilk cümlenin son kelimesi 0.2 sn sonraki konuşmacıya kayınca tüm cümle sonraki
     konuşmacıya yazılıyordu).
  2. `_sentences`: nokta/soru/ünlem ve `≥ 0.6 sn` boşluğa ek olarak, virgül/noktalı virgül (`,;:`) veya `≥ 0.25 sn`
     (`TURN_GAP`) duraksamada akustik konuşmacı değişiyorsa (her iki tarafta en az 2 tutarlı kelime) cümleyi böler.
  3. `_sentence_labels`: noktalamasız/duraksamasız geçişte bile cümle başındaki `≥ 2 kelime, ≥ 0.8 sn` koşu önceki
     cümledeki konuşmacıdan (`prev_spk`) farklı bir konuşmacıya geçiyorsa korunur (orta yerdeki sandviç sapmalar ise
     bölünmez).
- **Konuşmacılar ekranında satır içi düzenleme (`stt/gui.html`, `stt/gui.py`, `stt/voices.py`):**
  - **Metin düzeltme (`.lntxt`):** yanlış anlaşılan kelimeler doğrudan satır kutusunda düzeltilir; `voices._sync_words`
    kelime zamanlarını günceller ya da temizler (böylece `.srt` eski kelimeleri yazmaz).
  - **Satır bölme (`✂ Böl`, `voices.split_segment`):** iki konuşmacının cümleleri tek satırda birleşmişse imleç
    yerinden (imleç yoksa ilk cümle sonundan `.?!…`, o da yoksa ilk `,;:` ya da ortadan) iki satıra böler; kelime
    zamanları varsa bölünme saniyesini kelime sınırından alır.
  - **Aynı anda konuşma (`👥 Aynı anda`):** iki kişinin aynı anda söylediği kelime/cümle için aynı `[start, end]`
    aralığında ikinci konuşmacıya satır ekler. `tune._solo_spans` üst üste binen saniyeleri ses imzası çıkarımından
    (`voice_embeddings`) düşer; böylece karışık ses tek kişinin imzasını bozmaz.
  - **Kaçırılan söz (`+ Satır`), satır silme (`✕`) ve süre düzenleme (`.tm`'ye tıklama):** atlanan sözler eklenir,
    uydurma satırlar silinir.
  - **Eğitim imzasını koruma (`keep_duzeltme`):** `"Kaydet ve eğit"` ile çıkarılmış `src="duzeltme"` ses imzası,
    sonradan düz `"Kaydet"`e basıldığında küme saflığı `< %90` olsa bile silinmez.

## 1.9.5'te yapılanlar (ses kütüphanesi imza yenileme, Gemini raporu 0.4/3)
- "Kaydet": videonun kütüphanedeki kayıtları baştan kurulur (adı değişen / genel ada dönen konuşmacının eski kaydı
  kalmaz). Konuşmasının %10'undan fazlası başka kişiye taşınmış kümenin imzası öğrenilmez (`voices.cluster_purity`,
  `PURE_MIN`); bunun için ilk ayırma turları `ses/<key>.json` `turns`'te saklanır (eski çıktıda ilk düzeltmeden önce
  satırlardan çıkarılır: `ensure_orig_turns`). Öğrenilmeyenler ekranda yazar.
- "Kaydet ve eğit" (`stt egit`): her düzeltilmiş videoda kişi imzaları düzeltilmiş satırların sesinden yeniden
  çıkarılır (`tune.voice_embeddings`: pyannote'un imza modeli `pipe._embedding`, ≥1.5 sn satırlar, uçlardan 0.2 sn
  kırpılır, ≤10 sn parçalar, kişi başına ≤120 sn; ayırmaya giden sesle aynı ön işlem) ve o videonun karışık küme
  imzalarının yerine yazılır (`src: "duzeltme"`). Elle eklenen ("+ Yeni kişi") kişiler de öğrenilir. Doğrulandı:
  pyannote 4 VBx küme imzasını aynı modelin ham imzalarının ağırlıklı ortalamasıyla veriyor → karşılaştırılabilir.
- `sesler.json` iki süreçten yazılabildiği için kilit dosyası (`_FileLock`); okunamazsa üzerine yazılmaz (eskiden
  geçici okuma hatasında kütüphane boş sanılıp bütün sesler silinebilirdi); bozuk dosya `.bozuk-<zaman>` yedeklenir.
- Gerçek modelle / Windows'ta DENENMEDİ (testler sahte imza modeliyle). Kullanıcıda ilk denemede günlükte
  "Ses imzaları düzeltilmiş satırlardan yenilendi" satırı istensin.

## 1.9.4'te yapılanlar (Gemini raporu doğrulamasından, `docs/GEMINI_4_ARGON_BULGULAR.md` 0.4)
- Kişi sayısı: işe verilen sayı kesin; `stt.toml` `min_speakers`/`max_speakers` yalnız sınır (eskiden `min_speakers`
  tek başına kesin sayı oluyordu); basın toplantısında en az 2 otomatik, podcast/film'de alt sınır yok (pyannote 4 VBx
  küme az bulursa KMeans ile zorla bölüyordu: tek kişilik video ikiye bölünüyordu). `Pipeline._speaker_bounds`.
- DiariZen seçiliyken pyannote yalnız yedek gerekince yüklenir, HF token zorunlu değil (`Diarizer._ensure_pipe`).
- `diar_agc` (boş = `agc` ile aynı; Ayarlar'da seçilir): ayırmaya giden seste ses yükseltme ayrı açılıp kapanır.
  Çıktıya `settings.diar_agc` ve `settings.diar_model` (DiariZen dahil) yazılır; düzeltme kaydı (`duzeltme/*.json`
  `diar`) bunu saklar; "Kontrol" farklı ayarla işlenmiş videoları **ayara göre ayrı** gösterir (A/B: agc açık/kapalı,
  DiariZen/community-1 = K6d).
- Eğitim (`stt egit`) birleştirilmemiş satırlarla ölçer (`duzeltme/*.json` `segs`): birleşik turlarda <1 sn sessizlik
  "konuşma" sayılıp hata çıkıyordu (`min_duration_off` yanlı seçiliyordu). Eski kayıtlarda yalnız `turns` var.
- "Konuşmacılar" ekranında süre satırın gerçek bitişiyle (aradaki sessizlik sayılmıyor).

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
3. ✅ **(1.8.0, `stt/tune.py`)** Ölçüm: `tune.speaker_error` (doğru cevabın konuşma süresinde yanlış/eksik kişi oranı,
   açgözlü ad eşleme, değişimlerin ±0.5 sn'si sayılmaz; DER'in yanlış alarm kısmı yok çünkü referans satır düzeyinde).
   Eski plan: **Ölçüm (DER):** kullanıcı 2–3 videoda doğru konuşmacıları düzeltsin (adım 4), `pyannote.metrics`
   `DiarizationErrorRate` ile `stt olc --konusmaci` ekle. Her değişiklik bu sayıyla karşılaştırılır.
4. ✅ **(1.7.0)** **Düzeltme arayüzü = eğitim verisi:** "Metni göster" okuyucusunda satırın konuşmacısına tıklayıp değiştirme;
   düzeltmeler `<çıktı>/<ad>.duzeltme.json` (başla, bitir, doğru konuşmacı) olarak saklanır. Bu hem ölçüm verisi
   hem eğitim verisidir.
5. ✅ **(1.7.0, `stt/voices.py`)** **Ses imzası (eğitimsiz "öğrenme"):** düzeltilmiş/onaylı çıktılardan her bilinen kişi (ör. her TD) için
   pyannote gömme vektörü (`pyannote/wespeaker-voxceleb-resnet34-LM`, `Inference(window="whole")`) ortalaması
   `<out>/.stt/sesler/<ad>.npy`. Yeni videoda kümeler bu imzalarla kosinüs benzerliğine göre adlandırılır
   (eşik ~0.6) → aynı TD her videoda aynı ad; "en uzun konuşan = TD" tahmini gereksizleşir.
6. ✅ **(1.8.0)** Okuyucu "Konuşmacılar" → "Kaydet ve eğit" (`App.speakers_train` → ayrı süreç `stt egit`, monitörsüz
   kart). `duzeltme/*.json` videolarının sesi `.stt/egitim/<key>/ses.wav` (yoksa kaynaktan yeniden indirilir), müzik
   ayıklama + agc gerçek akıştaki gibi; `pipe.training=True` ile bölütleme/ses imzaları önbellekte, yalnız kümeleme
   ızgarası (`clustering.threshold` 0.4–0.9, `Fa`, `segmentation.min_duration_off`) denenir. En iyisi
   `.stt/konusmaci_ayar.json`'a (model başına) yazılır; `Diarizer(params=...)` yüklerken uygular. Windows'ta denenmedi.
   Eski plan: **Ayar optimizasyonu (az veriyle):** 1–2 saat düzeltilmiş veriyle `pyannote.pipeline.Optimizer` ile
   `clustering.threshold` ve `segmentation.min_duration_off` aranır, sonuç `stt.toml`'a yazılır.
7. **Gerçek ince ayar (çok veri, en son):** ~10+ saat etiketli veriyle `pyannote/segmentation-3.0` ince ayarı
   (pyannote.audio eğitim tarifi, `pyannote.database` protokolü, RTX 3090 yeter). Ancak 3–6 yetmezse.
8. Alternatif model: NVIDIA Sortformer (`docs/GELISTIRME.md` Nemotron notu) — adım 3'teki DER ile karşılaştır.

## Kullanıcıya söylenecekler (kısa)
- Kişi sayısını biliyorsanız "Kişi sayısı"na yazın (en büyük etki, hemen).
- Eğitmek = bir videonun metnini açıp "Konuşmacılar"da TÜM satırları düzeltmek, "Kaydet ve eğit". Ne kadar çok
  düzeltilmiş video, o kadar iyi. Düzeltilmeyen satırlar doğru sayılır (yarım düzeltme eğitimi bozar).
