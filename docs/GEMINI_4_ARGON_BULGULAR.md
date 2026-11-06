# GEMINI 4 ARGON BULGULAR — Uçtan Uca Kod Denetimi, Konuşmacı Ayırma Kök Neden Analizi ve SOTA (Dünyanın En İyisi) Araştırma Raporu

> **Claude Opus / Fable İçin Not:** Bu rapor, kullanıcının talebi doğrultusunda tüm kod tabanının (`stt/`, `installer/`, `tests/`, `docs/`) uçtan uca taranmasıyla hazırlanmıştır. Uygulama koduna dokunulmamıştır. Bir sonraki geliştirme oturumunda aşağıdaki bulguları ve çözüm mimarisini temel alarak düzeltmeleri uygulayabilirsin.
>
> **⚠️ Önce 0. bölümü oku (Claude doğrulaması, 2026-11-06):** maddelerin yaklaşık yarısı yanlış ya da abartılı.
> Bölüm 1-5'teki kod alıntılarının bir kısmı koddaki hâliyle aynı değil (ör. `SENTENCE_GAP_SEC`, `STRONG_CHANGE_WORDS`,
> `diarize_min_speakers`, `_job_busy`, `window_s` adları kodda yok). Bir maddeyi uygulamadan önce 0. bölümdeki sonucuna bak.

---

## 0. CLAUDE DOĞRULAMASI (2026-11-06) — HER MADDE KODLA KONTROL EDİLDİ

Yöntem: her iddia için kod okundu; kütüphane davranışı kurulu sürümlerin kaynağından kontrol edildi (pyannote.audio 4.0.7,
faster-whisper 1.2.1, DiariZen `main`); `merge.py` davranışı gerçek kodla küçük örneklerle denendi. Windows'ta / ekran
kartında hiçbir şey denenmedi. İşaretler: ✅ doğru · ⚠️ kısmen / abartılı · ❌ yanlış · ❓ kanıtsız (ölçümle karar verilmeli).

### 0.1 Konuşmacı ayırma "kök nedenleri" (bölüm 1)
| Madde | Sonuç | Kanıt |
|---|---|---|
| KN1: cümle içindeki kısa başka-konuşmacı parçası yutuluyor | ⚠️ | Doğru ama bilinçli eşik (`merge.SPLIT_MIN_WORDS=4`, `SPLIT_MIN_SEC=1.5`; YAPILACAKLAR K4-AÇIK). Denendi: noktalamasız tek "cümle"de başta 3 kelime / 0.9 sn muhabir sorusu ve 3 kelime / 1.5 sn başka konuşmacı çoğunluğa katılıyor; 4 kelime / 1.7 sn bölünüyor. Eşiği ölçümsüz düşürmek yanlış bölünmeyi artırır (raporun kendi KN3'ü: kelime zamanları kayıyor). |
| KN1: "?" istisnası yalnız cümle sonundaki parça için | ✅ | `merge.py:190-194` yalnız son parçaya bakıyor. |
| KN2: 0.4–1.0 sn soru koruması "ölü kod" | ❌ | Soru cümlesi süreye göre değil kelime sayısına göre yumuşatılıyor (`merge.py:271`, `min_run=2`). Denendi: sondaki 2 kelime / 0.6 sn "and Saka?" ayrı kalıyor. Yalnız TEK kelimelik soru parçası yutuluyor. |
| KN2: cümle başındaki <1 sn parça sağdakine katılıyor | ✅ | Tasarım gereği (<1 sn = sıçrama sayılıyor); KN1 ile aynı eşik konusu. |
| KN2: `_smooth` yerinde değişim hatası | ❌ | Hata değil: art arda kısa parçalar uzun komşuya katılıyor (istenen). Raporun alıntıladığı kod `merge.py`'deki kod değil. |
| KN3: toplu çeviride `condition_on_previous_text` eksik | ❌ | faster-whisper 1.2.1 toplu modda bunu her zaman `False` yapıyor (`faster_whisper/transcribe.py:547`); vermek hiçbir şeyi değiştirmez. |
| KN3: `no_words` parçada konuşmacı satır düzeyinde | ⚠️ | Doğru, ama yalnız iki deneme (toplu, sonra toplusuz) başarısız olunca (`engines.py:373-386`). "Hata kelime zamanı (DTW) çekirdeğinden" iddiası kanıtsız. |
| KN3: forced alignment (WhisperX tarzı) | ❓ | Yararlı olabilir, ölçülmeli. "±30–50 ms" iyimser. Dikkat: `torchaudio.functional.forced_align` / MMS_FA 2.8'de "deprecated", 2.9'da kaldırıldı → "torchaudio zaten kurulu, ek kurulum yok" önerisi torch yükseltmesinde kırılır. |
| KN4: `agc()` konuşmacı ayırma sesine de uygulanıyor | ✅ | `engines.py:778-780` (tek `agc` ayarı, Whisper ile ortak). |
| KN4: "+20 dB yankıyı büyütüp ses imzalarını bozuyor" | ❓ | Kanıtsız varsayım. `agc` uzaktaki muhabirleri bulmak için eklendi (`audio.py:71-75`). Kaldırmak muhabir kaçırmayı geri getirebilir → ayrı `diar_agc` ayarıyla A/B ölçümü. |
| KN5.1: DiariZen seçiliyken pyannote da yükleniyor, HF token isteniyor | ✅ | `engines.py:511-512, 530`. Bilinçli: DiariZen hata verirse pyannote'a düşülüyor. İyileştirme: pyannote'u yalnız gerekince yükle. |
| KN5.2: VRAM çakışması | ⚠️ | 24 GB 3090'da büyük olasılıkla sığar; ölçülmedi. |
| KN5.3: DiariZen'de `min/max_speakers` çalışmıyor | ❌ | Çalışıyor: `DiariZenPipeline.__call__` `self.min_speakers/max_speakers`'ı kümelemeye veriyor (`diarizen/pipelines/inference.py:153-163`, `main`). |
| KN5.4: DiariZen'de ses imzası yok | ✅ | Zaten biliniyordu (YAPILACAKLAR K6). |
| KN5.5: DiariZen ortamı torch 2.1.1+cu121 | ✅ | Doğru; kullanıcının RTX 3090/3080'inde sorun değil (RTX 50 serisinde çalışmaz → pyannote'a düşer). |
| KN5 önerisi: SUPlime (`rewayai/suplime`, `suplime-large`) | ⚠️ | Model gerçek; pyannote.audio ≥4.0.7 (uygulamadaki sürüm) + `pip install suplime` (MIT) ile ana ortamda çalışıyor, HF token istemiyor. Ama "doğrudan pyannote" değil (ek paket şart). Ağırlık lisansı doğrulanamadı (rapordaki "CC BY-NC" belirsiz). Skorlar yayıncının kendi ölçümü ve 8 veri seti (12 değil). Yine de denenecek en iyi aday: ayrı ortam gerekmez. |
| KN6.1: ses kütüphanesi düzeltilmemiş küme imzasını öğreniyor | ✅ | `voices.py:275-289` ilk ayırmadaki küme ortalamasını kaydediyor; düzeltmeden sonra yeniden hesaplanmıyor; elle eklenen yeni kişi hiç öğrenilmiyor. Küme çoğunlukla doğru kişiyse etkisi küçük. Düzeltmek için imza ekran kartında yeniden çıkarılmalı. |
| KN6.2: "Konuşmacılar" ekranında süre yanlış | ✅ | `gui.py:715` `end` göndermiyor, `gui.html:768` tahmin ediyor. Yalnız ekrandaki saniye; kaydedilen dosyada süre doğru (`voices.apply_corrections`). |
| KN6.3: `stt egit` 99 kat yavaş (önbellek her denemede sıfırlanıyor) | ❌ | Aynı `f` sözlüğü her denemede kullanılıyor (`tune.py:217, 226`); pyannote 4.0.7 `training=True` iken bölütleme ve imzaları bu sözlükte saklıyor (`speaker_diarization.py:321-326, 361-370`), `instantiate` silmiyor. Sinir ağı bir kez çalışır. |
| KN6.3: referans (Whisper satırları) ile pyannote zamanları farklı | ⚠️ | 0.5 sn yaka (`tune.speaker_error`, collar) ile hafifletilmiş. Ama referansta birleştirilen <1 sn boşluklar "hata" sayılıyor → mutlak hata şişer; `segmentation.min_duration_off` bu yüzden yanlı seçilebilir. |
| KN7: `min_speakers=2` tek kişilik videoyu ikiye bölüyor | ✅ | Etkisi gerçek ve rapordakinden geniş: `diarize_min_speakers` diye ayar yok; 2 `pipeline.py:564`'te sabit ve basın + podcast/film HER modda uygulanıyor (K2 notu "basın modunda" diyordu). pyannote 4 VBx, bulduğu küme 2'den azsa KMeans ile zorla 2'ye bölüyor (`clustering.py:626-636`). **Raporda olmayan ek hata:** `stt.toml`'da `min_speakers = 3` yazılırsa `_speakers_wanted` (`pipeline.py:320`) bunu kesin sayı yapıyor (en çok da 3). |

### 0.2 "Diğer hatalar" tablosu (bölüm 2)
| # | Sonuç | Kanıt |
|---|---|---|
| 1 `_speaker_at` geri giden zaman | ⚠️ | Kenar durum: segmentler sıralı (`merge_chunks`), 3 tur geri bakılıyor. Pratikte görülmedi; düşük öncelik. |
| 2 "Var olmayan" model adları | ❌ | **YANLIŞ — değiştirme!** `claude-haiku-5-5` ($0.10/$0.50), `claude-sonnet-5-5` ($2/$10), `claude-opus-5-5` ($4/$20) gerçek; fiyatlar `llm.py` ile aynı. `gpt-5-mini` ve `gpt-transcribe` (OpenAI, 2026 yazı) gerçek. Ek not: `llm_model` elle `claude-haiku-4-5` yapılırsa `effort` parametresi 400 verir. |
| 3 OpenAI anahtar kutusu yok / anahtar silinemiyor | ✅ | OpenAI seçilebiliyor (`gui.html:400`) ama anahtar yalnız `OPENAI_API_KEY` ortam değişkeninden okunuyor. Silinememe bilinçli ("boş = kalsın", `gui.py:368-370`); silme düğmesi yok. |
| 4 GUI'de "link 11 ARS" yutuluyor | ❌ | Yanlış çerçeve: GUI'de linkin yanına GW/takım yazma hiçbir link türünde yok (https'li de); GUI otomatik eşleştirme kullanıyor. "link 11 ARS" yalnız CLI liste dosyasında (README:258). İstenirse yeni özellik. |
| 5 `ddelayed` yarış koşulu | ❌ | Aynı karttaki ayırma, metin görevi sıradayken / çalışırken zaten verilmiyor (`pipeline.py:836-838`); ayırma metnin bitmesine bağlı değil (yalnız sesi kullanır). Önerilen `_job_busy` kodda yok. |
| 6 `self.jobs` kilitsiz gezinme | ❌ | Bütün yazmalar `self.lock` altında; kilitsiz okumalar `list(...)` kopyası ya da kilit altında (`_dispatch`). |
| 7 Sürüm uyumsuzluğu | ⚠️ | Doğru ama zararsız: kurulum/güncelleme `pyproject.toml` `[diarize]` (4.0.7) kullanıyor; `requirements.txt` yalnız elle geliştirme. `pyproject.toml` değişirse güncellemede pip çalışır → boşuna değiştirme. |
| 8 Canlı yazıda 25 sn'de bir tekrar | ❌ | Uydurma: `live.py`'de `window_s`, `emitted_until`, `step_s` yok; parçalar sessiz noktadan örtüşmesiz kesiliyor (`live.py:42-50, 162-165`). |
| 9 Kanal takibi "görüldü"yü erken yazıyor | ⚠️ | Doğru yer `gui.py:666` (channel.py:94 değil). Başarısız video kuyrukta "Tekrar dene" ile görünür kalıyor, kuyruk kalıcı; yalnız takipçi kendiliğinden yeniden denemiyor. Düşük öncelik. |
| 10 HDemucs parça başına normalize | ❌ | Ortalama/std bütün ses için bir kez hesaplanıyor (`engines.py:672`); torchaudio örneğiyle aynı. |

### 0.3 Özellik tablosu ve SOTA bölümü (bölüm 3-4) düzeltmeleri
- #13 "Claude/OpenAI 404 veriyor" ❌ (yukarıda 0.2/2). #12 canlı tekrar ❌. #4 "DiariZen ciddi hatalı" ⚠️ abartı: min/max
  çalışıyor; kalanlar: pyannote da yükleniyor, ses imzası yok, kullanıcıda hiç denenmedi (K6b).
- #14 "Gemini ve Parakeet (onnx-asr) motoru çalışıyor" ❌: kodda yalnız `whisper` ve `openai` motoru var (`config.py:19`).
- "`nvidia/Nemotron-3-Diarization`" ❓ bulunamadı. NVIDIA'nın yayımladığı `nvidia/diar_streaming_sortformer_4spk-v2.1`:
  en çok 4 konuşmacı (basın toplantısına az), ağırlıklar NVIDIA Open Model License ("Apache 2.0" değil), NeMo Windows'ta ağır.
- Parakeet-TDT-0.6B-v2 / Canary-1B: İngilizce (Canary birkaç Avrupa dili); Türkçe/Azerbaycanca yok.
- ClearerVoice, DeepFilterNet3, BS-RoFormer, 3D-Speaker, WeSpeaker, Groq: gerçek projeler, hiçbiri burada ölçülmedi.
  Ses iyileştirme konuşma tanımayı / ayırmayı bazen bozar → ölçmeden eklenmez.

### 0.4 Claude'un önerdiği sıra (bölüm 5'in yerine)
0. **Ölçüm verisi önce:** kullanıcı 3–5 basın toplantısını "Konuşmacılar" ekranında düzeltip kaydetsin (K5); "Kontrol"de
   "Konuşmacı doğruluğu" satırı çıkar. Bu olmadan eşik/ses değişikliği kör yapılır.
1. Ölçüm gerektirmeyen güvenli düzeltmeler: (a) `min_speakers` 2 yalnız basın modunda + `cfg.min_speakers`'ın kesin sayı
   olması hatası; (b) "Konuşmacılar" ekranında süre (`end`); (c) OpenAI anahtar kutusu + anahtar silme; (d)
   `requirements.txt` pyannote 4.0.7; (e) DiariZen seçiliyken pyannote'u gerekince yükle.
   **✅ 1.9.4'te hepsi yapıldı** (+ ayrı `diar_agc` ayarı, eğitimde birleştirilmemiş satırlar, "Kontrol"de ayara göre
   ayrı doğruluk). Ayrıntı: `docs/KONUSMACI_AYIRMA.md` "1.9.4'te yapılanlar".
2. Ölçümle A/B (K5 verisi gelince): `merge` eşikleri (K4-AÇIK), ayırmada agc açık/kapalı (1.9.4: Ayarlar'dan seçilir,
   "Kontrol" ayara göre ayrı gösterir), SUPlime ↔ community-1, kelime hizalama (torchaudio 2.9'da `forced_align`
   kaldırıldı → `ctc-forced-aligner` gibi ayrı paket). SUPlime ağırlık lisansı için Gemini "CC BY-NC 4.0" diyor
   (Claude huggingface.co'ya erişemedi, doğrulamadı); kişisel/ticari olmayan kullanım için uygun.
3. Ses kütüphanesi: imzayı düzeltilmiş satırlardan yeniden çıkar. **✅ 1.9.5:** "Kaydet" karışık kümeyi öğrenmez, "Kaydet ve eğit" (ayrı süreç, ekran kartı) imzayı düzeltilmiş satırların sesinden çıkarır.

---

## 1. KONUŞMACI AYIRMA (SPEAKER DIARIZATION) NEDEN HÂLÂ TATMİN ETMİYOR? (7 KÖK NEDEN VE KOD HATALARI)

Uygulamada `pyannote/speaker-diarization-community-1` gibi güçlü bir model kullanılmasına rağmen konuşmacı ayırmanın pratikte (özellikle basın toplantılarında, hızlı soru-cevaplarda ve podcast'lerde) hatalı sonuç üretmesinin sebebi **modelin kendisi değil, modelin öncesindeki ses işleme (`audio.py`) ve sonrasındaki kelime-cümle birleştirme (`merge.py`, `voices.py`, `diarizen.py`) katmanlarındaki algoritmik hatalardır.**

### Kök Neden 1: `stt/merge.py` İçindeki `_sentence_labels` Fonksiyonunun Kısa Konuşmaları ve Soruları Yutması (KRİTİK HATA)
- **Konum:** `stt/merge.py:185-216` (`_sentence_labels`) ve `stt/merge.py:21-26` (sabitler).
- **Sorun:**
  - `SENTENCE_GAP_SEC = 0.6`, `STRONG_CHANGE_WORDS = 4`, `STRONG_CHANGE_SEC = 1.5`, `QUESTION_MIN_SEC = 0.4`.
  - `_sentence_labels`, Whisper'ın noktalama işaretlerine (`.?!…`) veya `> 0.6 sn` boşluğa göre kelimeleri "cümlelere" (`_sentences`) ayırır.
  - Eğer Whisper hızlı bir konuşmacı değişiminde (ör. muhabir araya girip soru sorduğunda veya teknik direktör hemen cevap verdiğinde) araya `.` veya `?` koymadıysa ve duraksama `< 0.6 sn` ise, **iki farklı konuşmacının sözleri tek bir cümle (`sent`) içine düşer.**
  - Satır `198-204`:
    ```python
    strong = [k for k, (a, b) in enumerate(runs)
              if (b - a) >= STRONG_CHANGE_WORDS and sum(durs[a:b]) >= STRONG_CHANGE_SEC]
    if len({raw[runs[k][0]] for k in strong}) >= 2:
        ...
    ```
    Bir cümlenin içinde konuşmacı değişiminin kabul edilmesi için **en az iki konuşmacının** aynı cümle içinde hem **en az 4 kelime** hem de **en az 1.5 saniye** konuşmuş olması şart koşulmuştur!
  - Aksi halde (`major = max(vote, key=vote.get)`), cümlenin tamamı çoğunluk konuşmacısına atanır ve diğer konuşmacının kelimeleri **tamamen ezilir!**
  - **Soru İstisnasındaki Hata (Satır `207-214`):**
    `QUESTION_MIN_SEC = 0.4` istisnası yalnızca `runs[-1]` (cümlenin **en sonundaki** koşu) için ve yalnızca cümle `?` ile bitiyorsa (`words[s1 - 1].get("word", "").rstrip().endswith("?")`) çalışır!
    - Eğer muhabir cümlenin **başında** kısa bir soru sorduysa (`runs[0]`) ve Whisper soru işareti yerine virgül koyup teknik direktörün cevabına devam ettiyse (ör. *"How is Saka, yeah he trained yesterday and..."*), muhabirin sorusu `runs[0]` olduğu için `runs[-1]` kontrolüne girmez ve teknik direktörün konuşmasına yutulur!
    - Podcast/sohbet modunda (`herkes`) kısa onaylamalar, tepkiler ve söz kesmeler (*"Exactly."*, *"Why?"*, *"No way."*, *"Tabii ki."*, *"Evet."*) `< 4 kelime` veya `< 1.5 sn` olduğu için konuşmacı değişimi olarak asla görünmez.

### Kök Neden 2: `_smooth` Fonksiyonunun Cümle Başı Kısa Konuşmaları Çift Ezmesi ve Yerinde (In-Place) Mutasyon Hatası
- **Konum:** `stt/merge.py:141-160` (`_smooth`) ve `stt/merge.py:194`.
- **Sorun:**
  - `_sentence_labels` her cümle için önce `raw = _smooth(raw, MIN_TURN_WORDS, durs=durs, min_sec=MIN_TURN_SEC)` (`MIN_TURN_WORDS = 2`, `MIN_TURN_SEC = 1.0`) çağırır.
  - `_smooth` içinde:
    ```python
    left = out[runs[k - 1][0]] if k > 0 else None
    right = out[runs[k + 1][0]] if k + 1 < len(runs) else None
    rep = left if (left == right or right is None) else (right if left is None else left)
    ```
    Eğer cümlenin **ilk** koşusu (`k == 0`) `< 1.0 sn` ise (`short = sum(durs[a:b]) < 1.0`), `left` `None` olduğu için `rep = right` olur ve cümlenin başındaki `< 1.0 sn`'lik konuşmacı anında ikinci konuşmacıya dönüştürülür!
  - Daha da kötüsü: `0.4 sn – 0.99 sn` arasındaki kısa bir soru cümlenin sonunda bile olsa, eğer `< 1.0 sn` ise satır `194`'teki `_smooth` tarafından **daha satır `207`'deki `QUESTION_MIN_SEC = 0.4` kontrolüne hiç ulaşamadan** bir önceki konuşmacıya (`left`) ezilir! Yani satır `207-214` arasındaki `QUESTION_MIN_SEC = 0.4` kodu, `0.4 sn – 1.0 sn` arasındaki sorular için `_smooth` yüzünden **ölü koddur (dead code)!**
  - Ayrıca `_smooth`, `out` listesini döngü içinde yerinde (`out[a:b] = [rep] * (b - a)`) değiştirdiği için ardışık kısa koşularda `left = out[runs[k - 1][0]]` orijinal konuşmacıyı değil, ezilmiş konuşmacıyı okur.

### Kök Neden 3: Whisper Kelime Zaman Damgası Kayması (`±500 ms`), CTranslate2 Hizalama Çökmesi (`no_words`) ve Forced Alignment Eksikliği
- **Konum:** `stt/engines.py:373-401` (`WhisperEngine._fallback`, `WhisperEngine._run`) ve `stt/merge.py:92-101` (`_explode_words`).
- **Sorun:**
  - `faster-whisper`'ın cross-attention tabanlı `word_timestamps=True` (CTranslate2 DTW hizalama) çıktısı, özellikle sessizlik sonrası ilk kelimelerde ve hızlı konuşma geçişlerinde `±200 ms ile ±600 ms` arasında kayma (drift) yapar.
  - Üstelik `stt/engines.py:390-394` satırlarında `self.batched is not None` olduğunda (`BatchedInferencePipeline`), `self.batched.transcribe(...)` çağrılırken `condition_on_previous_text=self.cond` parametresi verilmemekte ve `BatchedInferencePipeline` içsel VAD kırpması yaptığı için kelime zaman damgaları standart pipeline'a göre daha fazla kaymaktadır.
  - **Kritik Bağlantı (`parallel_for failed: invalid device ordinal` ve `no_words = True`):**
    Kullanıcının makinesinde sıkça görülen CTranslate2 `parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal` hatası, CTranslate2'nin `word_timestamps=True` (cross-attention DTW median filter) CUDA çekirdeğinden kaynaklanmaktadır! `WhisperEngine._fallback` (`stt/engines.py:384`) bu hatayı yakaladığında parçayı `words=False` (`res["no_words"] = True`) ile çevirmektedir. **Bir parçada `words=False` olduğunda o 5 dakikalık parçadaki tüm kelime zaman damgaları silinir ve konuşmacı ataması cümle/kelime bazında değil, kaba segment bazında yapılır!**
  - `pyannote` akustik olarak konuşmacı değişimini milimetrik bulsa bile, Whisper kelimenin zamanını `0.4 sn` erken/geç raporladığında veya `_fallback` kelime zamanlarını tamamen kapattığında konuşmacılar birbirine karışır.
  - **Çözüm:** `WhisperX` mimarisinde olduğu gibi CTranslate2'nin sorunlu iç `word_timestamps=True` çekirdeğine güvenmek yerine, Whisper çıktısından sonra **`torchaudio` Wav2Vec2 / MMS CTC Forced Alignment** (`±30–50 ms` hassasiyet) adımı eklenmelidir. Bu adım hem CTranslate2'nin `parallel_for` çökmesini ortadan kaldırır hem de kelime sınırlarını 10 kat daha hassas hizalar!

### Kök Neden 4: `stt/audio.py` İçindeki Agresif `agc()` Fonksiyonunun (`+20 dB`) Oda Yankısını Patlatıp Konuşmacı Vektörlerini Bozması
- **Konum:** `stt/audio.py:69-103` (`agc`) ve `stt/engines.py:779` (`Diarizer.run`).
- **Sorun:**
  - `agc()` fonksiyonu `-52 dB` (`gate = 10 ** (-52 / 20)`) üzerindeki her `0.4 sn` çerçeveyi `-20 dBFS` hedefe çekmek için **`gmax = 10.0` (+20 dB)** kazanç uygular!
  - Basın toplantısı salonlarında konuşmacı nefes aldığında veya uzak mikrofon açıkken `-50 dB` ile `-35 dB` arasındaki salon yankısı, klima/fan uğultusu ve kamera sesleri **10 kat (+20 dB)** büyütülür.
  - `Diarizer.run` (`stt/engines.py:779`) bu `agc(audio)` çıktısını doğrudan `pyannote` (WeSpeaker ResNet34) modeline verir. WeSpeaker konuşmacı gömmeleri (embeddings) arka plan yankısına ve kanal karakteristiğine çok duyarlıdır:
    1. Uzak mikrofondan gelen iki farklı muhabirin arkasındaki yükseltilmiş salon yankısı aynı olduğu için kümeleme algoritması **farklı muhabirleri tek kişi** sanır.
    2. Teknik direktörün sakin konuştuğu (yüksek AGC kazançlı) anlarla yüksek sesle konuştuğu (düşük AGC kazançlı) anların spektral dağılımı değiştiği için **aynı teknik direktörü 2–3 farklı konuşmacıya böler (over-splitting).**
  - **Çözüm:** Konuşmacı ayırma (diarization) modeline **kesinlikle ham veya yalnızca hafif tepe-normalize edilmiş (peak-normalized) / DeepFilterNet3-MossFormer2 ile yankıdan arındırılmış** ses verilmelidir; `+20 dB` kaba çerçeve AGC'si diarization kolundan kaldırılmalıdır.

### Kök Neden 5: `DiariZen` Entegrasyonundaki 5 Kritik Mimari Hata (`stt/diarizen.py` ve `stt/engines.py`)
- **Konum:** `stt/diarizen.py:25-60, 86-157` ve `stt/engines.py:508-535, 786-790`.
- **Sorunlar:**
  1. **Pyannote'un Gereksiz Yere VRAM'e Yüklenmesi (`stt/engines.py:511, 530`):** `self.engine == "diarizen"` seçili olsa bile `Diarizer.__init__` içinde `if not token: raise RuntimeError("diarize için HF_TOKEN gerekli...")` kontrolü çalışır ve hemen ardından `self.pipe = self._load()` ile **pyannote modeli GPU VRAM'ine yüklenir!**
  2. **Aynı GPU'da İki Ayrı Süreç (VRAM Çakışması):** `Diarizer.run` (`stt/engines.py:787`), DiariZen için `.venv-diarizen` içinde yeni bir Python alt süreci (`subprocess.run`) başlatır. Bu sırada üst süreçte Pyannote + HDemucs GPU belleğinde durmaktadır.
  3. **`min_speakers` / `max_speakers` Atamasının Çalışmaması (`stt/diarizen.py:42-45`):**
     ```python
     if a.min_speakers > 0:
         pipe.min_speakers = a.min_speakers
     if a.max_speakers > 0:
         pipe.max_speakers = a.max_speakers
     ```
     `BUTSpeechFIT/DiariZen` içindeki `DiariZenPipeline` sınıfında `pipe.min_speakers` özelliğini sonradan atamak kümeleme (`AgglomerativeClustering`) parametrelerini **değiştirmez**; bu değerler `config.toml` içindeki `clustering.args` sözlüğünden okunur.
  4. **Ses Kütüphanesi ve Eğitimin Devre Dışı Kalması (`stt/engines.py:788`):** DiariZen çalıştığında `self.last_emb = None` atanır. Bu yüzden DiariZen kullanıldığında **Ses Kütüphanesi (`sesler.json` ile otomatik isim tanıma) ve `stt egit` tamamen çalışmaz!**
  5. **Eski PyTorch 2.1.1+cu121 Bağımlılığı (`stt/diarizen.py:107`):** Ana uygulama `torch==2.8.0+cu128` (RTX 50 / Blackwell uyumlu) kullanırken `.venv-diarizen` içine `torch==2.1.1+cu121` (~2.5 GB) kurulmaktadır. RTX 50 serisi kartlarda `cu121` CUDA çekirdekleri çalışmaz!
  - **Çözüm:** Ayrı `.venv-diarizen` yerine doğrudan `pyannote.audio 4.x` ile yerel çalışan **`rewayai/suplime-large` / `rewayai/suplime`** (WavLM + Conformer powerset + WeSpeaker SimAM-ResNet34) veya DiariZen'in `pyannote 4.x` uyumlu ağırlıkları tek ortamda kullanılmalıdır.

### Kök Neden 6: Ses Kütüphanesi (`stt/voices.py`) ve Eğitim (`stt/tune.py`)Kusurları
- **Konum:** `stt/voices.py:258-277` (`learn_from_doc`), `stt/gui.py:709-747`, `stt/tune.py:180-245`.
- **Sorunlar:**
  1. **Kirli Küme Vektörünün (Dirty Centroid) Kaydedilmesi (`stt/voices.py:268-270`):**
     Kullanıcı arayüzde konuşmacı hatalarını düzelttiğinde (örneğin `SPEAKER_00` içindeki yanlış satırları başka konuşmacıya taşıyıp `SPEAKER_00`'a `"Mikel Arteta"` adını verdiğinde), `learn_from_doc` yeni ses gömmesi (embedding) **çıkarmaz**; ilk çalışmada kaydedilmiş `ses/<key>.json` içindeki **ham, düzeltilmemiş** `emb["SPEAKER_00"]` merkez vektörünü `"Mikel Arteta"` olarak `sesler.json`'a kaydeder! Eğer o küme başlangıçta karışık bir kümeyse, ses kütüphanesi bozuk vektörle kirlenir ve sonraki videolarda yanlış kişileri Arteta olarak etiketler.
  2. **Arayüzde Konuşmacı Sürelerinin Yanlış Hesaplanması (`stt/gui.py:715` ve `stt/gui.html:767-769`):**
     `speakers_load` (`stt/gui.py:715`) arayüze segmentleri gönderirken `end` alanını göndermez:
     `{"id": sg.get("id"), "start": sg.get("start"), "sid": sg.get("speaker_id"), "text": sg.get("text", "")}`.
     `gui.html:768` ise `end` olmadığı için süreyi `const e = nx ? Math.min(nx.start, sg.start + 30) : sg.start + 5;` diye bir sonraki segmentin başlangıcından tahmin eder! Arada 25 saniye sessizlik varsa 1 saniyelik konuşmayı 25 saniye olarak gösterir!
  3. **`tune.py` İçinde 99 Kat Yavaşlık ve Yanlış Referans (`stt/tune.py:224-243`):**
     - `pipe.training = True` ataması `pyannote.audio`'da `pipe.instantiate(cur)` çağrıldığında iç önbelleği sıfırladığı için 99 parametre kombinasyonunun her birinde sinir ağı çıkarımını (segmentation + embedding) baştan çalıştırır.
     - Ayrıca `ref["turns"]` (`voices.reference_turns(doc)`) Whisper segment zamanlarını (`< 1.0 sn` boşlukları birleştirerek) kullanırken `_hyp(pipe(f))` ham pyannote zamanlarını kullanır; Whisper'ın zaman kaymaları diarization hatası gibi ölçülür.

### Kök Neden 7: `diarize_min_speakers = 2` Varsayılanının Tek Kişilik Videoları Bozması (`stt/config.py:130`, `stt/pipeline.py:194`)
- **Konum:** `stt/config.py:130` (`diarize_min_speakers: int = 2`), `stt/pipeline.py:194-199` (`_diar_kw`).
- **Sorun:** Tek bir teknik direktörün veya oyuncunun kulüp televizyonuna verdiği birebir (soruların duyulmadığı veya kesildiği) röportajlarda `min_speakers=2` zorlaması yüzünden tek kişinin sesi zorla 2 ayrı konuşmacıya bölünür. Ayrıca WhisperX / Pyannote 3.1+ saha testlerinde katı `min_speakers`/`max_speakers` sınırlarının kümeleme kalitesini düşürdüğü, `min_speakers=1` (veya sınırsız bırakıp `threshold` ile karar verme) seçeneğinin çok daha sağlıklı olduğu kanıtlanmıştır.

---

## 2. KOD TARAMASINDA BULUNAN DİĞER TÜM HATALAR (DOSYA VE SATIR NUMARALARIYLA)

| # | Dosya ve Satır | Hata Özeti | Teknik Açıklama ve Çözüm |
|---|---|---|---|
| **1** | `stt/merge.py:104-123` (`_speaker_at`) | Kelime zamanlarının kesin artan (monotonic) olduğu varsayımı | `j = idx_hint; while j < len(turns) and turns[j][1] < t0: j += 1` ve `i = max(0, idx_hint - 3)` kullanılıyor. `_fill_gaps` ile eklenen boşluk segmentlerinde veya Whisper kelime zamanı titremelerinde `t0` geriye gidebilir. Ayrıca `best is None` olduğunda (boşluğa düşen kelime) `for s, e, spk in turns:` ile tüm turlar baştan taranıyor (`O(N*M)`). `bisect` ile ikili arama yapılmalı. |
| **2** | `stt/llm.py:23-30` ve `stt/engines.py:456`, `stt/config.py:118` | Var olmayan / hatalı varsayılan LLM ve OpenAI model adları | `llm.py` içinde varsayılan modeller `"claude-haiku-5-5"`, `"claude-sonnet-5-5"`, `"claude-opus-5-5"` ve `"gpt-5-mini"` olarak yazılmış; `config.py:118`'de `openai_model = "gpt-transcribe"` yazılmış. Anthropic ve OpenAI API'lerinde bu model kimlikleri `404 NotFoundError` verir (doğruları: `claude-haiku-4-5` / `claude-sonnet-4-5` / `gpt-4o-mini` ve `gpt-4o-transcribe` / `whisper-1`). |
| **3** | `stt/gui.py:368-371` ve `stt/gui.html:398-406` | GUI Ayarlarında OpenAI API Anahtarı alanı yok + girilen anahtarlar silinemiyor | `gui.html:400`'de özet sağlayıcı olarak `openai` seçilebiliyor ama Ayarlar penceresinde `openai_api_key` giriş kutusu yok! Ayrıca `save_settings` içinde `for k in ("hf_token", "gemini_api_key", "anthropic_api_key", "api_token"): if s.get(k): upd[k] = ...` yazıldığı için kullanıcı yanlış girdiği bir anahtarı kutuyu boşaltıp kaydederek **asla silemez**. |
| **4** | `stt/gui.py:508` (`Api.start`) | GUI'de `https://` olmayan linklerde FPL argümanlarının (`11 ARS`) yutulması | CLI'daki `read_link_list` (`sources.py:72`) `www.youtube.com/... 11 ARS` satırındaki `11 ARS` FPL bilgisini korurken, `gui.py:508` `urls.append("https://" + line.split()[0].rstrip(").,;"))` yaparak `line.split()[1:]` içindeki hafta ve takım kodunu çöpe atıyor. |
| **5** | `stt/pipeline.py:829` (`_pump_once`) | Paylaşımlı GPU'da `ddelayed` (ertelenmiş konuşmacı ayırma) yarış koşulu | `busy = any(b["job"] == t["job"] for b in self.backlog)` yalnızca `self.backlog` içindeki yazıya dökme parçalarını kontrol ediyor; o anda GPU işçisinde çalışmakta olan (`self.inflight`) veya kuyrukta bekleyen (`self.queued_n`) yazıya dökme parçalarını kontrol etmiyor! Son parça henüz GPU'da işlenirken diarization görevi erken tetiklenebiliyor. `busy` kontrolüne `or self._job_busy(t["job"])` eklenmeli. |
| **6** | `stt/pipeline.py:715, 793, 1116, 1462` | `self.jobs.values()` üzerinde kilitsiz (`self.lock` olmadan) gezinme | `_prepare_safe` arka plan iş parçacıklarında `self.jobs[item.key] = job` yazarken, `_shared_gpu()`, `stop_live()`, `_health()` ve `kill()` fonksiyonları `self.lock` almadan `self.jobs.values()` üzerinde döngü kuruyor (`RuntimeError: dictionary changed size during iteration` riski). |
| **7** | `pyproject.toml:7, 32` vs `requirements.txt:15` vs `stt/__init__.py:4` | Sürüm ve bağımlılık uyumsuzluğu | `stt/__init__.py` içinde `__version__ = "1.9.3"`, `pyproject.toml` içinde `version = "1.6.0"` ve `pyannote.audio==4.0.7`, `requirements.txt` içinde ise `pyannote.audio==3.4.0` ve `installer/stt.nsi` içinde `VERSION "1.4.0"`. |
| **8** | `stt/live.py:79-91` (`_loop`) | Canlı altyazıda (`Canlı`) pencere kaydırma sınırında mükerrer/kopuk metin | `window_s = 25.0` dolduğunda `buf = buf[- int(self.step_s * SAMPLE_RATE):]` (`3 sn`) bırakılıp `emitted_until = 0.0` sıfırlanıyor. Ancak yeni pencerede Whisper ilk `3 sn`'lik kısmı tekrar yazıya döktüğünde tüm segmentler yeniden ekrana ve dosyaya basılıyor (her 25 saniyede bir 3 saniyelik mükerrer cümle tekrarı). |
| **9** | `stt/channel.py:90-99` (`check_once`) | Kanal takipçisinde (`watch_channels`) işlenen videoların durum kontrolü | `check_once` yeni videoları `pipe.submit(it)` ile kuyruğa atar atmaz `seen.setdefault(ch, []).append(vid)` ile `kanallar.json`'a "görüldü" olarak kaydediyor (`satır 94, 98`). Eğer indirme veya işleme hata verirse (`error`) ya da uygulama kapatılırsa, kanal takipçisi bir sonraki taramada o videoyu bir daha asla denemiyor! |
| **10** | `stt/engines.py:678-684` (`Diarizer._vocals`) | HDemucs vokal ayırmada mono sesin normalize edilip 44.1 kHz stereo'ya kopyalanması | HDemucs doğal dinamik aralıklı (`[-1, 1]`) stereo müzik beklerken, `_vocals` içinde her parça `(chunk - mean) / std` ile birim varyansa (`±3–5` genlik) ölçeklenip modele veriliyor ve çıkış tekrar `* std + mean` yapılıyor; ancak sessiz/konuşmasız bir parçada `std` çok küçük olduğu için gürültü patlaması oluşabiliyor. |

---

## 3. UYGULAMANIN VAAT ETTİĞİ ÖZELLİKLERİN KONTROL TABLOSU (YAPABİLİYOR MU?)

`README.md`, `BENI_OKU.txt`, `CLAUDE.md` ve arayüzde (`gui.html`) vaat edilen tüm özelliklerin kod karşılığı ve gerçek çalışma durumu:

| # | Vaat Edilen Özellik | Durum | Neden Tam Yapamıyor / Kodda Nerede Sorun Var? |
|---|---|---|---|
| 1 | **Çift GPU (RTX 5060 Ti + GTX 1050 Ti) ve Tek GPU Otomatik Görev Dağılımı** | ✅ **Çalışıyor** | `config.py:185-234` (`resolve_layout`) ve `pipeline.py` başarılı. Yalnızca `pipeline.py:829`'daki `ddelayed` kontrolünde `self._job_busy(t["job"])` eksikliği düzeltilmeli. |
| 2 | **Basın Toplantısı Konuşmacı Ayırma (TD + Muhabir Soru-Cevap)** | ⚠️ **Kısmen / Sorunlu** | `merge.py:185-216` (`_sentence_labels`) noktalama olmayan veya `< 0.6 sn` duraksamalı geçişlerde `< 4 kelime` / `< 1.5 sn` konuşmaları yutuyor; `_smooth` cümle başı `< 1.0 sn` soruları siliyor; `audio.py:agc` (`+20 dB`) yankıyı büyütüp konuşmacı kümelemesini bozuyor; kelime zamanları forced alignment olmadığı için kayıyor. |
| 3 | **Podcast / Film Modu (`herkes` ayrı konuşmacı)** | ⚠️ **Kısmen / Sorunlu** | Kısa diyaloglar, söz kesmeler ve onaylamalar (`< 1.5 sn`) `merge.py` tarafından karşı konuşmacıya yutuluyor. |
| 4 | **DiariZen (WavLM) İkinci Konuşmacı Ayırma Motoru** | ❌ **Ciddi Hatalı** | `engines.py:511, 530` yine de `HF_TOKEN` istiyor ve pyannote'u VRAM'e yüklüyor; `diarizen.py:42-45` `min/max_speakers` ayarını çalıştıramıyor; `last_emb = None` olduğu için Ses Kütüphanesi ve Eğitimi devre dışı bırakıyor; `.venv-diarizen` eski `torch 2.1.1+cu121` kuruyor. |
| 5 | **Ses Kütüphanesi (`sesler.json`) ile Otomatik İsim Tanıma (Arteta, Slot vb.)** | ⚠️ **Kusurlu** | `voices.py:258-277` (`learn_from_doc`) kullanıcının taşıdığı satırlara göre temiz embedding hesaplamak yerine ilk çalışmadaki karışık/ham küme merkezini (`emb[sid]`) kaydediyor; zamanla ses kütüphanesi kirleniyor. |
| 6 | **Arayüzden Konuşmacı Düzeltme ve Süre Gösterimi** | ⚠️ **Kusurlu** | `gui.py:715` segmentlerin `end` zamanını göndermediği için `gui.html:768` konuşmacı sürelerini bir sonraki segmentin başlangıcına kadar (30 sn'ye kadar sessizlikleri konuşma sayarak) yanlış hesaplıyor. |
| 7 | **Konuşmacı Ayırmayı Düzeltilmiş Videolarla Eğitme (`stt egit` / GUI "Eğit")** | ⚠️ **Çok Yavaş ve Kusurlu** | `tune.py:224-243` `pipe.instantiate(cur)` her adayda önbelleği sıfırladığı için 99 kombinasyonda sinir ağını 99 kez baştan koşturuyor; ayrıca Whisper segment zamanlarını ham akustik turlarla karşılaştırıyor. |
| 8 | **Arka Plan Müziği / Jenerik Temizleme (HDemucs Vokal Ayırma)** | ✅ **Çalışıyor (Geliştirilebilir)** | `engines.py:648-689` çalışıyor ancak HDemucs yerine BS-RoFormer / Mel-Band RoFormer veya ClearerVoice-Studio yankı+gürültü giderme basın toplantıları için çok daha uygun. |
| 9 | **Uzak Mikrofon / Kısık Sesli Muhabirleri Duyulur Yapma (AGC)** | ⚠️ **Yan Etkili** | `audio.py:69-103` (`agc`) `+20 dB` kazançla kısık sesi yükseltirken salon yankısını da 10 kat büyütüp diarization gömmelerini bozuyor. ASR ve Diarization ses kolları ayrılmalı! |
| 10 | **Whisper Boşluk Doldurma (`_fill_gaps` — Atlanan Soruları Kurtarma)** | ✅ **Çalışıyor** | `pipeline.py:495-563` diarization'ın konuşma bulup Whisper'ın boş geçtiği `>= 1.5 sn` aralıkları `vad=False` ve `+15 dB` ile yeniden tarıyor. Güzel bir özellik; ancak kurtarılan segmentlerin kelime zamanları `_speaker_at` içinde `idx_hint` varsayımını bozabiliyor. |
| 11 | **YouTube, X (Twitter), Instagram, TikTok, Facebook, Kulüp Siteleri İndirme** | ✅ **Çalışıyor** | `fetch.py` ve `browser.py` (`yt-dlp`, `curl_cffi`, tarayıcı çerezi, gerçek Chrome yakalayıcı) oldukça kapsamlı. |
| 12 | **Canlı Yayın / Mikrofon / Sistem Sesi Anlık Altyazı (`Canlı` sekmesi)** | ⚠️ **Kısmen Sorunlu** | `live.py:79-91` her 25 saniyede bir pencereyi sıfırlarken son 3 saniyeyi tekrar yazıya döküp mükerrer satır üretiyor; ayrıca canlı modda konuşmacı ayırma yok. |
| 13 | **Yapay Zeka (LLM) ile Özet, Soru-Cevap ve Sakatlık/FPL Raporu** | ⚠️ **Kısmen Kırık** | `Gemini` (`gemini-2.5-flash`) ve `Ollama` çalışıyor; ancak `Claude` (`claude-haiku-5-5`) ve `OpenAI` (`gpt-5-mini`) varsayılan model adları gerçekte var olmadığı için 404 hatası veriyor ve GUI'de OpenAI API anahtarı kutusu yok. |
| 14 | **Harici Transkripsiyon Motorları (OpenAI API / Gemini / Parakeet ONNX)** | ⚠️ **Kısmen Kırık** | `config.py:118` `openai_model = "gpt-transcribe"` var olmayan bir model adı (`gpt-4o-transcribe` veya `whisper-1` olmalı). Gemini ve Parakeet (`onnx-asr`) çalışıyor. |
| 15 | **FPL (`presser.py`) Entegrasyonu** | ⚠️ **GUI'de Kısmen Kırık** | CLI'da çalışıyor; GUI'de `https://` olmadan yapıştırılan linklerde `11 ARS` argümanları `gui.py:508` yüzünden düşüyor. |
| 16 | **Kanal Takibi (`Kanallar` sekmesi / `--kanallar`)** | ⚠️ **Hata Toleransı Eksik** | `channel.py:94` videoyu kuyruğa atar atmaz `kanallar.json`'a işlendi olarak yazıyor; indirme/işleme başarısız olursa video bir daha denenmiyor. |
| 17 | **İzleme Klasörü (`--izle`), Yerel HTTP API (`--api`), WER Ölçümü (`stt olcum`)** | ✅ **Çalışıyor** | `watch.py`, `api.py`, `bench.py` beklendiği gibi çalışıyor. |
| 18 | **Otomatik Güncelleme (`updater.py`) ve Sistem Kontrolü (`doctor.py`)** | ✅ **Çalışıyor** | Yedekli ve geri dönüşlü (rollback) güncelleme mekanizması sağlam. |

---

## 4. HER ÖZELLİK İÇİN "DÜNYANIN EN İYİSİ" (SOTA) VE EN İYİ ÜCRETSİZ / AÇIK KAYNAK ÇÖZÜMLER ARAŞTIRMASI

Uygulamanın her bir bileşeni için 2025–2026 itibarıyla dünyadaki en iyi (SOTA) açık kaynak ve ücretsiz çözümler aşağıda karşılaştırılmıştır:

### 4.1. Konuşmacı Ayırma (Speaker Diarization) — Dünya Liderleri
| Çözüm / Model | Lisans / Ücret | Neden Dünyanın En İyisi? | Bu Uygulamaya Nasıl Entegre Edilir? |
|---|---|---|---|
| **1. `rewayai/suplime-large` & `rewayai/suplime`** *(WavLM-Large/Base+ + Conformer Powerset + WeSpeaker SimAM-ResNet34)* | Açık Ağırlık (CC BY-NC 4.0 / MIT kod) — **Ücretsiz** | Pyannote'un 12 veri setli (AMI, VoxConverse, DIHARD, CHiME-6, NOTSOFAR-1 vb.) genel benchmark'ında **%15.86 DER** ile hem PyannoteAI'ın ücretli ticari `precision-2` modelini (%16.06) hem de `DiariZen-L-s80-v2`'yi (%17.38) geride bırakan açık ağırlıklı lider modeldir. **En büyük avantajı:** Doğrudan `pyannote.audio 4.x` (`pyproject.toml`'da zaten kurulu!) üzerinde çalışır; `.venv-diarizen` gibi ikinci bir Python 3.10 ortamına ihtiyaç duymaz! | `stt/diarizen.py`'daki karmaşık 2. sanal ortam yerine doğrudan ana `.venv` içinde `pyannote.audio 4.x` ile seçenek olarak eklenebilir. |
| **2. NVIDIA NeMo `MSDD` (Multi-Scale Diarization Decoder) + `TitaNet-Large` + `MarbleNet` & `Sortformer` (`nvidia/Nemotron-3-Diarization`)** | Apache 2.0 — **Tamamen Ücretsiz ve Açık Kaynak** | Tek bir pencere boyutu yerine **5 farklı zaman ölçeğinde (0.5 sn – 3.0 sn)** aynı anda konuşmacı gömmesi çıkarır ve sinir ağı tabanlı MSDD ile birleştirir. Basın toplantılarındaki **0.5–1.5 saniyelik kısa muhabir sorularını ve söz kesmeleri** yakalamada kümeleme tabanlı modellerden çok daha üstündür. | `nemo_toolkit[asr]` veya ONNX/PyTorch ağırlıklarıyla yüksek doğruluk modu olarak eklenebilir. |
| **3. `pyannote/speaker-diarization-community-1`** *(Mevcut varsayılan)* | CC-BY-4.0 — **Ücretsiz** | Pyannote 3.1'in halefi; `exclusive_speaker_diarization` çıktısıyla Whisper birleştirmesini kolaylaştırır. Modelin kendisi güçlüdür; yukarıdaki **Kök Neden 1, 2, 3, 4** düzeltildiğinde performansı katlanarak artacaktır. | Mevcut model korunmalı, ancak ön/son işleme (`audio.py` ve `merge.py`) yeniden yazılmalıdır. |

### 4.2. Kelime Zaman Damgası Hizalama (Word-Level Forced Alignment) — EN KRİTİK EKSİK HALKA
| Çözüm / Kütüphane | Lisans / Ücret | Neden Dünyanın En İyisi? | Bu Uygulamaya Nasıl Entegre Edilir? |
|---|---|---|---|
| **1. `WhisperX` (`wav2vec2` Forced Alignment) / `torchaudio.pipelines`** | BSD-2 / MIT — **Ücretsiz** | Whisper'ın `±500 ms` kayan kelime zaman damgalarını CTC fonem hizalaması ile **`±30–50 ms` hassasiyete** indirir. Konuşmacı değişimlerinde cümlenin ilk/son 1–3 kelimesinin yanlış kişiye atanması sorununu **kökten çözer.** `torchaudio` zaten kurulu olduğu için ekstra ağır paket gerektirmez! | `WhisperEngine` veya `Pipeline` içinde yazıya dökme bittikten sonra `torchaudio` Wav2Vec2 / MMS hizalayıcı ile kelime `start`/`end` zamanları güncellenir. |
| **2. `ctc-forced-aligner` (`MahmoudAshraf97/ctc-forced-aligner` — Meta MMS-300M)** | MIT — **Ücretsiz** | Meta'nın `MMS-300M` modeliyle **1100+ dili** (Türkçe, İngilizce, Azerbaycanca, Arapça, vb. tamamı dahil) tek bir hafif modelle GPU/CPU üzerinde milisaniye hassasiyetinde hizalar. | Çok dilli (TR/AZ/EN/ES/DE/FR) basın toplantıları için tek modelle evrensel forced alignment sağlar. |

### 4.3. Ses İyileştirme, Yankı Giderme ve Vokal Ayırma (Speech Enhancement & Dereverberation)
| Çözüm / Model | Lisans / Ücret | Neden Dünyanın En İyisi? | Bu Uygulamaya Nasıl Entegre Edilir? |
|---|---|---|---|
| **1. `ClearerVoice-Studio` (`alibabasglab/MossFormer2_SE_48K` & `FRCRN`)** | Apache 2.0 — **Ücretsiz** | Alibaba Speech Lab'in 2025–2026 SOTA açık kaynak konuşma iyileştirme aracı. Kaba `agc()` gibi gürültüyü ve yankıyı büyütmek yerine, **salon yankısını (reverberation) ve arka plan uğultusunu silip uzak mikrofondaki muhabirin sesini kristal netliğe** kavuşturur. | Basın toplantılarında `agc()` yerine (veya `agc` öncesi) uzak muhabir seslerini netleştirmek için kullanılır. |
| **2. `DeepFilterNet3`** | MIT / Apache 2.0 — **Ücretsiz** | Çok düşük gecikmeli, hafif ve ses rengini (speaker embedding'i) bozmadan arka plan gürültüsünü temizleyen en hızlı açık kaynak derin öğrenme filtresi. | Canlı yayında (`live.py`) ve ön işlemede CPU/GPU'yu yormadan çalışır. |
| **3. `BS-RoFormer` / `Mel-Band-RoFormer` (`python-audio-separator`)** | MIT — **Ücretsiz** | MDX23 ve 2025–2026 ses ayırma yarışmalarının açık ara dünya birincisi. Arka plan müziğini ve stadyum/jenerik seslerini insan sesinden ayırmada `htdemucs`'tan çok daha yüksek SDR skoruna sahiptir ve konuşma harmoniklerine zarar vermez. | `engines.py:_vocals` içinde `htdemucs` yerine veya alternatif olarak kullanılabilir. |
| **4. FFmpeg `speechnorm` / `dynaudnorm` (İki Ayrı Ses Kolu Mimarisi)** | LGPL / GPL — **Ücretsiz (Zaten kurulu)** | Kendi yazdığımız `0.4 sn` pencereli kaba `agc()` yerine FFmpeg'in konuşma için tasarlanmış `speechnorm` filtresi çok daha doğal çalışır. **Kritik Kural:** `speechnorm`/AGC yalnızca **Whisper (ASR)** koluna uygulanmalı, **Pyannote (Diarization)** koluna ise yankısız/doğal dinamik aralıklı ses verilmelidir! | `audio.py` içinde `extract_wav` ve `Diarizer.run` ayrıştırılarak sıfır maliyetle hemen uygulanabilir. |

### 4.4. Konuşmacı Tanıma / Ses Kütüphanesi (Speaker Verification / Identification)
| Çözüm / Model | Lisans / Ücret | Neden Dünyanın En İyisi? | Bu Uygulamaya Nasıl Entegre Edilir? |
|---|---|---|---|
| **1. `3D-Speaker` (`ERes2NetV2` / `CAM++`)** | Apache 2.0 — **Ücretsiz** | Özellikle **kısa süreli (`< 2 sn`) konuşmalarda** konuşmacı kimliğini doğrulamak için tasarlanmış çok ölçekli (multi-scale feature fusion) SOTA açık kaynak modeldir (`onnx` kuantize modelleri hanya birkaç MB'tır). | `voices.py` içinde veya kısa segmentlerin hangi konuşmacıya ait olduğunu doğrularken kullanılabilir. |
| **2. `WeSpeaker` (`voxblink2_samresnet34_ft` / `resnet34-LM`)** | Apache 2.0 — **Ücretsiz** | VoxBlink2 + VoxCeleb2 ile eğitilmiş SimAM-ResNet34 modeli, farklı mikrofon ve kanal koşullarında aynı kişiyi tanımada en yüksek doğruluklu açık kaynak gömme modelidir. | `stt/voices.py` içinde sadece **kullanıcının onayladığı, çakışmasız (`> 2.0 sn`) temiz segmentlerden** doğrudan ses gömmesi hesaplanarak `sesler.json`'a kaydedilmelidir. |

### 4.5. Konuşma Tanıma (ASR / Speech-to-Text)
| Çözüm / Model | Lisans / Ücret | Neden Dünyanın En İyisi? |
|---|---|---|
| **1. `faster-whisper` (`large-v3` & `large-v3-turbo`) + CTC Forced Alignment** | MIT — **Ücretsiz** | Çok dilli (Türkçe, İngilizce, Azerbaycanca vb.) kullanımda yerel GPU'da en iyi açık kaynak modeldir. `large-v3-turbo` (`deepdml/faster-whisper-large-v3-turbo-ct2`), `large-v3` doğruluğuna çok yakın olup **3–4 kat daha hızlıdır.** |
| **2. NVIDIA `Parakeet-TDT-0.6B-v2` & `Canary-1B`** | CC-BY-4.0 — **Ücretsiz** | HuggingFace Open ASR Leaderboard'da İngilizce konuşma tanımada Whisper `large-v3`'ten daha düşük WER (kelime hata oranı) ve 10 kat daha yüksek hız sunar. |
| **3. Groq Cloud API (`whisper-large-v3` / `whisper-large-v3-turbo`)** | **Ücretsiz Katman (Free Tier)** | Bulut tabanlı ücretsiz/hızlı alternatif arandığında saniyeler içinde 1 saatlik sesi yazıya döker. |

### 4.6. Akustik + Semantik (LLM) Hibrit Konuşmacı Birleştirme (Diarization Post-Processing)
- **Dünyadaki En İyi Yaklaşım (SpeakerLM / DiarizationLM / LLM-Guided Turn Attribution):**
  Akustik modeller (`pyannote`) bazen hızlı soru-cevap geçişlerinde `0.5 sn` gecikir veya kısa bir soruyu kaçırır. Ancak metin düzeyinde bakıldığında:
  - *"Thank you. Good afternoon Mikel, how is Bukayo Saka after the knock?"* -> **Kesinlikle Muhabir (Soru)**
  - *"Yeah, he trained with us yesterday and he feels good."* -> **Kesinlikle Teknik Direktör (Cevap)**
- **Nasıl Uygulanmalı?**
  1. `merge.py` içinde kural tabanlı (Sıfır Maliyetli) **Soru-Cevap Akustik Sınır Düzeltici**: Bir cümlenin sonunda `?` varsa veya cümle soru kalıbıyla (*"How / What / Why / Can I ask / Do you think / Hocam / Peki / Neden / Nasıl"*) başlıyorsa ve o cümlede akustik olarak `%25+` oranında farklı bir konuşmacı (veya yerel akustik değişim) varsa, `STRONG_CHANGE_SEC = 1.5` şartı aranmadan konuşmacı geçişine izin verilmelidir!
  2. İsteğe bağlı olarak **Gemini 2.5 Flash (Ücretsiz Katman)** veya yerel **Ollama (Qwen 2.5 / Llama 3.1)** ile yalnızca belirsiz/sınırda kalan cümlelerin konuşmacı etiketleri (`TD` mi `Muhabir` mi?) kontrol ettirilmelidir.

---

## 5. CLAUDE OPUS / FABLE İÇİN ÖNCELİKLİ UYGULAMA PLANI (ADIM ADIM YOL HARİTASI)

Claude, bir sonraki oturumda kullanıcı onay verdiğinde şu sırayla ilerlemelidir:

1. **Adım 1 — `stt/merge.py` Konuşmacı Birleştirme Algoritmasının Tamamen Yenilenmesi (En Yüksek Etki):**
   - `_smooth` içindeki cümle başı (`k == 0`) ezme hatasını ve yerinde (`in-place`) mutasyon hatasını düzelt; `_smooth` işlemini cümle içine değil, kelimeler cümlelere atanmadan önce yalnızca `< 0.25 sn` (1 kelimelik anlık akustik sıçrama) gürültüleri için uygula.
   - `_sentence_labels` içindeki `STRONG_CHANGE_WORDS = 4` ve `STRONG_CHANGE_SEC = 1.5` barajını düşür (`>= 2 kelime` ve `>= 0.6 sn`); ayrıca cümle içinde noktalama olmasa bile iki kelime arasında `>= 0.25 sn` duraksama varsa ve akustik konuşmacı değiştiyse cümleyi oradan böl!
   - Soru istisnasını (`QUESTION_MIN_SEC = 0.35`) yalnızca `runs[-1]` için değil, cümlenin başındaki (`runs[0]`) veya ortasındaki soru cümlecikleri için de çalıştır.
   - `_speaker_at` fonksiyonunu `bisect` ile `O(log N)` ve geri gitmelere dayanıklı hale getir.

2. **Adım 2 — `stt/audio.py` ve `stt/engines.py` İçinde ASR ve Diarization Ses Kollarının Ayrılması:**
   - `Diarizer.run` (`stt/engines.py:779`) içinden `+20 dB` `agc(audio)` çağrısını kaldır veya diarization için yalnızca hafif tepe normalizasyonu (`peak normalize`) + yüksek geçiren filtre (`80 Hz highpass`) uygula. Salon yankısının `+20 dB` büyütülüp konuşmacı gömmelerini (WeSpeaker embeddings) bozmasını engelle.

3. **Adım 3 — Kelime Zaman Damgaları İçin Forced Alignment (`torchaudio` Wav2Vec2 / MMS) Eklenmesi:**
   - `torchaudio.pipelines.MMS_FA` (veya `WAV2VEC2_ASR_BASE_960H`) ile Whisper'ın ürettiği kelimelerin başlangıç/bitiş zamanlarını ses dalgası üzerinde `±40 ms` hassasiyetle hizala.
   - `WhisperEngine._run` (`stt/engines.py:390`) içinde `self.batched.transcribe` çağrısına eksik parametreleri ekle veya diarization açıkken doğrudan hassas zaman damgası yolunu kullan.

4. **Adım 4 — `stt/voices.py` ve `stt/gui.py` Ses Kütüphanesi ve Arayüz Düzeltmeleri:**
   - `stt/gui.py:715` (`speakers_load`) çıktısına `"end": sg.get("end")` alanını ekle ve `stt/gui.html:767-769` (`recount`) içinde gerçek `sg.end - sg.start` süresini kullan.
   - Kullanıcı GUI'de konuşmacı satırlarını taşıdığında veya düzelttiğinde, ilk çalışmadaki karışık `emb[sid]` merkezini kaydetmek yerine yalnızca o konuşmacıya ait temiz (`> 2.0 sn`, çakışmasız) segmentlerden yeni konuşmacı gömmesi (embedding) çıkarıp `sesler.json`'a kaydet.
   - `stt/config.py:130` `diarize_min_speakers` varsayılanını `1` yap (tek kişilik röportajların zorla ikiye bölünmesini önle).

5. **Adım 5 — `stt/diarizen.py` ve Diğer Kod Hatalarının Temizlenmesi:**
   - `stt/engines.py:510-535` içinde `self.engine == "diarizen"` seçiliyse Pyannote modelini VRAM'e yükleme ve `HF_TOKEN` zorunluluğunu kaldır; `diarizen.py` içinde `min_speakers`/`max_speakers` parametrelerini `config.toml` (`clustering.args`) üzerinden geçir veya `pyannote 4.x` ile doğrudan çalışan `rewayai/suplime` modelini entegre et.
   - `stt/llm.py:23-30` ve `stt/config.py:118` içindeki hatalı model adlarını (`claude-haiku-4-5`, `claude-sonnet-4-5`, `gpt-4o-mini`, `gpt-4o-transcribe`) düzelt.
   - `stt/gui.py:368-371` içinde API anahtarlarının silinebilmesini sağla ve `openai_api_key` alanını `gui.html`'e ekle.
   - `stt/gui.py:508` içinde şemasız linklerdeki FPL argümanlarını (`11 ARS`) koru.
   - `stt/pipeline.py:829` (`ddelayed`) ve `self.jobs.values()` kilit (`self.lock`) eksikliklerini gider.
   - `stt/channel.py:94` içinde kanal videolarını kuyruğa atarken değil, işleme başarıyla tamamlandığında `kanallar.json`'a kaydet.
