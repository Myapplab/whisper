# Yapılacaklar / oturum devri

> **ÖNEMLİ — Gemini denetim raporu (`docs/GEMINI_4_ARGON_BULGULAR.md`):** kullanıcı Gemini'ye uçtan uca kod taraması,
> konuşmacı ayırma kök nedenleri ve SOTA araştırması yaptırdı. **Claude 2026-11-06'da her maddeyi kodla doğruladı:**
> sonuç raporun başındaki **0. bölümde**. Yaklaşık yarısı yanlış/abartılı (ör. Claude 5 ve `gpt-transcribe` model adları
> gerçek, DiariZen'de min/max çalışıyor, `stt egit` önbelleği çalışıyor, canlı yazı "tekrarı" kodda yok). Doğru çıkanlar
> ve önerilen sıra: 0.4. Konuşmacı ayırma işinden önce 0. bölümü oku; yanlış maddeleri uygulama.
> **Kullanıcı kararı (2026-11):** öncelik kalite, kullanım ikinci planda (CLAUDE.md "İki profil, önce kalite").

Kullanıcı iki Claude hesabıyla (iki profil) sırayla çalışır: biri sınıra gelince öbürü kaldığı yerden devam eder.
Ortak hafıza bu dosyadır. **Oturum sonunda güncelle** (Son durum + liste) ve `main`'e push et.

## ŞU ANKİ İŞ: konuşmacı ayırma iyileştirme (adım adım; her adım ayrı push)
Kullanıcı istedi: DiariZen + "acımasız eleştiri"nin 10 maddesinin hepsi. Kural: bir adım bitince push et, burada ✅ yap,
sıradakine geç. Kullanım biterse diğer profil ilk ✅ olmayan adımdan devam eder.
**SIRADAKİ (1.9.5 sonrası):** kullanıcı Güncelle'ye basıp 3-5 basın toplantısını düzeltecek, "Kontrol" ekranını paylaşacak. Gelince: K5 ölçümüyle K4-açık (merge eşikleri) + agc A/B ("Kontrol" ayara göre ayrı doğruluk gösterir) + K6b/K6d (DiariZen) + SUPlime denemesi. İlk eğitimde günlükte "Ses imzaları düzeltilmiş satırlardan yenilendi" satırı istensin (gerçek modelle denenmedi).
- [x] K1 (#10) ✅ (doctor: sürüm 3.x uyarısı; yedek model uyarısı zaten vardı) Kontrol: pyannote < 4 ise "community-1 çalışmıyor, eski 3.1 kullanılıyor" uyarısı + düzeltme yolu.
- [x] K2 (#4) ✅ ZATEN VARDI (pipeline `_diarize_task`: min_speakers en az 2) Basın modunda en az 2 konuşmacı otomatik (`min_speakers`), kullanıcı sayı girerse o geçerli.
      ✅ 1.9.4 DÜZELTİLDİ: kod 2'yi her modda uyguluyordu (tek kişilik videoda pyannote 4 VBx kişiyi ikiye bölüyordu)
      ve `stt.toml` `min_speakers` kesin sayı gibi davranıyordu. Artık `Pipeline._speaker_bounds`: işe verilen sayı
      kesin, genel min/max sınır, en az 2 yalnız basın modunda.
- [x] K3 (#9) ✅ (gui.html: sarı "Bitti · konuşmacısız"; index'te diarized zaten vardı) Konuşmacısız biten iş görünür olsun: kartta sarı "Bitti · konuşmacısız" + index'te diarized=false.
- [x] K4 (#3) ✅ ("?" ile biten cümlenin sonundaki kısa muhabir parçası ayrı; `merge.QUESTION_MIN_SEC`). AÇIK: noktalamasız ara soru ("what about Saka") hâlâ yutuluyor, K5 ölçümüyle eşik gevşetilsin Kısa soru yutulmasın: cümle içi değişimde "?" ile biten kısa muhabir parçası korunur (önce simülasyon).
- [x] K5 (#1) ✅ `tune.measure` + "Kontrol"de "Konuşmacı doğruluğu" satırı. Düzeltmede programın ilk çıktısı `duzeltme/*.json` "hyp" olarak saklanıyor (eskiden üzerine yazılıp kayboluyordu). Yalnız 1.8.9 sonrası düzeltilen videolar ölçülür. KULLANICIDAN: 3-5 basın toplantısı düzeltsin, Kontrol ekranı görüntüsü.
- [~] K6 DiariZen: KOD YAZILDI (1.9.0), GERÇEKTE DENENMEDİ. `stt/diarizen.py`: `python -m stt diarizen-kur` →
      Python 3.10 (py -3.10 ya da python.org kurucusu, kullanıcı klasörüne) + `.venv-diarizen` + torch 2.1.1 cu121 +
      GitHub zip (`<araç>/diarizen`, pyannote çatalı içinde, git gerekmez). `diar_engine = "diarizen"` (stt.toml) →
      `Diarizer._diarizen`: müzik ayıklanmış/agc'li ses geçici wav → ayrı süreç (ayırma işçisinin CUDA_VISIBLE_DEVICES'ı
      geçer, DiariZen cuda:0 = 3090). Hata → pyannote. Ses imzası YOK (ses kütüphanesi DiariZen'le öğrenmez).
      min/max konuşmacı `pipe.min_speakers/max_speakers` özniteliğiyle (DOĞRULANDI 2026-11: etkili; DiariZen `__call__`
      `self.min_speakers/max_speakers`'ı kümelemeye veriyor, `diarizen/pipelines/inference.py:153-163`). SIRADAKİ: K6b kullanıcıda kurulum denemesi (günlükte "[diarizen kurulum]"),
      K6c ✅ (1.9.2: Ayarlar > Konuşmacı ayırma motoru + 'DiariZen'i kur', `App.diarizen_install`), K6d K5 ölçümüyle community-1 karşılaştırması.
- [x] K7 (#2) ✅ `merge._td_guess`: uzun (≥5 sn) turların toplamı, tek uzun turu olan yarı puan (açılış yapan sözcü); ses kütüphanesi zaten önce geliyordu. Tercüman durumu çözülmedi (dil bilgisi gerekir).
- [x] K8 (#5) ✅ `merge.exclusive_turns`: başka turun tamamen içindeki ≥1 sn söz kazanır (daha kısası "hı hı"). pyannote 4 zaten örtüşmesiz veriyor; bu kural 3.x ve DiariZen çıktısı için.
- [~] K9 (#6) kısmen: sessizlik sonrası erken ilk kelime 1.8.6'da (`EARLY_GAP`). Başka kural ölçümsüz eklenmedi (zarar riski); K5 sonuçlarına göre.
- [x] K10 (#7) ✅ müzik ayıklama zaten yalnız podcast/film'deydi; ek: boş RAM ya da sanal bellek < 4 GB ise atlanır (`Diarizer._low_memory`). Kullanıcıda şu an sanal bellek 3.8 GB boş → sayfa dosyası büyütülene kadar atlanacak.
- [x] K11 (#8) ✅ `VoiceLibrary.identify`: iki ada farkı < 0.05 ve eşik+0.15 altındaysa ad verilmez (`voices.MARGIN/SURE`). Eşiğin ölçümle ayarı K5 verisi gelince.

## Son durum
- 2026-11-06 (kod değişmedi, sürüm aynı 1.9.7): kullanıcı istedi → uçtan uca hata taraması + hedefe (FPL veri seti)
  göre geliştirme planı: **`docs/TARAMA_2026-11-06.md`** (Claude çekirdeği okudu, 3 inceleme ajanı kalan modülleri;
  bulgular kodla doğrulandı, Windows'ta denenmedi). En kritik alan `match.py`/`dataset.py`: "derby"/"nice" takım
  sanılıyor, yanlış rakip fikstür aramasını kapatıyor, "Post-Match" başlığı maç seçiminde kullanılmıyor, 2025-26
  hocaları (Frank, M. Silva...) listede yok → video "Podcast ve Film"e gidiyor, FPL verisi oturumda bir kez
  yükleniyor. Çekirdek: parça planı değişince eski parçalar "bitmiş" sayılıyor (yeniden üretildi), boşluk doldurma
  kapısı AGC yüzünden işlemiyor, başarısız işlerin ses.wav + çerez dosyaları hiç silinmiyor. Arayüz: takip eğitim
  sürerken video kaybediyor, token'sız yerel API CSRF'e açık, K5 "Konuşmacı doğruluğu" muhabir adlarıyla şişiyor
  (K5 ölçümünden ÖNCE düzeltilmeli). Önerilen sıra raporun 0. ve E. bölümünde. SIRADAKİ: kullanıcı onaylarsa A
  bölümü (veri seti) ilk sürüm olarak; her düzeltmeye test (başlık örnekleri raporda).
- 1.9.7 (Claude, Gemini'nin 1.9.6'sının incelemesi; 226 test): 1.9.6 Gemini (başka araç) tarafından yazılmıştı; Claude
  kodla ve uçtan uca (jsdom + gerçek `App.speakers_save`) denedi. Düzeltilen gerilemeler: (1) `merge.py`: ayırma sınırı
  ≥0.8 sn geç kalınca yeni konuşmacının ilk 2-3 kelimesi ("Well, I think") önceki konuşmacıya yazılıyordu (1.9.6'nın
  baştaki-konuşmacı ve virgül bölmesi kuralları); artık baştaki konuşmacı bir önceki konuşmacıysa bölünmez, virgül
  bölmesi yalnız cümle başında. 1.9.6'nın gerçek düzeltmesi (`_smooth` k=0: kısa ilk cümle son kelimesinin kaymasıyla
  tümden sonrakine geçiyordu) duruyor. Bu kurallar ÖLÇÜLMEDİ: K5 verisi gelince bakılsın. (2) "Kaydet ve eğit"ten
  sonra aynı ekranda tekrar Kaydet eski satır numaralarıyla yanlış satırın kelime zamanlarını alıyordu → ekran
  yeniden açılıyor. (3) `keep_duzeltme` adla eşleşiyordu: iki kişinin adı yer değiştirince eski temiz imza yanlış
  kişide kalıyordu → imza çıkarıldığı satırların parmak iziyle (`voices.spans_fp`) saklanıyor, satırlar aynıysa
  korunuyor. (4) `cluster_purity` "Aynı anda" satırlarını iki kez sayıp kümeyi karışık gösteriyordu. (5) Süre
  düzenlemede tam saniyeye yuvarlama (değiştirilmeyen uç da kayıyordu) → ondalıklı; kısaltılan satırın kelimeleri
  satıra kırpılıyor. (6) "+ Yeni kişi" kayıtlı `ELLE_1`'in kimliğini yeniden alıyordu. Not: `voices.split_segment`
  yalnız testlerde; asıl bölme gui.html `splitSeg`. Windows'ta / WebView2'de DENENMEDİ.
- 1.9.6 (kullanıcı geri bildirimi; 218 test): eğitimden önce Konuşmacılar ekranında tam düzeltme yapılabilmesi ve ilk
  cümlenin yutulmaması. (1) `merge.py`: Konuşmacı 1 bir cümle söyleyip ardından Konuşmacı 2 5 cümle söylediğinde ilk
  cümlenin son kelimesindeki kayma yüzünden tüm cümlenin Konuşmacı 2'ye yazılması (`_smooth` `k=0` hatası) ve virgül /
  kısa duraksamayla bağlanan konuşmacı geçişlerinin tek cümlede erimesi (`_sentences`, `_sentence_labels`) düzeltildi.
  (2) `gui.html` / `gui.py` / `voices.py`: Konuşmacılar ekranında her satırın metni doğrudan düzenlenebilir (`.srt`
  dahil güncellenir), `✂ Böl` birleşik satırı imleçten ya da ilk cümle sonundan ikiye böler, `👥 Aynı anda` aynı
  saniyede ikinci konuşmacı satırı ekler (`tune._solo_spans` üst üste konuşulan saniyeleri ses imzasından çıkarır),
  `+ Satır` kaçırılan sözü ekler, `✕` satırı siler, süreye tıklayınca başlangıç-bitiş değiştirilir. (3) `"Kaydet ve
  eğit"` ile çıkarılmış `src="duzeltme"` imzası sonradan düz `"Kaydet"`e basılınca silinmez (`keep_duzeltme`). Windows'ta
  ve gerçek GPU'da DENENMEDİ.
- 1.9.5 (Gemini raporu 0.4/3, kullanıcının 2. adımı; 209 test): ses kütüphanesi. "Kaydet": videonun kayıtları baştan
  kurulur, konuşmasının %10'undan fazlası başka kişiye taşınmış kümenin imzası öğrenilmez (ekranda "Sesi şimdi
  öğrenilmedi: ..."). "Kaydet ve eğit": imzalar düzeltilmiş satırların sesinden yeniden çıkarılır, elle eklenen
  kişiler de öğrenilir (eğitim sonucu "Ses imzaları ... yenilendi"). `sesler.json` kilitli + okunamazsa üzerine
  yazılmaz. Ayrıntı: `docs/KONUSMACI_AYIRMA.md` "1.9.5". Gerçek pyannote modeliyle / Windows'ta DENENMEDİ.
- 1.9.4 (Gemini raporu 0.4/1, kullanıcı istedi; 195 test): en az 2 konuşmacı yalnız basın toplantısında, `stt.toml`
  min/max yalnız sınır; "Konuşmacılar" ekranında gerçek süre; Ayarlar'da OpenAI anahtarı + kayıtlı anahtarları "Sil"
  (ortam değişkenindeki silinemez, ekranda yazar); DiariZen seçiliyken pyannote yalnız yedek gerekince yüklenir, HF
  token şart değil; `requirements.txt` pyannote 4.0.7; `diar_agc` ayarı (Ayarlar > "Konuşmacı ayırmada ses yükseltme");
  çıktı `settings.diar_agc/diar_model`, düzeltme kaydı `diar` + birleştirilmemiş `segs`; eğitim `segs` ile ölçer;
  "Kontrol" farklı ayarlı videoları ayrı gösterir. Arayüz burada jsdom ile denendi (Ayarlar, Konuşmacılar); Windows'ta,
  ekran kartında ve gerçek DiariZen'le DENENMEDİ.
- 2026-11-06 (kod değişmedi, sürüm aynı 1.9.3): kullanıcı istedi → CLAUDE.md "önce kalite, kullanım ikinci planda".
  Gemini raporu (`docs/GEMINI_4_ARGON_BULGULAR.md`) madde madde doğrulandı → raporun 0. bölümü. Raporda olmayan gerçek
  hata: `min_speakers=2` her modda (yukarıda K2 notu). Kullanıcı "sıradaki prompt"u verecek; büyük olasılıkla bu
  bulguların uygulanması → önce 0.4 sırası (güvenli düzeltmeler), eşik/ses değişiklikleri K5 ölçümüyle.
- 1.9.3: ayırma sürerken metin işçisi şeridi "boşta" yerine "metin bitti · konuşmacı ayırma sürüyor". Not: Görev
  Yöneticisi'nin varsayılan grafikleri (3D/Copy/Video) CUDA yükünü göstermez → "%0" yanıltıcı; "Cuda"/"Compute_0" seçilmeli.
- Sürüm 1.8.7: kullanıcı ekranı: 38 dk podcast'te ayırma "CUDA error: out of memory" ile düştü, aynı anda Chrome
  "Not enough memory" diyordu (RAM + sanal bellek dolu → CUDA da OOM verir). Eskiden OOM hiç tekrar denenmiyor,
  "NVIDIA sürücüsünü güncelleyin" deniyordu. Artık OOM = bellek hatası yolu (`pipeline.is_oom`): ayırma süreci
  kapanır, metin bitince 60/120 sn sonra tekrar (2.'de müzik ayıklamasız); sonunda bellek ipucu (sayfa dosyası).
  `wait_memory`: sanal bellek 1 GB altındaysa beklemeye devam. Kullanıcıda ölçüldü: RAM 23.9 (boş 7.7), sanal bellek
  toplam 31.9 (boş 3.8) GB → sayfa dosyası ~8 GB, asıl darboğaz bu. Sayfa dosyası 16–32 GB sabit yapılması önerildi.
- Sürüm 1.8.6 (senaryo turu 3, `tests/test_sim3.py`, 164 test): metin birleştirmede dakikalar arayla söylenen aynı
  cümle artık tek 10 dk'lık segmente dönmüyor (`merge.REPEAT_GAP`); tekrar döngüsü kelimelerden de atılıyor (konuşmacı
  bölününce geri geliyordu); sessizlikten sonra erken zamanlanan ilk kelime sonraki konuşmacıya (`EARLY_GAP`);
  bilinen Whisper uydurmaları atılıyor (`merge.is_hallucination`: Amara.org / Altyazı M.K. her zaman,
  "thanks for watching" yalnız düşük güvende); SRT altyazıları üst üste binmiyor; Windows yasak adları (CON, NUL...)
  `safe_filename`'de; dosya kilidinde (virüs tarayıcı) `util.replace_retry`; hatalı ayar (port "abc"/70000, takip
  aralığı 0/NaN) uygulamayı açılmaz yapmıyor (`config._set/_num`); https'siz / youtube-nocookie / `/v/` linkleri aynı
  video; BOM'lu index.jsonl okunuyor; iş sürerken Windows uykuya geçmiyor (`util.keep_awake`, ekran kapanabilir).
  Windows'ta denenmedi: gece kuyruğunda bilgisayarın uyumadığı, günlükte uyarı olmadığı doğrulansın.
- Sürüm 1.8.5 (senaryo turu 2): "Tekrar dene"de atlanan parça ve başarısız ayırma yeniden yapılır
  (`pipeline._finished`; eskiden boş/hata dosyası "bitti" sayılıyordu); atlanan bölüm kartta gösterilir, linki
  yeniden ekleyince video yeniden işlenir (`_redo_reason`); canlı yayın sürerken sıralı çalışma kapalı (canlı metin
  gecikmesin); eğitim sürerken Başlat engelli, uygulama kapanınca eğitim süreci kapanır.
  Bilinen sınır: iki kartta metin işçisi varsa sıralı çalışma tam değil (ortak görev kuyruğu); kullanıcıda 3080 boş.
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
