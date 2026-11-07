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
**SIRADAKİ (1.9.8 sonrası):** kullanıcı Güncelle → "Veri seti > Yeniden sınıflandır" ekran görüntüsü; sonra rapor
E.2 (C5) ve aşağıdaki K5 planı. **(1.9.5'ten kalan):** kullanıcı Güncelle'ye basıp 3-5 basın toplantısını düzeltecek, "Kontrol" ekranını paylaşacak. Gelince: K5 ölçümüyle K4-açık (merge eşikleri) + agc A/B ("Kontrol" ayara göre ayrı doğruluk gösterir) + K6b/K6d (DiariZen) + SUPlime denemesi. İlk eğitimde günlükte "Ses imzaları düzeltilmiş satırlardan yenilendi" satırı istensin (gerçek modelle denenmedi).
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
- 1.9.23 (Claude; 642 test): kullanıcı: "kanalın search bölümündeki videoları listeleyebilsin", ör.
  `youtube.com/@avfcofficial/search?query=unai%20emery%20press%20conference`. `channel.search_query_of`: kanal içi arama
  (`/@x|channel|c|user/.../search?query=`) ve YouTube araması (`/results?search_query=`) linki. "Kanal · liste"de
  (Linkler'e yapıştırılırsa oraya taşınır) link olduğu gibi listelenir (yt-dlp YoutubeTabIE "search" sekmesi /
  YoutubeSearchURLIE, kaynağında doğrulandı); oynatma listesi gibi işlenmemişlerin hepsi seçili (konu süzgeci yok,
  "Başlıkta geçerse alma" geçerli); kanal taraması ve yapay zekâ seçimi yapılmaz; "Ne arıyorsunuz?" yok sayılır; koleksiyon
  "<kanal> - <arama>". Takip ve `stt kanal` de aynı. DENENMEDİ: gerçek YouTube (bu ortamdan erişim yok), Windows.
- 1.9.22 (Claude; 638 test): kullanıcı ekranı (1.9.21 çalışıyor: üstte "RTX 3090" + "RTX 3080 yardımcı · metin/ayırma"
  = esnek düzen açıldı, "Esnek güç açık", güvenli düzen listesi ✓ monitör Intel'de, ✓ güç 238/245 W). Soru: "MSI'da
  hepsi resetlendi ama satır hâlâ kırmızı". Sebep: uygulama Afterburner'ın içindeki ayarı okuyamaz, yalnız açık olduğunu
  görür. Artık satırda "Reset + Save yaptım" düğmesi (`App.oc_ack`, `cfg.oc_ack` = açık programların adı; başka program
  açılırsa onay geçmez; "Onayı kaldır"). Şeritte esnek kart "esnek · metin / ayırma" yazar. Windows'ta denenmedi.
- 1.9.21 (Claude; kullanıcının 2 isteği; 637 test; bağımsız inceleme 9 hata buldu, hepsi testle düzeltildi — GELISTIRME "1.9.21"): (1) **Kanalın tamamında arama:** "Kanal · liste" > "Ne
  arıyorsunuz?" (takım / kişi / konu, virgülle). Kullanıcı önce "verdiğim takımın erkek A takımının bütün basın
  toplantıları" dedi, sonra "yalnız basın toplantısı değil, her şey: kanaldaki bütün ilgili videolar". Aranan yazılınca
  YouTube kanalının tamamı taranır (kanal içi arama + ilgili listeler + son videolar), uyanlar seçili ve listenin başında
  gelir. Takım adı → yalnız A takımı; konu "Basın toplantısı" → yalnız o takımın toplantıları (rakibinki değil); "Hepsi" →
  takımla ilgili her şey. Takip ve `stt kanal --ara` de aynı. (2) **Esnek güç** (kart şeridinde düğme, varsayılan açık):
  monitör Intel'deyken 3090 ana, 3080 "esnek ikinci kart" (metin ya da konuşmacı ayırma, iş nerede birikmişse); başka
  programlar işlemciyi / RAM'i kullanınca kademeli yavaşlar, bırakınca tam güce döner. Ayarlar > **Güç düzeni**:
  Güvenli (%70, varsayılan) / Dengeli (%85) / Tam (%100) — kullanıcı "max güç" istedi; 1.9.16'daki yeniden başlamalar
  yüzünden varsayılan değişmedi, kullanıcıya seçim ve risk söylendi. Ayrıntı GELISTIRME "1.9.21". DENENMEDİ: Windows,
  gerçek YouTube (bu ortamda ağ yok; testler sahte yt-dlp ile), gerçek kartlar; esnek eşikleri (%35 / %60 / %20) tahmin.
  KULLANICIDAN: Güncelle → bir kanal + takım adıyla "Videoları listele" ekran görüntüsü (kaç video bulundu, yanlış seçilen
  var mı); iş sürerken kart şeridi (3090, 3080 metin / ayırma, Esnek güç) ekran görüntüsü; günlükte `[esnek]` satırları
  (tarayıcı açıkken kademe değişiyor mu, gereksiz yavaşlama var mı).
- 1.9.20 (Claude; 601 test): kullanıcı Intel sürücüsünü kurdu ("Intel display ... enable") ama Ayarlar hâlâ "✗ Monitör RTX 3080 bağlı" diyor. Kod okundu: bu ileti yalnız Intel bağdaştırıcısı Windows'ta GÖRÜNÜP masaüstüne bağlı DEĞİLKEN çıkar = sürücü tamam, kablo büyük olasılıkla hâlâ 3080'de. İleti artık bunu açıkça söylüyor ("sürücü ✓, kalan tek adım kablo"); ikinci monitör/TV durumu ayrı ileti; masaüstü Intel'deyken nvidia-smi'nin yanlış "Enabled"ı 3080'i kilitlemiyor. DENENMEDİ: Windows. KULLANICIDAN: kablo anakarta takılı mı (fotoğraf), Güncelle sonrası Ayarlar listesi ekran görüntüsü. Kablo anakarttayken yine ✗ ise: PowerShell `Get-CimInstance Win32_VideoController | Select Name,CurrentHorizontalResolution` çıktısı istensin.
- 1.9.19 (Claude; 597 test; bağımsız inceleme: ciddi hata yok, 5 küçük bulgu düzeltildi): kullanıcı uyumadan önce: **"200'den fazla video olur mu; bilgisayar çökmesin, GPU az
  harcasın, yavaş ama stabil, MSI Afterburner gibi ama uygulama kendi ayarlasın"**. Cevap: kuyrukta sınır yok; "Kanal ·
  liste"deki "En fazla 200" kutusu 5000'e kadar yazılabilir. Yapılan: **Gece modu** (kart şeridinde düğme + Ayarlar >
  Ekran kartı; kuyruk sürerken de hemen geçerli; ayrıntı GELISTIRME "1.9.19"): tek kart, her görevden sonra ısıya göre
  mola (hedef 72°C, 80°C'de soğuma molası, 66°C'de sürer), işçiler 90 dk'da bir yenilenir, önden 3 video; her modda
  disk koruması (3 GB'tan az boş yer varsa indirme bekler). Fan eğrisi / hız aşırtma yapılmadı (yazılımdan güvenli
  değil; güç + saat sınırı 1.9.17'de var). DENENMEDİ: Windows, gerçek kart. KULLANICIDAN: sabah kart şeridi ekran
  görüntüsü + günlükte `[gece]` ve `[kartlar]` satırları (mola kaç sn'ye çıktı, soğuma molası oldu mu), kaç video bitti.
- 1.9.18 (Claude; 573 test): kullanıcıda işlemci i3-10100 (F değil, iGPU var). BIOS (ASUS TUF, Advanced › System Agent ›
  Graphics Configuration) "iGPU Multi-Monitor" zaten Enabled'dı; Primary Display "CPU Graphics" yapıldı, kablo anakarta
  takılınca açılış logosu göründü ama Windows "takıldı": aygıt "Microsoft Basic Display Adapter" = Intel sürücüsü yok.
  Kablo 3080'e geri takılınca giriş ekranı geldi (Windows sağlam). `engines.igpu_advice` artık bu durumda "Intel
  sürücüsü kurulu değil" diyor (eskiden "BIOS'ta kapalı olabilir"). SIRADAKİ: kullanıcı Intel sürücüsünü kursun
  (Windows Update isteğe bağlı sürücüler / intel.com Driver & Support Assistant), sonra kablo anakarta → Ayarlar listesi.
  Not: kullanıcı komutu PowerShell yerine Komut İstemi'nde (cmd) çalıştırdı; komut verirken "PowerShell'i açın" de.
- 1.9.17 KULLANICIDA (ekran görüntüsü 2026-11-07): "Güvenli güç ayarını uygula" Windows'ta ÇALIŞTI (UAC + nvidia-smi -pl):
  3080 238 W (varsayılanı 340 W), 3090 245 W, saat 1600 MHz; Afterburner kapalı ✓; Önerilen ✓. Eksik tek adım: "Anakart
  görüntü çıkışı (Intel) görünmüyor" → BIOS'ta iGPU kapalı ya da işlemci i3-10100F. Kullanıcıya işlemci adı sorgusu ve
  BIOS adımı verildi. F modelse: ucuz küçük ikinci kart (GT 710/1030) monitör için, ya da tek kart (3090) kalır.
  Henüz doğrulanmadı: yeniden başlatma sonrası görevin sınırı yeniden koyması (listede ✓ kalmalı).
- 1.9.17 (Claude; 572 test; bağımsız inceleme 9 bulgu, hepsi düzeltildi): kullanıcı: **"iki GPU'nun çalıştığı ama bilgisayarda hiçbir problem çıkmayacağı ve ekranın
  kararmayacağı bir düzen kur"**. Yapılan (`stt/power.py`, ayrıntı GELISTIRME "1.9.17"): Ayarlar > Ekran kartı >
  **"İki kart güvenli düzen"** listesi (monitör anakartta mı / güç sınırı / açık Afterburner / kart ayarı Önerilen mi)
  + **"Güvenli güç ayarını uygula"**: tek Windows izniyle iki karta güç sınırı (%70: 3090 245 W, 3080 224 W) + saat
  sınırı 1600 MHz, her açılışta "STT Guvenli Guc" görevi yeniden uygular; "Geri al" kaldırır. Güç sınırı yokken iki kart
  birlikte çalıştırılmaz (yalnız 3090, uyarı; `safe_dual`). Sınır iş sürerken kalkarsa (uyku, sürücü sıfırlanması, Afterburner) görev izinsiz yeniden çalıştırılır, olmazsa tek karta inilir. İş sürerken 5 dk'da bir günlükte `[kartlar]` ısı/güç satırı.
  Kontrol'de "Güç sınırı", "Hız aşırtma", "İki kart düzeni" satırları. Ekran kararmasının çözümü yine monitör kablosunun
  anakarttaki (Intel) çıkışa takılması (yazılımla monitörlü karta güvenli yük verilemez). DENENMEDİ: Windows, UAC,
  nvidia-smi -pl/-lgc, schtasks /XML. KULLANICIDAN: monitör kablosu anakarta → Afterburner Reset+Save → Güncelle →
  Ayarlar > "Güvenli güç ayarını uygula" → listenin ekran görüntüsü; birkaç saat iş sonrası günlükte `[güç]` ve
  `[kartlar]` satırları; yeniden başlama olursa Kontrol ekranı. Arka paneldeki anakart çıkışı yoksa / görüntü gelmezse
  BIOS'ta iGPU (i3-10100F ise iGPU yok: o zaman ucuz ikinci küçük kart ya da tek kart).
- 1.9.16 (Claude; 554 test): kullanıcı "Önerilen"e aldı, **bilgisayar sürekli yeniden başlıyor** (yalnız 3090 yükte).
  Uygulama Windows açılışında kendiliğinden başlamıyor, bekleyen işler "Devam et" istemeden başlamıyor (döngü
  uygulamadan değil). Büyük olasılıkla donanım kararsızlığı: 3090 hız aşırtması (bellek +1441 / çekirdek +143) ya da
  güç kaynağı. Yapılanlar: (1) 1.9.14'te 3090 ayırma toplu boyutu 32'ye çıkmıştı (kartı daha sürekli tam yükte
  tutar) → kararlılık doğrulanana dek yine 8 (`diar_batch` stt.toml'da 32 yapılabilir). (2) `gpuguard`: mavi ekran
  kayıtları (WER-SystemErrorReporting 1001, Kernel-Power 41 BugcheckCode) ve `restart_reason` (0x116 ekran kartı,
  0x124 donanım, kod yoksa güç kesilmesi); STT iş yaparken aralığı (`mark_busy`, `.stt/isleme.json`); açılışta STT
  çalışırken yeniden başladıysa sebebiyle uyarı (`App._crash_check`). Kullanıcıya: STT'yi kapatıp Afterburner Reset +
  Save + "Apply overclocking at system startup" kapalı, Power Limit %75; PowerShell ile son yeniden başlama kayıtları;
  güç kaynağı kaç W. DENENMEDİ: Windows (PowerShell sorgusu dahil).
- 1.9.15 (Claude; 552 test; bağımsız inceleme 5 hata buldu, düzeltildi: RAM beklemesinin 10 dk sınırı sıfırlanıyordu, RAM azken ayırma 3080'e kayıyordu, monitör bilgisi bilinmeyince koruma kapalıydı, eski çökme kuralı motorun yazdığı dosyayla bozuluyordu, 'Hepsi' modunda koruma sonrası 3080 çalışmaya devam ediyordu -> artık ilk görevinde kapatılıyor): kullanıcı: **"ekran tamamen kararıyor ve bir daha açılmıyor, 5 kez üst üste"** (3080
  yardımcı moddayken; 1.9.13). Yapılanlar: (1) koruma işaret dosyası yoksa (1.9.13 işaret yazmıyordu) son 3 günün
  çökmelerini sayar; "Hepsi" ve elle kart numarası da korunur (`gpuguard.display_card_used`); monitörlü kart ana kart
  olarak kullanıldığında da aralık yazılır. (2) Kontrol monitörlü karta CUDA bağlamı açmıyor (CUDA_VISIBLE_DEVICES);
  "Monitör bağlantısı" satırı: anakarttaki Intel çıkışı var mı / monitör nerede (`engines.igpu_advice`). **Kalıcı ve
  stabil çözüm: monitör kablosu anakarttaki (Intel UHD 630) çıkışa** → iki NVIDIA kart da "monitör yok" olur,
  "Önerilen" ikisini de tam güçle kullanır (3080'e metin, 3090'a metin + ayırma). i3-10100'de iGPU var (F değilse);
  BIOS'ta kapalı olabilir. (3) TARAMA B-1: ayırmanın RAM beklemesi ana süreçte (`_diar_mem_ok`), 3090 bu sırada
  metne devam eder. Kullanıcıya: Afterburner Reset, NVIDIA Denetim Masası "CUDA - Sysmem Fallback Policy: Prefer No
  Sysmem Fallback", güç kaynağı kaç W, kararınca bilgisayar açık kalıyor mu. "Normal RAM kullanılabilir mi": model
  hesabı RAM'de yapılırsa i3'te çok yavaş; RAM zaten darboğaz değil (24 GB'ın yarısı boş); ekran kartı belleği
  dolunca RAM'e taşmak (sysmem fallback) yavaşlatır, kapalı tutulmalı. SIRADAKİ: monitör iGPU'ya alınınca iki serbest
  kartta ikinci ayırma işçisi (şu an ayırma yalnız 3090'da; yardımcı mantığı yalnız "yardim" modunda).
- 1.9.14 (Claude; kullanıcının 6 isteği; 542 test; iki araştırma ajanı + bağımsız inceleme (4 hata + 4 küçük, hepsi testle düzeltildi: sıra beklerken kopya video yeniden işleniyordu, canlı yayın uzun kuyrukta başlamıyordu, komut satırı kapanışta asılı kalıyordu, ekran koruması STT kapalıyken olan çökmeyi sayıyordu -> artık yalnız yardımcının açık olduğu aralıklar); rapor
  **`docs/TARAMA_2026-11-07.md`**): (1) oynatma listesinde işlenmemişlerin hepsi seçili (konu süzgeci yok);
  "manager's preview" basın toplantısı sayılır. (2) **Ekran kararması** (2 kez, "HDMI sinyali yok"): Kontrol'de
  "Ekran sürücüsü" satırı (Windows olay günlüğü), yardımcı kart açıkken sürücü çöktüyse yardımcı kendiliğinden kapanır
  (`stt/gpuguard.py`). Kullanıcıya: önce Ayarlar > Önerilen, MSI Afterburner'da Reset (3090: çekirdek +143, bellek
  +1441 aşırtma var; CUDA'da kararsızlık yapabilir). (3) **Kanalda basın toplantısı bulucu (deneme, kapalı)**:
  Kanal · liste > "Kanalın tamamında basın toplantılarını ara". (4) Hız: 3090'da ayırma toplu boyutu 32 (8'di);
  önden hazırlık en çok 6 video (`PREP_AHEAD`; eskiden 100 video önden indiriliyordu, i3 / disk). (5) İndirme:
  çerez okuma yt-dlp'nin `only_once` uyarısıyla tamamen düşüyordu (D1, düzeldi); sabit "Chrome/130" kimliği
  taklitle çelişiyordu (kaldırıldı); Türkçe hata mesajları (giriş / bölge / Cloudflare / ok.ru / Rutube / VK);
  tanınmayan kanal linki bütün kanalı indirmesin (`playlist_items 1`); ok.ru / m.ok.ru / videoembed, Rutube shorts,
  VK video_ext -> vkvideo sayfası (`site_url_variants`); mail.ru gömme / meta; VK / Rutube kanal linkleri; ok.ru /
  mail.ru kanal listesi için açık mesaj. (6) Kartlarda "Hazır · sırada", "Konuşmacılar ayrılıyor", "Metin bitti ·
  konuşmacı ayırma sırada". DENENMEDİ: Windows, gerçek siteler (ağ kapalıydı), PowerShell olay sorgusu, hız.
  **AÇIK (TARAMA_2026-11-07 B):** ayırma RAM beklerken 3090'ı boşta tutuyor (B-1, yüksek etki); C5 K5 ölçümünden
  önce; B1 parça imzası dil/model içermiyor; B3/D4 başarısız işlerin ses.wav / çerezleri silinmiyor; B6 parça
  json'larını sakla; yt-dlp nightly kararı (kullanıcıya sorulmalı); gizli tarayıcıda alan başına çerez ve
  kullanıcının çerezleri; 3090'da metin + ayırmayı aynı anda (ölçerek). KULLANICIDAN: Kontrol ekran görüntüsü
  ("Ekran sürücüsü" satırı), Afterburner Reset sonrası kararma olup olmadığı, bulucu denemesi.
- 1.9.13 (Claude; kullanıcı iki kez istedi; 510 test; bağımsız inceleme 4 hata buldu, hepsi testle düzeltildi: kalıcı düşen rol öbürünü kilitliyordu, yer yokken nvidia-smi her turda, bellek aşan görev 3080'e tekrar tekrar (artık `no_helper`), ana ayırma açılırken 3080 gereksiz açılıyordu (15 sn + ana işçi hazır)): **yardımcı kart (3080) konuşmacı ayırmaya da geçer.** Ekranda
  videolar "3/3" metin bitmiş, ayırma bekliyordu (darboğaz 3090'daki tek ayırma süreci). 3080 aynı anda tek iş yapar:
  metin ya da ayırma; hangisinde iş birikmişse (20 sn'den uzun) ona geçer. Ayrıntı GELISTIRME "1.9.13". DENENMEDİ:
  Windows, gerçek kartlar (3 GB ayırma belleği tahmin; pyannote'un 3080'de hızı bilinmiyor). KULLANICIDAN: Güncelle →
  kart şeridi (4 kutu: 3090, 3080 metin, 3080 ayırma, Konuşmacı ayırma) ekran görüntüsü; günlükte "[diar2]" satırları.
- 1.9.12 (Claude; 495 test): (1) **DiariZen kurulumu Windows'ta "pesq" derlemesinde düşüyordu** (C derleyicisi yok).
  DiariZen kaynağı indirilip tarandı: pesq / pystoi hiç import edilmiyor (yalnız eğitim/ölçüm). `diarizen.SKIP_REQS`
  (pesq, pystoi, jupyterlab, pre-commit) çıkarılıp `requirements-stt.txt` ile kuruluyor. Windows'ta DENENMEDİ; başka
  derleme isteyen paket çıkarsa günlükteki "[diarizen kurulum]" satırı istensin. (2) **Konuşmacısız çevirme:**
  "Konuşmacıları ayır" anahtarı zaten vardı ama her açılışta token varsa yeniden açılıyordu; seçim artık hatırlanıyor
  (`ui_diarize`), açıklaması yazıldı. (3) Kullanıcının ekranı: 4 video "Metne çevriliyor 3/3" = metin bitti, konuşmacı
  ayırma bekliyor → **darboğaz konuşmacı ayırma**, yardımcı 3080 doğru olarak boşta. SIRADAKİ: konuşmacı ayırmanın
  ikinci işçisi 3080'de (kullanıcı iki kez istedi). Tasarım notu: 3080 aynı anda tek rol (metin YA DA ayırma; ikisi
  belleğe sığmaz, aynı kartta da sırayla çalışmak zorunda); "diar" tek işçi varsayımı pipeline'da çok yerde
  (dqueued_n, inflight["diar"], cancel, _drop_dead_diar, _on_msg hata yolları).
- 1.9.11 (Claude; kullanıcı istedi; 493 test; bağımsız inceleme 1 hata buldu, düzeltildi: başka program yüzünden toplu boyut kalıcı küçülüyordu): yardımcı kart (3080) artık "hafif" değil: kullanıcının önerisiyle kartta YouTube /
  sekmeler için **ayrılan bellek** (Ayarlar > "Yardımcı kartta YouTube / tarayıcı için boş bırakılacak bellek",
  varsayılan 2 GB, kaydedince hemen geçerli) dışında kalan bellek ve güç işe verilir: toplu boyut kalan yere göre
  (8/4/2), dinlenme ve kullanım-yüzdesi beklemesi yok. Başka programlar belleği alırsa bekler / kapanır, sonra daha
  küçük toplu boyutla açılır. Ayrıntı GELISTIRME "1.9.11". Kullanıcıya söylendi: GPU hesap gücü bölünemez, YouTube
  ayrı video çözücüyü kullanır. DENENMEDİ: Windows, gerçek kartlar (toplu boyut bellek tahminleri README'den, ölçülmedi).
  KULLANICIDAN: Güncelle → yardımcı + ayrılan bellek → kart şeridi + Görev Yöneticisi 3080 "Dedicated GPU memory".
- 1.9.10 (Claude; 488 test): kullanıcı 1.9.9'da Ayarlar > Ekran kartı kullanımında yalnız "Önerilen" ve "Hepsi"
  gördü ("yardımcı" ve "Yalnızca RTX ..." yok). Sebep (eskiden de vardı): Ayarlar uygulama açılışında (`hello`)
  dolduruluyordu, o anda motor başlamadığı için `App._gpus` boş → kart listesi boş. Düzeltme: `get_settings` motor
  hazır değilse kartları `gpu_info`'dan (nvidia-smi) okur; Ayarlar her açılışta `get_settings` ile tazelenir.
  Chromium'da denendi (önce 2 seçenek, sonra 5); Windows'ta denenmedi. KULLANICIDAN: Güncelle → yardımcı modu →
  kart şeridi ekran görüntüsü.
- 1.9.9 (Claude; kullanıcı istedi; 487 test; bağımsız inceleme ajanı 6 bulgu, hepsi düzeltildi): **yeni kart modu "yardımcı"** (`gpus = "yardim"`, Ayarlar > Ekran kartı
  kullanımı > "RTX 3090 + RTX 3080 yardımcı"). Kullanıcı 125 videoluk kuyrukta 3090'ın yetmediğini, 3080'in de
  "YouTube'a yer bırakarak" çalışmasını istedi. 3080 yalnız fazla iş varken (özellikle 3090 konuşmacı ayırırken, o
  sırada 3090'ın metin işçisi bekliyor) metin çevirir; toplu boyut 4, en düşük işlemci önceliği, her parçadan sonra
  2 sn dinlenme + nvidia-smi ölçümü; başka program kartı %50 üstü kullanıyorsa ya da 1.5 GB'tan az boş bellek varsa
  bekler, 2 dk sürerse 3080 işçisi kapanır (bellek boşalır), kart boşalınca açılır; baştan kapalı, 5 dk gerekmezse
  yine kapanır. ESKİ HATA DA DÜZELDİ: konuşmacı ayırma süreci bellek/CUDA hatasından sonra yeniden açılınca kuyruk
  kilidi ölü süreçte kalıp ayırma sonsuza dek bekleyebiliyordu (GELISTIRME "1.9.9"). Konuşmacı ayırma 3080'e
  GİTMEZ. Ayrıntı GELISTIRME.md "1.9.9". Ekran görüntüsündeki Görev Yöneticisi "3D %0" CUDA işini göstermez
  (grafik açılır menüsünden "Cuda"/"Compute_0" seçilmeli) — kullanıcıya söylendi. Kullanıcı "konuşmacı ayırma modu
  da ekle" dedi: ikinci ayırma işçisinin 3080'de çalışması YAPILMADI (aynı kartta ayırma + metin sırayla çalışmak
  zorunda; yardımcı zaten ayırma sırasında metni üstleniyor, kazanç az, risk yüksek) — kullanıcıya açıklandı,
  denedikten sonra hâlâ isterse yapılacak. DENENMEDİ: Windows, gerçek kartlar (nvidia-smi utilization'ın WDDM'de
  okunması, IDLE önceliğin hızı, YouTube akıcılığı). KULLANICIDAN: Güncelle → modu seç → kart şeridi + Görev
  Yöneticisi (Cuda grafiği) ekran görüntüsü; YouTube'da takılma olursa söylesin (toplu boyut 2'ye indirilebilir).
- 1.9.8 (Claude; `docs/TARAMA_2026-11-06.md` A bölümü = veri seti eşleştirmesi; 465 test): kullanıcı "A bölümünü
  yap" dedi. Rapordaki başlık örneklerinin hepsi `tests/test_v198.py`'de; listedeki her hoca ve her takım adı toplu
  deneniyor. Dört tur bağımsız inceleme ajanı (12 + 10 + 8 + 9 bulgu, hepsi testle gösterilip düzeltildi). En
  ciddileri: yeniden sınıflandırma Windows'ta yalnız harf büyüklüğü değişen adda yeni yazdığı dosyayı silebiliyordu;
  ertelenmiş silme sonradan düzeltilmiş / aynı adı almış dosyayı silebiliyordu (artık boyut + değişiklik zamanı +
  sahiplik denetimi); "Durdur" ile aynı anda basılınca arayüz kilitlenebiliyordu (kilit sırası). Yapılanlar:
  (1) `match.py`: "derby / nice / Sporting / sabah / Zira / om" tek başına takım değil (niteleyiciyle ya da başka
  takıma bağlıyken: "Leeds v Derby", "Qarabağ - Sabah", "Sporting will be..."); NBA bağlamında "Spurs" Tottenham
  değil; "Arteta on Saka's fitness" rakip değil; McKenna, O'Neil, "ERIK TEN HAG", "SLOT'S", Türkçe ekler; aynı
  soyadlı başka kişi (Andreas Pereira, Nuno Mendes, Derrick Rose) hoca değil; ev-deplasman sırası yalnız "A vs B /
  A - B / A 2-1 B"de; "Manchester Utd", "Sheffield Utd", tireli adlar. **Hoca listesi 2026-27'ye göre yenilendi**
  (web aramasıyla doğrulandı, 2026-11-06'da sezonda görevden alınan yoktu): Arteta, Emery, Rose (Bournemouth),
  Andrews, Hürzeler, Xabi Alonso (Chelsea), Lampard (Coventry), Sage (Palace), Moyes, Arbeloa (Fulham), Jakirović
  (Hull), O'Neil (Ipswich), Farke, Iraola (Liverpool), Maresca (Man City), Carrick (Man Utd), Jaissle (Newcastle),
  Glasner (Forest), Le Bris, De Zerbi (Spurs). Hoca → takım yalnız tarafsız kanalda ve 2026-07-01..2027-06-30
  tarihli videoda. Kulübün kendi kanalı + başlıkta tanınan rakip → otomatikte basın toplantısı (taraftar / gazete /
  podcast kanalı, özet, canlı yayın, forma tanıtımı hariç).
  (2) `dataset.py`: maç öncesi / sonrası başlıktan (`phase_hint`): "Post-Match / reacts / Arsenal 2-1 Chelsea"
  yalnız oynanmış maça, "Pre-Match / previews / pre-Chelsea" yalnız oynanacak maça; yalnız gün bilinen tarihte maç
  günü = maç sonrası. Sözlükte olmayan rakip (Wycombe) = kupa maçı, PL haftasına yazılmaz ("Manchester" gibi
  örtüşen ad o maçı bulur); açıklamadaki kupa adı o rakiple PL maçını ezmez. Yedek PL listesi 2026-27 (Coventry,
  Ipswich, Hull çıktı; Wolves, Burnley, West Ham düştü). FPL'ye ulaşılamazsa 10 dk sonra tekrar (eskiden oturumda
  bir kez → günlerce GW'siz). index'e fixture_id, kickoff_utc, is_home, fpl_short, opponent_short, score,
  speaker_role, speaker_source, diar_model, title; yollar "/" ile.
  (3) Yeni **"Veri seti > Yeniden sınıflandır"** (`stt yeniden-siniflandir [--dene]`, `stt/reclassify.py`): metni
  yeniden çevirmeden eski çıktıları doğru klasöre taşır, mac.json (GW, skor), index, durum kaydı, koleksiyonlar
  yenilenir. Önce ne değişeceğini gösterir, onayla uygular. FPL yoksa çalışmaz; eski sezona, Podcast ve Film'e ve
  mac.json'da "ek_bilgiler" dolu klasöre dokunmaz. Elle düzeltilmiş .txt gövdesi / .srt korunur; açık (kilitli)
  dosyada kayıt atlanır ya da silme sonraki çalıştırmaya kalır (`.stt/siniflandirma_bekleyen.json`). Sınıflandırma
  sürerken Başlat / Tekrar dene / Devam et / Konuşmacılar Kaydet bekler; kanal takibi bu sırada videoları "görüldü"
  saymaz (TARAMA C1'in bu kısmı düzeldi). Arayüz Chromium'da (köprüyle), komut satırı uçtan uca denendi.
  DENENMEDİ: Windows / WebView2, gerçek FPL verisi (bu ortamdan erişim kapalı; testler sahte fikstürle), gerçek
  kullanıcı veri seti. KULLANICIDAN: Güncelle → Veri seti → "Yeniden sınıflandır" → çıkan pencerenin ekran görüntüsü
  (uygulamadan önce). AÇIK KALAN: A-E'deki yapılandırılmış sakatlık/kadro JSON'u (E.4); yeni GW bulunan eski
  kayıtlar için FPL presser.py otomatik çalıştırılmıyor. SIRADAKİ (rapor E sırası): C5 (K5 ölçümünden önce), sonra
  kullanıcının düzelttiği toplantılarla eşik/agc/DiariZen; B1, B2, B3, D1, D4, D6, D7.
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
